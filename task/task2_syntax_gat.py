import argparse
import csv
import os
import pickle
import random
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


LABEL_TO_ID = {-1: 0, 0: 1, 1: 2}
PAD = "<pad>"
UNK = "<unk>"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_data_dir(path: str) -> Path:
    data_dir = Path(path)
    if data_dir.is_absolute() or data_dir.exists():
        return data_dir
    return PROJECT_ROOT / data_dir


def resolve_result_dir(path: str) -> Path:
    result_dir = Path(path)
    if not result_dir.is_absolute():
        result_dir = PROJECT_ROOT / result_dir
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_semeval_seg(path: Path) -> List[Dict]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    examples = []
    for i in range(0, len(lines), 3):
        template, aspect, polarity = lines[i], lines[i + 1], int(lines[i + 2])
        sentence = template.replace("$T$", aspect)
        examples.append(
            {
                "tokens": sentence.lower().split(),
                "aspect_tokens": aspect.lower().split(),
                "label": LABEL_TO_ID[polarity],
            }
        )
    return examples


def load_graphs(path: Path) -> List[torch.Tensor]:
    with path.open("rb") as f:
        graphs = pickle.load(f)
    # The original graph dict uses source line numbers as keys: 0, 3, 6, ...
    # Sorting converts it back to sample order so Dataset indices are stable.
    return [torch.tensor(graphs[k], dtype=torch.float) for k in sorted(graphs, key=int)]


def build_vocab(examples: List[Dict], min_freq: int = 1) -> Dict[str, int]:
    freq: Dict[str, int] = {}
    for ex in examples:
        for token in ex["tokens"]:
            freq[token] = freq.get(token, 0) + 1
    vocab = {PAD: 0, UNK: 1}
    for token, count in sorted(freq.items()):
        if count >= min_freq:
            vocab[token] = len(vocab)
    return vocab


class SyntaxDataset(Dataset):
    def __init__(self, examples: List[Dict], graphs: List[torch.Tensor], vocab: Dict[str, int], max_len: int = 96):
        self.examples = examples
        self.graphs = graphs
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        ex = self.examples[idx]
        ids = [self.vocab.get(t, self.vocab[UNK]) for t in ex["tokens"]][: self.max_len]
        graph = self.graphs[idx][: self.max_len, : self.max_len]
        return {"ids": ids, "graph": graph, "label": ex["label"]}


def collate(batch: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = max(len(x["ids"]) for x in batch)
    ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    adj = torch.zeros(len(batch), max_len, max_len, dtype=torch.float)
    labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)
    for i, item in enumerate(batch):
        length = len(item["ids"])
        ids[i, :length] = torch.tensor(item["ids"], dtype=torch.long)
        mask[i, :length] = True
        adj[i, :length, :length] = item["graph"][:length, :length]
        adj[i, torch.arange(length), torch.arange(length)] = 1.0
    return ids, mask, adj, labels


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.3):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.src = nn.Linear(out_dim, 1, bias=False)
        self.dst = nn.Linear(out_dim, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.linear(x)
        scores = self.src(h) + self.dst(h).transpose(1, 2)
        scores = self.leaky_relu(scores)
        scores = scores.masked_fill(adj <= 0, -1e9)
        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        return torch.bmm(attention, h)


class SyntaxGATClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 200, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.bilstm = nn.LSTM(embed_dim, hidden_dim // 2, batch_first=True, bidirectional=True)
        self.gat1 = GraphAttentionLayer(hidden_dim, hidden_dim, dropout)
        self.gat2 = GraphAttentionLayer(hidden_dim, hidden_dim, dropout)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 3))

    def forward(self, ids: torch.Tensor, mask: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(ids)
        lstm_out, _ = self.bilstm(emb)
        h = torch.relu(self.gat1(lstm_out, adj))
        h = torch.relu(self.gat2(h, adj))
        h = h.masked_fill(~mask.unsqueeze(-1), 0.0)
        pooled = h.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
        return self.classifier(pooled)


def metrics(preds: List[int], golds: List[int]) -> Tuple[float, float]:
    acc = sum(p == g for p, g in zip(preds, golds)) / max(1, len(golds))
    f1s = []
    for label in range(3):
        tp = sum(p == label and g == label for p, g in zip(preds, golds))
        fp = sum(p == label and g != label for p, g in zip(preds, golds))
        fn = sum(p != label and g == label for p, g in zip(preds, golds))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1s.append(2 * precision * recall / max(1e-8, precision + recall))
    return acc, sum(f1s) / 3


def train_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total = 0.0
    loss_fn = nn.CrossEntropyLoss()
    for ids, mask, adj, labels in loader:
        ids, mask, adj, labels = ids.to(device), mask.to(device), adj.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(ids, mask, adj)
        loss = loss_fn(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total += loss.item()
    return total / max(1, len(loader))


@torch.no_grad()
def evaluate(model, loader, device) -> Tuple[float, float]:
    model.eval()
    preds, golds = [], []
    for ids, mask, adj, labels in loader:
        logits = model(ids.to(device), mask.to(device), adj.to(device))
        preds.extend(logits.argmax(-1).cpu().tolist())
        golds.extend(labels.tolist())
    return metrics(preds, golds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 2: syntactic feature representation with dependency graph and GAT.")
    parser.add_argument("--data_dir", default="data/SemEval-2014-Task4-Laptop")
    parser.add_argument("--max_len", type=int, default=96)
    parser.add_argument("--embed_dim", type=int, default=200)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--result_dir", default="result")
    args = parser.parse_args()

    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = resolve_data_dir(args.data_dir)
    result_dir = resolve_result_dir(args.result_dir)
    train_examples = read_semeval_seg(data_dir / "Laptops_Train.xml.seg")
    test_examples = read_semeval_seg(data_dir / "Laptops_Test_Gold.xml.seg")
    train_graphs = load_graphs(data_dir / "Laptops_Train.xml.seg.graph")
    test_graphs = load_graphs(data_dir / "Laptops_Test_Gold.xml.seg.graph")
    vocab = build_vocab(train_examples)

    train_loader = DataLoader(SyntaxDataset(train_examples, train_graphs, vocab, args.max_len), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(SyntaxDataset(test_examples, test_graphs, vocab, args.max_len), batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    model = SyntaxGATClassifier(len(vocab), args.embed_dim, args.hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"Train={len(train_examples)} Test={len(test_examples)} Vocab={len(vocab)} Device={device}")
    rows = []
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, device)
        acc, f1 = evaluate(model, test_loader, device)
        rows.append({"model": "Syntax-GAT", "epoch": epoch, "loss": f"{loss:.4f}", "accuracy": f"{acc:.4f}", "macro_f1": f"{f1:.4f}"})
        print(f"[Syntax-GAT] epoch={epoch} loss={loss:.4f} acc={acc:.4f} macro_f1={f1:.4f}")
    write_csv(result_dir / "task2_syntax_gat.csv", rows)
    if rows:
        best = max(rows, key=lambda row: float(row["macro_f1"]))
        write_csv(result_dir / "task2_syntax_gat_summary.csv", [best])
    print(f"Saved results to {result_dir / 'task2_syntax_gat.csv'}")


if __name__ == "__main__":
    main()
