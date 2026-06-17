import argparse
import ast
import csv
import os
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


PAD = "<pad>"
UNK = "<unk>"
SPAN_LABELS = {"O": 0, "A": 1, "OPE": 2}
SENTIMENT_LABELS = {"NEG": 0, "NEU": 1, "POS": 2}
ID_TO_SENTIMENT = {0: "NEG", 1: "NEU", 2: "POS"}
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


def read_aste(path: Path) -> List[Dict]:
    examples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sentence, _, triplet_text = line.partition("#### #### ####")
        tokens = sentence.strip().split()
        triplets = ast.literal_eval(triplet_text.strip())
        normalized = []
        for aspect, opinion, sentiment in triplets:
            normalized.append((tuple(aspect), tuple(opinion), sentiment))
        examples.append({"tokens": tokens, "triplets": normalized})
    return examples


def build_vocab(datasets: Sequence[List[Dict]]) -> Dict[str, int]:
    vocab = {PAD: 0, UNK: 1}
    for data in datasets:
        for ex in data:
            for token in ex["tokens"]:
                token = token.lower()
                if token not in vocab:
                    vocab[token] = len(vocab)
    return vocab


def enumerate_spans(length: int, max_span_width: int) -> List[Tuple[int, int]]:
    spans = []
    for start in range(length):
        for end in range(start, min(length, start + max_span_width)):
            spans.append((start, end))
    return spans


def indices_to_span(indices: Tuple[int, ...]) -> Tuple[int, int]:
    return min(indices), max(indices)


class ASTESpanDataset(Dataset):
    def __init__(self, examples: List[Dict], vocab: Dict[str, int], max_len: int = 96, max_span_width: int = 4):
        self.examples = examples
        self.vocab = vocab
        self.max_len = max_len
        self.max_span_width = max_span_width

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        ex = self.examples[idx]
        tokens = ex["tokens"][: self.max_len]
        ids = [self.vocab.get(t.lower(), self.vocab[UNK]) for t in tokens]
        spans = enumerate_spans(len(tokens), self.max_span_width)
        span_labels = [SPAN_LABELS["O"] for _ in spans]
        sentiment_labels = [-100 for _ in spans]
        span_to_id = {span: i for i, span in enumerate(spans)}
        pairs = []
        gold_triplets = set()
        for aspect_idx, opinion_idx, sentiment in ex["triplets"]:
            if max(aspect_idx + opinion_idx) >= len(tokens):
                continue
            aspect_span = indices_to_span(aspect_idx)
            opinion_span = indices_to_span(opinion_idx)
            if aspect_span in span_to_id:
                span_labels[span_to_id[aspect_span]] = SPAN_LABELS["A"]
                sentiment_labels[span_to_id[aspect_span]] = SENTIMENT_LABELS[sentiment]
            if opinion_span in span_to_id:
                span_labels[span_to_id[opinion_span]] = SPAN_LABELS["OPE"]
            if aspect_span in span_to_id and opinion_span in span_to_id:
                pairs.append((span_to_id[aspect_span], span_to_id[opinion_span], SENTIMENT_LABELS[sentiment]))
                gold_triplets.add((aspect_span, opinion_span, SENTIMENT_LABELS[sentiment]))
        return {
            "ids": ids,
            "spans": spans,
            "span_labels": span_labels,
            "sentiment_labels": sentiment_labels,
            "pairs": pairs,
            "gold_triplets": sorted(gold_triplets),
        }


def collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    batch_size = len(batch)
    max_len = max(len(x["ids"]) for x in batch)
    max_spans = max(len(x["spans"]) for x in batch)
    ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    spans = torch.zeros(batch_size, max_spans, 2, dtype=torch.long)
    span_mask = torch.zeros(batch_size, max_spans, dtype=torch.bool)
    span_labels = torch.full((batch_size, max_spans), -100, dtype=torch.long)
    sentiment_labels = torch.full((batch_size, max_spans), -100, dtype=torch.long)
    gold_triplets = []

    for i, item in enumerate(batch):
        length = len(item["ids"])
        n_spans = len(item["spans"])
        ids[i, :length] = torch.tensor(item["ids"], dtype=torch.long)
        mask[i, :length] = True
        spans[i, :n_spans] = torch.tensor(item["spans"], dtype=torch.long)
        span_mask[i, :n_spans] = True
        span_labels[i, :n_spans] = torch.tensor(item["span_labels"], dtype=torch.long)
        sentiment_labels[i, :n_spans] = torch.tensor(item["sentiment_labels"], dtype=torch.long)
        gold_triplets.append(item["gold_triplets"])
    return {
        "ids": ids,
        "mask": mask,
        "spans": spans,
        "span_mask": span_mask,
        "span_labels": span_labels,
        "sentiment_labels": sentiment_labels,
        "gold_triplets": gold_triplets,
    }


class SpanASTEModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 200, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder = nn.LSTM(embed_dim, hidden_dim // 2, batch_first=True, bidirectional=True)
        self.span_width = nn.Embedding(8, hidden_dim)
        self.span_mlp = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.span_classifier = nn.Linear(hidden_dim, len(SPAN_LABELS))
        self.sentiment_classifier = nn.Linear(hidden_dim, len(SENTIMENT_LABELS))

    def span_representations(self, encoded: torch.Tensor, spans: torch.Tensor) -> torch.Tensor:
        batch = torch.arange(encoded.size(0), device=encoded.device).unsqueeze(1)
        start = spans[..., 0]
        end = spans[..., 1]
        start_vec = encoded[batch, start]
        end_vec = encoded[batch, end]
        width = (end - start).clamp(max=7)
        width_vec = self.span_width(width)
        return self.span_mlp(torch.cat([start_vec, end_vec, width_vec], dim=-1))

    def forward(self, ids: torch.Tensor, spans: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.embedding(ids)
        encoded, _ = self.encoder(emb)
        span_repr = self.span_representations(encoded, spans)
        span_logits = self.span_classifier(span_repr)
        sentiment_logits = self.sentiment_classifier(span_repr)
        return span_logits, sentiment_logits


def masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, weight: torch.Tensor = None) -> torch.Tensor:
    valid = targets != -100
    if not valid.any():
        return logits.sum() * 0.0
    return nn.functional.cross_entropy(logits[valid], targets[valid], weight=weight)


def loss_fn(outputs, batch) -> torch.Tensor:
    span_logits, sentiment_logits = outputs
    device = span_logits.device
    span_weight = torch.tensor([0.08, 1.0, 1.0], device=device)
    span_loss = masked_cross_entropy(span_logits, batch["span_labels"], span_weight)
    sentiment_loss = masked_cross_entropy(sentiment_logits, batch["sentiment_labels"])
    return span_loss + 0.8 * sentiment_loss


@torch.no_grad()
def evaluate(model, loader, device, span_threshold: float = 0.30, max_opinions_per_aspect: int = 2) -> Tuple[float, float, float]:
    model.eval()
    pred_total = 0
    gold_total = 0
    correct = 0
    for batch in loader:
        gold_triplets = batch["gold_triplets"]
        device_batch = {k: v.to(device) for k, v in batch.items() if k != "gold_triplets"}
        span_logits, sentiment_logits = model(device_batch["ids"], device_batch["spans"])
        span_prob = torch.softmax(span_logits, dim=-1)
        sentiment_pred = sentiment_logits.argmax(-1)
        for b in range(span_prob.size(0)):
            pred = set()
            n = int(device_batch["span_mask"][b].sum().item())
            spans = device_batch["spans"][b]
            aspect_ids = [
                i for i in range(n)
                if span_prob[b, i, SPAN_LABELS["A"]].item() >= span_threshold
            ]
            opinion_ids = [
                i for i in range(n)
                if span_prob[b, i, SPAN_LABELS["OPE"]].item() >= span_threshold
            ]
            for a_id in aspect_ids:
                a_start, a_end = spans[a_id].tolist()
                a_center = (a_start + a_end) / 2
                ranked_opinions = sorted(
                    opinion_ids,
                    key=lambda o_id: abs(((spans[o_id, 0] + spans[o_id, 1]).item() / 2) - a_center),
                )
                for o_id in ranked_opinions[:max_opinions_per_aspect]:
                    pred.add((tuple(spans[a_id].tolist()), tuple(spans[o_id].tolist()), int(sentiment_pred[b, a_id].item())))
            gold = set(gold_triplets[b])
            pred_total += len(pred)
            gold_total += len(gold)
            correct += len(pred & gold)
    precision = correct / max(1, pred_total)
    recall = correct / max(1, gold_total)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    return precision, recall, f1


def train_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total = 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "gold_triplets"}
        optimizer.zero_grad()
        loss = loss_fn(model(batch["ids"], batch["spans"]), batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total += loss.item()
    return total / max(1, len(loader))


def run_dataset(args, dataset_name: str, device: torch.device, result_dir: Path) -> Dict:
    data_dir = resolve_data_dir(args.data_dir) / dataset_name
    train_data = read_aste(data_dir / "train.txt")
    dev_data = read_aste(data_dir / "dev.txt")
    test_data = read_aste(data_dir / "test.txt")
    vocab = build_vocab([train_data, dev_data])
    train_loader = DataLoader(ASTESpanDataset(train_data, vocab, args.max_len, args.max_span_width), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(ASTESpanDataset(dev_data, vocab, args.max_len, args.max_span_width), batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(ASTESpanDataset(test_data, vocab, args.max_len, args.max_span_width), batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    model = SpanASTEModel(len(vocab), args.embed_dim, args.hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"Dataset={dataset_name} Train={len(train_data)} Dev={len(dev_data)} Test={len(test_data)} Vocab={len(vocab)}")
    best_dev = -1.0
    best_state = None
    rows = []
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, device)
        p, r, f1 = evaluate(model, dev_loader, device, args.span_threshold, args.max_opinions_per_aspect)
        if f1 > best_dev:
            best_dev = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        rows.append({"dataset": dataset_name, "split": "dev", "epoch": epoch, "loss": f"{loss:.4f}", "precision": f"{p:.4f}", "recall": f"{r:.4f}", "f1": f"{f1:.4f}"})
        print(f"[{dataset_name}] epoch={epoch} loss={loss:.4f} dev_p={p:.4f} dev_r={r:.4f} dev_f1={f1:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    p, r, f1 = evaluate(model, test_loader, device, args.span_threshold, args.max_opinions_per_aspect)
    test_row = {"dataset": dataset_name, "split": "test", "epoch": "best_dev", "loss": "", "precision": f"{p:.4f}", "recall": f"{r:.4f}", "f1": f"{f1:.4f}"}
    rows.append(test_row)
    print(f"[{dataset_name}] TEST precision={p:.4f} recall={r:.4f} f1={f1:.4f}")
    write_csv(result_dir / f"task4_span_aste_{dataset_name}.csv", rows)
    print(f"Saved results to {result_dir / f'task4_span_aste_{dataset_name}.csv'}")
    return test_row


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 4: span-based aspect sentiment triplet extraction.")
    parser.add_argument("--data_dir", default="data/ASTE-Data-V2-EMNLP2020/data/triplet_data")
    parser.add_argument("--datasets", nargs="+", default=["14lap", "14res", "15res", "16res"])
    parser.add_argument("--max_len", type=int, default=60)
    parser.add_argument("--max_span_width", type=int, default=3)
    parser.add_argument("--embed_dim", type=int, default=200)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--span_threshold", type=float, default=0.30)
    parser.add_argument("--max_opinions_per_aspect", type=int, default=2)
    parser.add_argument("--result_dir", default="result")
    args = parser.parse_args()

    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result_dir = resolve_result_dir(args.result_dir)
    print(f"Device={device}")
    summary_rows = []
    for dataset_name in args.datasets:
        summary_rows.append(run_dataset(args, dataset_name, device, result_dir))
    write_csv(result_dir / "task4_span_aste_summary.csv", summary_rows)


if __name__ == "__main__":
    main()
