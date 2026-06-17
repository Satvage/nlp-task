import argparse
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


LABEL_TO_ID = {-1: 0, 0: 1, 1: 2}
ID_TO_LABEL = {0: "negative", 1: "neutral", 2: "positive"}
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
        tokens = sentence.lower().split()
        aspect_tokens = aspect.lower().split()
        examples.append(
            {
                "template": template,
                "sentence": sentence,
                "tokens": tokens,
                "aspect": aspect.lower(),
                "aspect_tokens": aspect_tokens,
                "label": LABEL_TO_ID[polarity],
            }
        )
    return examples


def build_vocab(examples: List[Dict], min_freq: int = 1) -> Dict[str, int]:
    freq: Dict[str, int] = {}
    for ex in examples:
        for token in ex["tokens"] + ex["aspect_tokens"]:
            freq[token] = freq.get(token, 0) + 1
    vocab = {PAD: 0, UNK: 1}
    for token, count in sorted(freq.items()):
        if count >= min_freq:
            vocab[token] = len(vocab)
    return vocab


def load_glove(path: str, vocab: Dict[str, int], embed_dim: int) -> torch.Tensor:
    matrix = torch.empty(len(vocab), embed_dim).uniform_(-0.05, 0.05)
    matrix[vocab[PAD]].zero_()
    if not path:
        print("No GloVe file provided. Use trainable random embeddings for the GloVe baseline.")
        return matrix
    glove_path = Path(path)
    if not glove_path.exists():
        print(f"GloVe file not found: {glove_path}. Use random embeddings instead.")
        return matrix
    loaded = 0
    with glove_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip().split()
            if len(parts) != embed_dim + 1:
                continue
            word = parts[0]
            if word in vocab:
                matrix[vocab[word]] = torch.tensor([float(x) for x in parts[1:]], dtype=torch.float)
                loaded += 1
    print(f"Loaded {loaded}/{len(vocab)} GloVe vectors.")
    return matrix


class SemEvalDataset(Dataset):
    def __init__(self, examples: List[Dict], vocab: Dict[str, int], max_len: int = 96):
        self.examples = examples
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        ex = self.examples[idx]
        ids = [self.vocab.get(t, self.vocab[UNK]) for t in ex["tokens"]][: self.max_len]
        aspect_ids = [self.vocab.get(t, self.vocab[UNK]) for t in ex["aspect_tokens"]][: self.max_len]
        return {
            "ids": ids,
            "aspect_ids": aspect_ids,
            "label": ex["label"],
            "sentence": ex["sentence"],
            "aspect": ex["aspect"],
        }


def collate_glove(batch: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = max(len(item["ids"]) for item in batch)
    max_aspect_len = max(len(item["aspect_ids"]) for item in batch)
    ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    aspect_ids = torch.zeros(len(batch), max_aspect_len, dtype=torch.long)
    aspect_mask = torch.zeros(len(batch), max_aspect_len, dtype=torch.bool)
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    for i, item in enumerate(batch):
        row = torch.tensor(item["ids"], dtype=torch.long)
        ids[i, : len(row)] = row
        mask[i, : len(row)] = True
        aspect_row = torch.tensor(item["aspect_ids"], dtype=torch.long)
        aspect_ids[i, : len(aspect_row)] = aspect_row
        aspect_mask[i, : len(aspect_row)] = True
    return ids, mask, aspect_ids, aspect_mask, labels


class AdditiveAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.score(torch.tanh(self.proj(sequence))).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), sequence).squeeze(1)
        return context, weights


class AspectAwareAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.sequence_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.aspect_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self, sequence: torch.Tensor, aspect_vector: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.score(
            torch.tanh(self.sequence_proj(sequence) + self.aspect_proj(aspect_vector).unsqueeze(1))
        ).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), sequence).squeeze(1)
        return context, weights


class GloveAttentionClassifier(nn.Module):
    def __init__(self, embedding_matrix: torch.Tensor, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False, padding_idx=0)
        self.encoder = nn.LSTM(
            embedding_matrix.size(1), hidden_dim, batch_first=True, bidirectional=True
        )
        encoded_dim = hidden_dim * 2
        self.aspect_projection = nn.Linear(embedding_matrix.size(1), encoded_dim)
        self.attention = AspectAwareAttention(encoded_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(encoded_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    def forward(
        self,
        ids: torch.Tensor,
        mask: torch.Tensor,
        aspect_ids: torch.Tensor,
        aspect_mask: torch.Tensor,
    ) -> torch.Tensor:
        emb = self.embedding(ids)
        encoded, _ = self.encoder(emb)
        aspect_emb = self.embedding(aspect_ids)
        aspect_emb = aspect_emb.masked_fill(~aspect_mask.unsqueeze(-1), 0.0)
        aspect_mean = aspect_emb.sum(dim=1) / aspect_mask.sum(dim=1, keepdim=True).clamp_min(1)
        aspect_vector = torch.tanh(self.aspect_projection(aspect_mean))
        context, _ = self.attention(encoded, aspect_vector, mask)
        features = torch.cat([context, aspect_vector, context * aspect_vector], dim=-1)
        return self.classifier(features)


class BertAttentionClassifier(nn.Module):
    def __init__(self, model_name_or_path: str, dropout: float = 0.3):
        super().__init__()
        from transformers import AutoModel

        self.bert = AutoModel.from_pretrained(model_name_or_path)
        hidden = self.bert.config.hidden_size
        self.attention = AdditiveAttention(hidden)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 3))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        context, _ = self.attention(output.last_hidden_state, attention_mask.bool())
        return self.classifier(context)


class BertDataset(Dataset):
    def __init__(self, examples: List[Dict], tokenizer, max_len: int = 96):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        ex = self.examples[idx]
        encoded = self.tokenizer(
            ex["sentence"],
            ex["aspect"],
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(ex["label"], dtype=torch.long),
        }


def accuracy_and_macro_f1(preds: List[int], golds: List[int]) -> Tuple[float, float]:
    acc = sum(p == g for p, g in zip(preds, golds)) / max(1, len(golds))
    f1_scores = []
    for label in range(3):
        tp = sum(p == label and g == label for p, g in zip(preds, golds))
        fp = sum(p == label and g != label for p, g in zip(preds, golds))
        fn = sum(p != label and g == label for p, g in zip(preds, golds))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1_scores.append(2 * precision * recall / max(1e-8, precision + recall))
    return acc, sum(f1_scores) / len(f1_scores)


def class_weights(examples: Sequence[Dict], device: torch.device) -> torch.Tensor:
    counts = torch.zeros(3, dtype=torch.float)
    for ex in examples:
        counts[ex["label"]] += 1
    weights = counts.sum() / counts.clamp_min(1)
    weights = weights / weights.mean()
    return weights.to(device)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    device: torch.device,
    bert: bool = False,
    loss_fn: nn.Module = None,
) -> float:
    model.train()
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    total = 0.0
    for batch in loader:
        optimizer.zero_grad()
        if bert:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            logits = model(input_ids, mask)
        else:
            ids, mask, aspect_ids, aspect_mask, labels = [x.to(device) for x in batch]
            logits = model(ids, mask, aspect_ids, aspect_mask)
        loss = loss_fn(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total += loss.item()
    return total / max(1, len(loader))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, bert: bool = False) -> Tuple[float, float]:
    model.eval()
    preds, golds = [], []
    for batch in loader:
        if bert:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            logits = model(input_ids, mask)
        else:
            ids, mask, aspect_ids, aspect_mask, labels = [x.to(device) for x in batch]
            logits = model(ids, mask, aspect_ids, aspect_mask)
        preds.extend(logits.argmax(-1).cpu().tolist())
        golds.extend(labels.cpu().tolist())
    return accuracy_and_macro_f1(preds, golds)


def run_glove(args, train_examples: List[Dict], test_examples: List[Dict], device: torch.device) -> List[Dict]:
    vocab = build_vocab(train_examples)
    embedding = load_glove(args.glove_path, vocab, args.embed_dim)
    train_loader = DataLoader(SemEvalDataset(train_examples, vocab, args.max_len), batch_size=args.batch_size, shuffle=True, collate_fn=collate_glove)
    test_loader = DataLoader(SemEvalDataset(test_examples, vocab, args.max_len), batch_size=args.batch_size, shuffle=False, collate_fn=collate_glove)
    model = GloveAttentionClassifier(embedding, args.hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss = nn.CrossEntropyLoss(weight=class_weights(train_examples, device) if args.class_weight else None)
    rows = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, bert=False, loss_fn=loss)
        acc, f1 = evaluate(model, test_loader, device, bert=False)
        rows.append({"model": "GloVe-Aspect-Attention", "epoch": epoch, "loss": f"{train_loss:.4f}", "accuracy": f"{acc:.4f}", "macro_f1": f"{f1:.4f}"})
        print(f"[GloVe] epoch={epoch} loss={train_loss:.4f} acc={acc:.4f} macro_f1={f1:.4f}")
    return rows


def run_bert(args, train_examples: List[Dict], test_examples: List[Dict], device: torch.device) -> List[Dict]:
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("transformers is not installed. Skip BERT experiment.")
        return []
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    train_loader = DataLoader(BertDataset(train_examples, tokenizer, args.max_len), batch_size=args.bert_batch_size, shuffle=True)
    test_loader = DataLoader(BertDataset(test_examples, tokenizer, args.max_len), batch_size=args.bert_batch_size, shuffle=False)
    model = BertAttentionClassifier(args.bert_model, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.bert_lr, weight_decay=args.weight_decay)
    loss = nn.CrossEntropyLoss(weight=class_weights(train_examples, device) if args.class_weight else None)
    rows = []
    for epoch in range(1, args.bert_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, bert=True, loss_fn=loss)
        acc, f1 = evaluate(model, test_loader, device, bert=True)
        rows.append({"model": "BERT-Attention", "epoch": epoch, "loss": f"{train_loss:.4f}", "accuracy": f"{acc:.4f}", "macro_f1": f"{f1:.4f}"})
        print(f"[BERT] epoch={epoch} loss={train_loss:.4f} acc={acc:.4f} macro_f1={f1:.4f}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 1: context representation with GloVe/BERT attention for ABSA.")
    parser.add_argument("--data_dir", default="data/SemEval-2014-Task4-Laptop")
    parser.add_argument("--glove_path", default="", help="Optional path to glove.6B.300d.txt or similar file.")
    parser.add_argument("--embed_dim", type=int, default=300)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--max_len", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--class_weight", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--bert_model", default="bert-base-uncased", help="Use a local path if network download is unavailable.")
    parser.add_argument("--bert_batch_size", type=int, default=8)
    parser.add_argument("--bert_epochs", type=int, default=4)
    parser.add_argument("--bert_lr", type=float, default=2e-5)
    parser.add_argument("--skip_bert", action="store_true")
    parser.add_argument("--result_dir", default="result")
    args = parser.parse_args()

    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = resolve_data_dir(args.data_dir)
    result_dir = resolve_result_dir(args.result_dir)
    train_examples = read_semeval_seg(data_dir / "Laptops_Train.xml.seg")
    test_examples = read_semeval_seg(data_dir / "Laptops_Test_Gold.xml.seg")
    print(f"Train={len(train_examples)} Test={len(test_examples)} Device={device}")

    rows = run_glove(args, train_examples, test_examples, device)
    if not args.skip_bert:
        rows.extend(run_bert(args, train_examples, test_examples, device))
    write_csv(result_dir / "task1_context_representation.csv", rows)
    if rows:
        best_by_model = {}
        for row in rows:
            if row["model"] not in best_by_model or float(row["macro_f1"]) > float(best_by_model[row["model"]]["macro_f1"]):
                best_by_model[row["model"]] = row
        write_csv(result_dir / "task1_context_representation_summary.csv", list(best_by_model.values()))
    print(f"Saved results to {result_dir}")


if __name__ == "__main__":
    main()
