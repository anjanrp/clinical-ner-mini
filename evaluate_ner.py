import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
)
import evaluate


def pick_device():
    # Prefer Apple MPS if available, else CPU (add CUDA if you run on a NVIDIA GPU machine)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def find_columns(dataset):
    cols = dataset.column_names
    token_col_candidates = ["tokens", "words"]
    tag_col_candidates = ["tags", "ner_tags", "labels"]

    token_col = next((c for c in token_col_candidates if c in cols), None)
    tag_col = next((c for c in tag_col_candidates if c in cols), None)

    if token_col is None or tag_col is None:
        raise ValueError(f"Could not find token/tag columns in: {cols}")
    return token_col, tag_col


def get_label_list(ds, tag_col):
    feat = ds.features[tag_col]
    if hasattr(feat, "feature") and hasattr(feat.feature, "names"):
        return feat.feature.names
    if hasattr(feat, "names"):
        return feat.names

    # Fallback: build from observed integers (less ideal)
    uniq = set()
    for ex in ds:
        uniq.update(ex[tag_col])
    return [str(i) for i in sorted(uniq)]


def tokenize_and_align_labels(examples, tokenizer, token_col, tag_col):
    tokenized = tokenizer(
        examples[token_col],
        is_split_into_words=True,
        truncation=True,
        padding=False,
    )

    aligned_labels = []
    for i, labels in enumerate(examples[tag_col]):
        word_ids = tokenized.word_ids(batch_index=i)
        prev_word_id = None
        label_ids = []

        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)  # special tokens
            elif word_id != prev_word_id:
                label_ids.append(labels[word_id])  # first subword gets the label
            else:
                label_ids.append(-100)  # ignore subsequent subwords for eval
            prev_word_id = word_id

        aligned_labels.append(label_ids)

    tokenized["labels"] = aligned_labels
    return tokenized


@torch.no_grad()
def run_eval(model, dataloader, id2label, device):
    metric = evaluate.load("seqeval")
    model.eval()
    model.to(device)

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)

        preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
        labels = batch["labels"].cpu().numpy()

        true_labels = []
        true_preds = []
        for p_row, l_row in zip(preds, labels):
            sent_labels = []
            sent_preds = []
            for p, l in zip(p_row, l_row):
                if l == -100:
                    continue
                sent_labels.append(id2label[int(l)])
                sent_preds.append(id2label[int(p)])
            true_labels.append(sent_labels)
            true_preds.append(sent_preds)

        metric.add_batch(predictions=true_preds, references=true_labels)

    return metric.compute()


def load_bc5cdr_parquet():
    base = "https://huggingface.co/datasets/tner/bc5cdr/resolve/refs%2Fconvert%2Fparquet/bc5cdr"
    data_files = {
        "train": f"{base}/train/0000.parquet",
        "validation": f"{base}/validation/0000.parquet",
        "test": f"{base}/test/0000.parquet",
    }
    return load_dataset("parquet", data_files=data_files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tner/bc5cdr")
    ap.add_argument("--model", default="tner/xlm-roberta-base-bc5cdr")
    ap.add_argument("--split", default="test", choices=["train", "validation", "test"])
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_samples", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    device = pick_device()
    print(f"Using device: {device}")

    # ---- load dataset ----
    if args.dataset == "tner/bc5cdr":
        # HF datasets>=4 removed script-based datasets; use parquet conversion.
        ds = load_bc5cdr_parquet()
    else:
        ds = load_dataset(args.dataset)

    token_col, tag_col = find_columns(ds[args.split])

    # ---- labels ----
    if args.dataset == "tner/bc5cdr":
        # Official label2id from dataset card
        label_list = ["O", "B-Chemical", "B-Disease", "I-Disease", "I-Chemical"]
    else:
        label_list = get_label_list(ds["train"], tag_col)

    id2label = {i: lab for i, lab in enumerate(label_list)}
    label2id = {lab: i for i, lab in enumerate(label_list)}

    # ---- model/tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(args.model, add_prefix_space=True)

    model = AutoModelForTokenClassification.from_pretrained(
        args.model,
        id2label=id2label,
        label2id=label2id,
    )

    # ---- select split + optionally subsample ----
    data = ds[args.split]
    if args.max_samples and args.max_samples > 0:
        data = data.select(range(min(args.max_samples, len(data))))

    # ---- tokenize + align labels ----
    tokenized = data.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, token_col, tag_col),
        batched=True,
        remove_columns=data.column_names,
    )

    # ---- dataloader ----
    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    dl = torch.utils.data.DataLoader(
        tokenized, batch_size=args.batch_size, collate_fn=collator
    )

    # ---- eval ----
    results = run_eval(model, dl, id2label, device)
    print("\n=== SeqEval Metrics ===")
    def to_py(x):
        try:
            import numpy as np
            if isinstance(x, (np.floating, np.integer)):
                return x.item()
        except Exception:
            pass
        if isinstance(x, dict):
            return {k: to_py(v) for k, v in x.items()}
        if isinstance(x, list):
            return [to_py(v) for v in x]
        return x

    clean = to_py(results)
    print(clean)


if __name__ == "__main__":
    main()
