"""Load and sample from intent classification datasets."""
import random
from datasets import load_from_disk
from config import DATASET_PATHS, SEED


def load_banking77():
    ds = load_from_disk(DATASET_PATHS["banking77"])
    label_names = ds["test"].features["label"].names
    return ds, label_names


def load_clinc150():
    ds = load_from_disk(DATASET_PATHS["clinc150"])
    label_names = ds["test"].features["intent"].names
    return ds, label_names


def sample_queries(dataset_name, n_samples, split="test", seed=SEED):
    """Sample n_samples queries stratified by intent label."""
    rng = random.Random(seed)

    if dataset_name == "banking77":
        ds, label_names = load_banking77()
        data = ds[split]
        texts = data["text"]
        labels = data["label"]
    elif dataset_name == "clinc150":
        ds, label_names = load_clinc150()
        data = ds[split]
        texts = data["text"]
        labels = data["intent"]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Group by label
    label_to_indices = {}
    for i, label in enumerate(labels):
        label_to_indices.setdefault(label, []).append(i)

    # Stratified sampling
    samples = []
    n_labels = len(label_to_indices)
    per_label = max(1, n_samples // n_labels)
    remaining = n_samples

    label_keys = sorted(label_to_indices.keys())
    rng.shuffle(label_keys)

    for label_id in label_keys:
        if remaining <= 0:
            break
        indices = label_to_indices[label_id]
        take = min(per_label, len(indices), remaining)
        chosen = rng.sample(indices, take)
        for idx in chosen:
            samples.append({
                "dataset": dataset_name,
                "index": idx,
                "text": texts[idx],
                "label_id": label_id,
                "label_name": label_names[label_id],
            })
        remaining -= take

    # If we still need more, sample randomly from remaining
    if remaining > 0:
        used_indices = {s["index"] for s in samples}
        all_indices = [i for i in range(len(texts)) if i not in used_indices]
        extra = rng.sample(all_indices, min(remaining, len(all_indices)))
        for idx in extra:
            samples.append({
                "dataset": dataset_name,
                "index": idx,
                "text": texts[idx],
                "label_id": labels[idx],
                "label_name": label_names[labels[idx]],
            })

    rng.shuffle(samples)
    return samples[:n_samples]


if __name__ == "__main__":
    b77 = sample_queries("banking77", 10)
    c150 = sample_queries("clinc150", 10)
    print("Banking77 samples:")
    for s in b77[:3]:
        print(f"  [{s['label_name']}] {s['text']}")
    print(f"\nCLINC150 samples:")
    for s in c150[:3]:
        print(f"  [{s['label_name']}] {s['text']}")
