"""core/tokenizer.py — токенизация и словарь. Без <UNK>."""

import re
import json
import numpy as np
from pathlib import Path
from core.config import STORAGE_DIR

VOCAB_FILE = STORAGE_DIR / "vocab.json"
TOKENS_FILE = STORAGE_DIR / "tokens.npy"

PAD = 0
BOS = 1
EOS = 2
SPECIAL = ["<PAD>", "<BOS>", "<EOS>"]


def vocab_file(base_dir: Path | None = None) -> Path:
    return Path(base_dir) / "vocab.json" if base_dir else VOCAB_FILE


def tokens_file(base_dir: Path | None = None) -> Path:
    return Path(base_dir) / "tokens.npy" if base_dir else TOKENS_FILE


def tokenize_text(text: str) -> list[str]:
    text = text.lower()
    return re.findall(r"[а-яёa-zA-Zа-яА-ЯЁ]+|[0-9]+|[.,!?;:—–\-\"\'()«»]", text)


def build_vocab(texts: list[str]) -> dict:
    freq: dict[str, int] = {}
    for text in texts:
        for t in tokenize_text(text):
            freq[t] = freq.get(t, 0) + 1
    words = sorted(freq.keys(), key=lambda w: (-freq[w], w))
    vocab = SPECIAL + words
    w2i = {w: i for i, w in enumerate(vocab)}
    i2w = {i: w for i, w in enumerate(vocab)}
    return {"vocab": vocab, "w2i": w2i, "i2w": i2w, "vocab_size": len(vocab)}


def build_and_save(main_file: str, extra_files: list[str] = None, base_dir: Path | None = None) -> dict:
    paths = [main_file] + (extra_files or [])
    texts = []
    for p in paths:
        try:
            texts.append(Path(p).read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            pass
    if not texts:
        raise FileNotFoundError("Нет доступных файлов датасета")

    vdata = build_vocab(texts)
    w2i = vdata["w2i"]
    all_ids = [BOS]
    total_words = 0
    for text in texts:
        toks = tokenize_text(text)
        total_words += len(toks)
        all_ids.extend(w2i[t] for t in toks)
        all_ids.append(EOS)
        all_ids.append(BOS)
    all_ids.append(EOS)
    arr = np.array(all_ids, dtype=np.int32)
    json_data = {
        "vocab": vdata["vocab"],
        "w2i": vdata["w2i"],
        "i2w": {str(k): v for k, v in vdata["i2w"].items()},
        "vocab_size": vdata["vocab_size"],
    }
    vocab_path = vocab_file(base_dir)
    tokens_path = tokens_file(base_dir)
    vocab_path.write_text(json.dumps(json_data, ensure_ascii=False), "utf-8")
    np.save(str(tokens_path), arr)
    return {
        "vocab_size": vdata["vocab_size"],
        "n_tokens": len(arr),
        "n_words": total_words,
        "n_files": len(texts),
    }


def load_vocab(base_dir: Path | None = None) -> dict:
    path = vocab_file(base_dir)
    data = json.loads(path.read_text("utf-8"))
    data["i2w"] = {int(k): v for k, v in data["i2w"].items()}
    return data


def load_tokens(base_dir: Path | None = None) -> np.ndarray:
    return np.load(str(tokens_file(base_dir)))


def extend_vocab_with_words(new_words: list[str], base_dir: Path | None = None) -> int:
    path = vocab_file(base_dir)
    if not path.exists():
        return 0
    data = json.loads(path.read_text("utf-8"))
    vocab = data["vocab"]
    w2i = data["w2i"]
    added = 0
    for w in new_words:
        w = w.lower().strip()
        if w and w not in w2i:
            idx = len(vocab)
            vocab.append(w)
            w2i[w] = idx
            added += 1
    if added:
        i2w = {str(i): w for i, w in enumerate(vocab)}
        data.update({"vocab": vocab, "w2i": w2i, "i2w": i2w, "vocab_size": len(vocab)})
        vocab_file(base_dir).write_text(json.dumps(data, ensure_ascii=False), "utf-8")
    return added
