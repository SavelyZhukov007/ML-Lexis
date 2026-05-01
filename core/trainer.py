"""
core/trainer.py — Lexis
- Хранит последние 20 чекпоинтов, best_model.pt всегда
- Правки пользователей в датасет
- Windows-safe state
"""

import gc
import math
import time
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset

import core.state as ST
from core.config import model_ckpt_dir
from core.model import build, count_params
from core.postprocess import postprocess
from core.tokenizer import BOS, load_vocab, load_tokens, tokenize_text
from core.db import add_model_sample, update_model_version, get_all_corrections

log = logging.getLogger("trainer")

os.environ["KMP_BLOCKTIME"] = "0"
MAX_RAM_BYTES = 900 * 1024 * 1024

_setup_done = False


def _setup():
    global _setup_done
    if _setup_done:
        return
    n = min(8, os.cpu_count() or 4)
    try:
        torch.set_num_threads(n)
    except RuntimeError:
        pass
    _setup_done = True
    log.info(f"PyTorch {torch.__version__} | потоков: {n}")


def _safe_batch_size(params: dict) -> int:
    bs = params["batch_size"]
    d = params["d_model"]
    sl = params["seq_len"]
    nl = params["n_layers"]
    per = sl * d * nl * 4 * 8
    while bs > 1 and bs * per > MAX_RAM_BYTES // 2:
        bs //= 2
    if bs != params["batch_size"]:
        log.warning(f"Batch size уменьшен до {bs} для экономии RAM")
    return bs


class TokenDataset(Dataset):
    def __init__(self, tokens: np.ndarray, seq_len: int):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.tokens[idx : idx + self.seq_len].astype(np.int64))
        y = torch.from_numpy(
            self.tokens[idx + 1 : idx + self.seq_len + 1].astype(np.int64)
        )
        return x, y


class CorrectionDataset(Dataset):
    """Датасет из правок пользователей: wrong → correct."""

    def __init__(self, pairs: list[tuple], w2i: dict, seq_len: int):
        from core.tokenizer import tokenize_text, BOS, EOS

        self.seq_len = seq_len
        self.samples = []
        for wrong, correct in pairs:
            wt = [BOS] + [w2i.get(t, BOS) for t in tokenize_text(wrong)] + [EOS]
            ct = [BOS] + [w2i.get(t, BOS) for t in tokenize_text(correct)] + [EOS]
            for _ in range(3):
                self.samples.append((wt, ct))

    def __len__(self):
        return len(self.samples)

    def _pad(self, seq: list) -> list:
        seq = seq[: self.seq_len]
        return seq + [0] * (self.seq_len - len(seq))

    def __getitem__(self, idx):
        wt, ct = self.samples[idx]
        return (
            torch.tensor(self._pad(wt), dtype=torch.long),
            torch.tensor(self._pad(ct), dtype=torch.long),
        )


def _ckpt_path(ckpt_dir: Path, epoch: int) -> Path:
    return ckpt_dir / f"epoch_{epoch:04d}.pt"


def _best_path(ckpt_dir: Path) -> Path:
    return ckpt_dir / "best_model.pt"


def _latest_ckpt(ckpt_dir: Path):
    ckpts = sorted(ckpt_dir.glob("epoch_*.pt"))
    if not ckpts:
        return None, 0
    path = ckpts[-1]
    epoch = int(path.stem.split("_")[1])
    return path, epoch


def _prune(ckpt_dir: Path, keep: int = 20):
    """Держим последние `keep` чекпоинтов. best_model.pt не трогаем."""
    ckpts = sorted(ckpt_dir.glob("epoch_*.pt"))
    for old in ckpts[:-keep]:
        try:
            old.unlink()
        except Exception:
            pass


def _save_ckpt(model, optimizer, epoch, vocab_size, params, ckpt_dir: Path, is_checkpoint=True):
    data = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "vocab_size": vocab_size,
        "params": params,
    }
    if is_checkpoint:
        torch.save(data, str(_ckpt_path(ckpt_dir, epoch)))
        _prune(ckpt_dir, keep=20)
    # best_model не содержит optimizer (экономия места)
    best_data = {k: v for k, v in data.items() if k != "optimizer"}
    return best_data


def _save_sample_text(model, vocab, params, prompt_ids, sample_num, ckpt_vs):
    raw_ids = model.generate(
        prompt_ids,
        max_new=120,
        temperature=params.get("temperature", 1.0),
        top_k=params.get("top_k", 50),
        top_p=params.get("top_p", 0.92),
        eos_id=vocab["w2i"].get("<EOS>", 2),
        vocab_size=ckpt_vs,
        repetition_penalty=params.get("repetition_penalty", 1.3),
        no_repeat_ngram=params.get("no_repeat_ngram", 3),
        min_new_tokens=20,
    )
    words = [vocab["i2w"].get(i, "") for i in raw_ids if vocab["i2w"].get(i, "")]
    text = " ".join(words).strip()
    if len(text) > 500:
        text = text[:500].rsplit(" ", 1)[0]
    return text


def _save_epoch_samples(model, vocab, params, version_id, epoch, model_root):
    if version_id is None:
        return
    base_prompt = [vocab["w2i"].get("<BOS>", 1)]
    variants = [
        {"temperature": 0.75, "top_k": 40, "top_p": 0.92},
        {"temperature": 1.0, "top_k": 50, "top_p": 0.95},
        {"temperature": 1.2, "top_k": 60, "top_p": 0.90},
        {"temperature": 0.9, "top_k": 30, "top_p": 0.85},
        {"temperature": 1.1, "top_k": 20, "top_p": 0.99},
    ]
    ckpt_vs = vocab["vocab_size"]
    for idx, variant in enumerate(variants, start=1):
        sample_params = {
            "temperature": variant["temperature"],
            "top_k": variant["top_k"],
            "top_p": variant["top_p"],
            "repetition_penalty": 1.3,
            "no_repeat_ngram": 3,
        }
        params_save = {**params, **sample_params}
        text = _save_sample_text(model, vocab, params_save, base_prompt, idx, ckpt_vs)
        add_model_sample(version_id, epoch, idx, params_save, text, saved_name=None)


def run(params: dict, version_id: int = None, model_slug: str | None = None):
    _setup()

    log.info("Загрузка данных...")
    model_root = None
    if model_slug:
        model_root = Path(model_ckpt_dir(model_slug)).parent
    vocab = load_vocab(base_dir=model_root)
    tokens = load_tokens(base_dir=model_root)
    vocab_size = vocab["vocab_size"]
    seq_len = params["seq_len"]
    batch_size = _safe_batch_size(params)

    split = int(len(tokens) * 0.9)
    train_tok = tokens[:split]
    val_tok = tokens[split:]

    main_train_ds = TokenDataset(train_tok, seq_len)
    val_ds = TokenDataset(val_tok, seq_len)

    try:
        from core.db import get_corrections_as_training_pairs

        pairs = get_corrections_as_training_pairs()
        if pairs:
            corr_ds = CorrectionDataset(pairs, vocab["w2i"], seq_len)
            train_ds = ConcatDataset([main_train_ds, corr_ds])
            log.info(f"  Добавлено {len(pairs)} правок в обучение")
        else:
            train_ds = main_train_ds
    except Exception as e:
        log.warning(f"Правки не загружены: {e}")
        train_ds = main_train_ds

    n_train = max(1, len(train_ds) // batch_size)
    n_val = max(1, len(val_ds) // batch_size)
    log.info(
        f"  Токенов: {len(tokens):,} | vocab: {vocab_size:,} | "
        f"batch: {batch_size} | train: {n_train:,}"
    )

    resume_epoch, _ = ST.get_resume_point()
    ckpt_dir = model_ckpt_dir(model_slug)
    ckpt_path, ckpt_epoch = _latest_ckpt(ckpt_dir)

    model = build(vocab_size, params)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params["lr"],
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(n_train * 10, 1000), eta_min=params["lr"] * 0.05
    )

    start_epoch = 0
    if ckpt_path and resume_epoch > 0:
        log.info(f"  Чекпоинт: {ckpt_path}")
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        saved_vs = ckpt.get("vocab_size", vocab_size)
        if saved_vs != vocab_size:
            _extend_model_vocab(ckpt["model"], saved_vs, vocab_size, params["d_model"])
        model.load_state_dict(ckpt["model"], strict=False)
        if "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except Exception:
                pass
        start_epoch = max(resume_epoch, ckpt_epoch)
        log.info(f"  Возобновление с эпохи {start_epoch + 1}")

    ST.set_total_params(count_params(model))
    ST.set_running()
    best_val = float("inf")
    epoch = start_epoch
    log.info("Обучение запущено.")

    while True:
        epoch += 1
        ST.mark_epoch_start()
        ST.set_epochs_planned(epoch + 20)
        ep_start = time.time()

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0
        )

        model.train()
        ep_loss, ep_acc, ep_steps = 0.0, 0.0, 0

        for step, (xb, yb) in enumerate(train_loader):
            if step % 10 == 0 and ST.is_stop_requested():
                log.info(f"  СТОП на шаге {step}")
                ST.save_pause(epoch, step)
                _save_ckpt(model, optimizer, epoch, vocab_size, params, ckpt_dir)
                return

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits.view(-1, vocab_size), yb.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            bl = loss.item()
            with torch.no_grad():
                preds = logits.argmax(-1)
                mask = yb != 0
                ba = (
                    (preds[mask] == yb[mask]).float().mean().item()
                    if mask.any()
                    else 0.0
                )

            ep_loss += bl
            ep_acc += ba
            ep_steps += 1

            if step % 5 == 0:
                ST.update_batch(epoch, step + 1, ep_loss / ep_steps, ep_acc / ep_steps)

            if step % 200 == 0:
                cur_lr = scheduler.get_last_lr()[0]
                log.info(
                    f"  [{epoch}:{step:5d}/{n_train}] loss={bl:.4f} "
                    f"PPL={math.exp(min(bl,30)):.1f} lr={cur_lr:.2e}"
                )

        model.eval()
        v_losses = []
        with torch.no_grad():
            for xv, yv in DataLoader(
                val_ds,
                batch_size=batch_size,
                shuffle=False,
                drop_last=True,
                num_workers=0,
            ):
                vl = criterion(model(xv).view(-1, vocab_size), yv.view(-1))
                v_losses.append(vl.item())

        tl = ep_loss / max(ep_steps, 1)
        vl = float(np.mean(v_losses)) if v_losses else tl
        tppl = math.exp(min(tl, 30))
        vppl = math.exp(min(vl, 30))
        cur_lr = scheduler.get_last_lr()[0]
        dur = time.time() - ep_start

        ST.update_epoch(epoch, tl, vl, vppl, tppl, cur_lr, dur, version_id=version_id)

        is_best = vl < best_val
        if is_best:
            best_val = vl
            best_data = _save_ckpt(
                model, optimizer, epoch, vocab_size, params, ckpt_dir, is_checkpoint=False
            )
            best_path = _best_path(ckpt_dir)
            torch.save(best_data, str(best_path))
            update_model_version(version_id, best_epoch=epoch, best_val_loss=vl, checkpoint_path=str(best_path))
            log.info(f"  ★ Новый лучший val={vl:.4f}")

        log.info(
            f"  Epoch {epoch} | train={tl:.4f} PPL={tppl:.1f} | "
            f"val={vl:.4f} PPL={vppl:.1f} | {dur:.0f}с"
        )

        # Сохраняем чекпоинт каждые 5 эпох и в конце
        if epoch % 5 == 0:
            _save_ckpt(model, optimizer, epoch, vocab_size, params, ckpt_dir)

        # Генерируем пробные сэмплы для истории модели
        try:
            _save_epoch_samples(model, vocab, params, version_id, epoch, model_root)
        except Exception as e:
            log.warning(f"Не удалось сохранить образцы после эпохи {epoch}: {e}")

        gc.collect()


def _extend_model_vocab(state_dict: dict, old_vs: int, new_vs: int, d_model: int):
    for key in ["tok_emb.weight", "head.weight"]:
        if key in state_dict:
            old_w = state_dict[key]
            if old_w.shape[0] == old_vs:
                extra = torch.zeros(new_vs - old_vs, old_w.shape[1])
                torch.nn.init.normal_(extra, std=0.02)
                state_dict[key] = torch.cat([old_w, extra], dim=0)
