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
    d, sl, nl = params["d_model"], params["seq_len"], params["n_layers"]
    per = sl * d * nl * 4 * 8
    while bs > 1 and bs * per > MAX_RAM_BYTES // 2:
        bs //= 2
    if bs != params["batch_size"]:
        log.warning(f"Batch size уменьшен до {bs}")
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
    def __init__(self, pairs: list, w2i: dict, seq_len: int):
        from core.tokenizer import BOS, EOS

        self.seq_len = seq_len
        self.samples = []
        for wrong, correct in pairs:
            wt = [BOS] + [w2i.get(t, BOS) for t in tokenize_text(wrong)] + [EOS]
            ct = [BOS] + [w2i.get(t, BOS) for t in tokenize_text(correct)] + [EOS]
            for _ in range(3):
                self.samples.append((wt, ct))

    def __len__(self):
        return len(self.samples)

    def _pad(self, seq):
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
    for old in sorted(ckpt_dir.glob("epoch_*.pt"))[:-keep]:
        try:
            old.unlink()
        except Exception:
            pass


def _save_ckpt(
    model, optimizer, epoch, vocab_size, params, ckpt_dir: Path, is_checkpoint=True
):
    data = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "vocab_size": vocab_size,
        "params": params,
        "arch": "v5",
    }
    if is_checkpoint:
        torch.save(data, str(_ckpt_path(ckpt_dir, epoch)))
        _prune(ckpt_dir, keep=20)
    return {k: v for k, v in data.items() if k != "optimizer"}


def _save_sample_text(model, vocab, params, prompt_ids, ckpt_vs):
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


def _save_epoch_samples(model, vocab, params, version_id, epoch):
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
    for idx, variant in enumerate(variants, 1):
        p = {**params, **variant, "repetition_penalty": 1.3, "no_repeat_ngram": 3}
        text = _save_sample_text(model, vocab, p, base_prompt, ckpt_vs)
        add_model_sample(version_id, epoch, idx, p, text, saved_name=None)


# ── Warmup + Cosine LR scheduler ─────────────────────────────────────────────


class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.05,
        last_epoch=-1,
    ):
        self.warmup_steps = max(warmup_steps, 1)
        self.total_steps = max(total_steps, warmup_steps + 1)
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        lrs = []
        for base_lr in self.base_lrs:
            min_lr = base_lr * self.min_lr_ratio
            if step < self.warmup_steps:
                lr = base_lr * step / self.warmup_steps
            else:
                progress = (step - self.warmup_steps) / (
                    self.total_steps - self.warmup_steps
                )
                lr = min_lr + 0.5 * (base_lr - min_lr) * (
                    1 + math.cos(math.pi * progress)
                )
            lrs.append(max(lr, min_lr))
        return lrs


def _extend_model_vocab(state_dict, old_vs, new_vs, d_model):
    for key in ["tok_emb.weight", "head.weight"]:
        if key in state_dict:
            old_w = state_dict[key]
            if old_w.shape[0] == old_vs:
                extra = torch.zeros(new_vs - old_vs, old_w.shape[1])
                nn.init.normal_(extra, std=0.02)
                state_dict[key] = torch.cat([old_w, extra], dim=0)


def run(params: dict, version_id: int = None, model_slug: str = None):
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
    grad_accum = params.get("grad_accum", 1)
    use_amp = params.get("use_amp", False) and torch.cuda.is_available()
    patience = params.get("early_stop_patience", 15)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split = int(len(tokens) * 0.9)
    train_tok, val_tok = tokens[:split], tokens[split:]
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
        f"  Токенов: {len(tokens):,} | vocab: {vocab_size:,} | batch: {batch_size} | train: {n_train:,} | device: {device}"
    )

    resume_epoch, _ = ST.get_resume_point()
    ckpt_dir = model_ckpt_dir(model_slug)
    ckpt_path, ckpt_epoch = _latest_ckpt(ckpt_dir)

    model = build(vocab_size, params).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params["lr"],
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
    )

    # Total steps for scheduler: estimated epochs * batches
    estimated_epochs = max(params.get("max_epochs", 200), 50)
    total_steps = n_train * estimated_epochs
    warmup_steps = min(200, total_steps // 20)
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_steps, total_steps, min_lr_ratio=0.05
    )

    scaler = torch.cuda.amp.GradScaler() if use_amp else None

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
    no_improve = 0
    epoch = start_epoch
    log.info("Обучение запущено.")

    while True:
        epoch += 1
        ST.mark_epoch_start()
        ST.set_epochs_planned(epoch + 20)
        ep_start = time.time()

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )

        model.train()
        ep_loss, ep_acc, ep_steps = 0.0, 0.0, 0
        optimizer.zero_grad(set_to_none=True)

        for step, (xb, yb) in enumerate(train_loader):
            if step % 10 == 0 and ST.is_stop_requested():
                log.info(f"  СТОП на шаге {step}")
                ST.save_pause(epoch, step)
                _save_ckpt(model, optimizer, epoch, vocab_size, params, ckpt_dir)
                return

            xb, yb = xb.to(device), yb.to(device)

            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = model(xb)
                    loss = (
                        criterion(logits.view(-1, vocab_size), yb.view(-1)) / grad_accum
                    )
                scaler.scale(loss).backward()
            else:
                logits = model(xb)
                loss = criterion(logits.view(-1, vocab_size), yb.view(-1)) / grad_accum
                loss.backward()

            if (step + 1) % grad_accum == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            bl = loss.item() * grad_accum
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
                    f"  [{epoch}:{step:5d}/{n_train}] loss={bl:.4f} PPL={math.exp(min(bl,30)):.1f} lr={cur_lr:.2e}"
                )

        # Validation
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
                xv, yv = xv.to(device), yv.to(device)
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
            no_improve = 0
            best_data = _save_ckpt(
                model,
                optimizer,
                epoch,
                vocab_size,
                params,
                ckpt_dir,
                is_checkpoint=False,
            )
            best_path = _best_path(ckpt_dir)
            torch.save(best_data, str(best_path))
            update_model_version(
                version_id,
                best_epoch=epoch,
                best_val_loss=vl,
                checkpoint_path=str(best_path),
            )
            log.info(f"  ★ Новый лучший val={vl:.4f}")
        else:
            no_improve += 1
            # Reduce LR on plateau
            if no_improve % 5 == 0:
                for pg in optimizer.param_groups:
                    pg["lr"] = max(pg["lr"] * 0.5, params["lr"] * 0.01)
                log.info(f"  Plateau: lr уменьшен (no_improve={no_improve})")

        log.info(
            f"  Epoch {epoch} | train={tl:.4f} PPL={tppl:.1f} | val={vl:.4f} PPL={vppl:.1f} | {dur:.0f}с | no_improve={no_improve}"
        )

        if epoch % 5 == 0:
            _save_ckpt(model, optimizer, epoch, vocab_size, params, ckpt_dir)

        try:
            _save_epoch_samples(model, vocab, params, version_id, epoch)
        except Exception as e:
            log.warning(f"Образцы не сохранены: {e}")

        # Early stopping
        if no_improve >= patience:
            log.info(f"  Early stopping: {patience} эпох без улучшения val_loss")
            ST.set_stopped_early()
            break

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
