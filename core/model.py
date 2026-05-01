"""
core/model.py — MiniGPT с repetition penalty и no_repeat_ngram_size.

Ключевые улучшения:
- repetition_penalty: штрафует уже встречавшиеся токены
- no_repeat_ngram: запрещает повтор n-грамм (убирает "кивнул и кивнул")
- min_new_tokens: гарантирует минимальную длину (не обрывается после 1 слова)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, seq_len: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        scale = math.sqrt(self.d_head)
        att = (q @ k.transpose(-2, -1)) / scale
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.drop(att)
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, dropout: float, seq_len: int
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, seq_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
        seq_len: int,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, d_ff, dropout, seq_len)
                for _ in range(n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # tied weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
        nn.init.zeros_(self.tok_emb.weight[0])

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)

    @torch.no_grad()
    def generate(
        self,
        prompt: list[int],
        max_new: int,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.92,
        eos_id: int = 2,
        vocab_size: int = None,
        repetition_penalty: float = 1.3,
        no_repeat_ngram: int = 3,
        min_new_tokens: int = 5,
    ) -> list[int]:
        """
        Генерация с защитой от повторений:

        repetition_penalty > 1.0:
            Логиты уже встречавшихся токенов делятся на penalty.
            1.0 = нет штрафа, 1.3 = умеренный штраф, 1.6 = сильный.

        no_repeat_ngram > 0:
            Запрещает любой токен, если последние (n-1) токенов уже
            встречались за ними в таком же порядке.
            3 = запрет трёхграмм (рекомендуется).

        min_new_tokens:
            EOS игнорируется пока не сгенерировано min_new_tokens токенов.
        """
        vs = vocab_size or self.tok_emb.num_embeddings
        ctx = [min(max(t, 0), vs - 1) for t in prompt]
        self.eval()
        generated = []

        for step in range(max_new):
            chunk = ctx[-self.seq_len :]
            inp = torch.tensor([chunk], dtype=torch.long).clamp(0, vs - 1)
            logits = self(inp)[0, -1, :vs].float()

            # ── Repetition penalty ───────────────────────────────
            if repetition_penalty != 1.0 and ctx:
                seen = set(ctx[-64:])  # штрафуем только последние 64 токена
                for tid in seen:
                    if 0 <= tid < vs:
                        if logits[tid] > 0:
                            logits[tid] /= repetition_penalty
                        else:
                            logits[tid] *= repetition_penalty

            # ── No-repeat n-gram ─────────────────────────────────
            if no_repeat_ngram > 1 and len(ctx) >= no_repeat_ngram - 1:
                prefix = tuple(ctx[-(no_repeat_ngram - 1) :])
                # Найдём все токены, которые следовали за этим prefix
                banned = set()
                for i in range(len(ctx) - no_repeat_ngram + 1):
                    if tuple(ctx[i : i + no_repeat_ngram - 1]) == prefix:
                        banned.add(ctx[i + no_repeat_ngram - 1])
                for tid in banned:
                    if 0 <= tid < vs:
                        logits[tid] = float("-inf")

            # ── Temperature ──────────────────────────────────────
            logits = logits / max(temperature, 1e-8)

            # ── Top-K ────────────────────────────────────────────
            if top_k > 0:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[-1]
                logits[logits < kth] = float("-inf")

            # ── Top-P (nucleus) ──────────────────────────────────
            probs = F.softmax(logits, dim=-1).numpy()
            if top_p < 1.0:
                si = np.argsort(probs)[::-1]
                cum = np.cumsum(probs[si])
                cut = int(np.searchsorted(cum, top_p)) + 1
                mask = np.zeros_like(probs)
                mask[si[:cut]] = probs[si[:cut]]
                s = mask.sum()
                if s > 0:
                    probs = mask / s

            nid = int(np.random.choice(len(probs), p=probs))

            # EOS только после min_new_tokens
            if nid == eos_id and step >= min_new_tokens:
                break

            generated.append(nid)
            ctx.append(nid)

        return generated


def build(vocab_size: int, params: dict) -> MiniGPT:
    return MiniGPT(
        vocab_size=vocab_size,
        d_model=params["d_model"],
        n_heads=params["n_heads"],
        n_layers=params["n_layers"],
        d_ff=params["d_ff"],
        dropout=params["dropout"],
        seq_len=params["seq_len"],
    )


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
