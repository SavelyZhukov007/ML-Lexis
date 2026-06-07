import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


def _precompute_freqs(dim: int, seq_len: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(seq_len * 2)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def _apply_rope(q: torch.Tensor, k: torch.Tensor, freqs: torch.Tensor):
    T = q.shape[2]
    fc = freqs[:T]
    q_ = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
    k_ = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
    return (
        torch.view_as_real(q_ * fc).flatten(3).type_as(q),
        torch.view_as_real(k_ * fc).flatten(3).type_as(k),
    )


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, seq_len: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = dropout
        freqs = _precompute_freqs(self.d_head, seq_len)
        self.register_buffer("freqs_cis", freqs)

    def forward(self, x, kv_cache=None):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        q, k = _apply_rope(q, k, self.freqs_cis)

        if kv_cache is not None:
            pk, pv = kv_cache
            if pk is not None:
                k = torch.cat([pk, k], dim=2)
                v = torch.cat([pv, v], dim=2)
            new_cache = (k, v)
        else:
            new_cache = None

        is_causal = kv_cache is None
        try:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop if self.training else 0.0,
                is_causal=is_causal,
            )
        except Exception:
            scale = math.sqrt(self.d_head)
            att = (q @ k.transpose(-2, -1)) / scale
            if is_causal:
                Sq, Sk = att.shape[-2], att.shape[-1]
                mask = torch.tril(torch.ones(Sq, Sk, device=x.device)).view(
                    1, 1, Sq, Sk
                )
                att = att.masked_fill(mask == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            out = att @ v

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out), new_cache


class TransformerBlock(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, dropout: float, seq_len: int
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, seq_len)
        self.norm2 = RMSNorm(d_model)
        self.ff = SwiGLU(d_model, d_ff, dropout)

    def forward(self, x, kv_cache=None):
        a, new_cache = self.attn(self.norm1(x), kv_cache)
        x = x + a
        x = x + self.ff(self.norm2(x))
        return x, new_cache


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, dropout, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, d_ff, dropout, seq_len)
                for _ in range(n_layers)
            ]
        )
        self.norm_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
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
        x = self.drop(self.tok_emb(idx))
        for block in self.blocks:
            x, _ = block(x)
        return self.head(self.norm_f(x))

    @torch.no_grad()
    def generate(
        self,
        prompt: list,
        max_new: int,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.92,
        eos_id: int = 2,
        vocab_size: int = None,
        repetition_penalty: float = 1.3,
        no_repeat_ngram: int = 3,
        min_new_tokens: int = 5,
    ) -> list:
        vs = vocab_size or self.tok_emb.num_embeddings
        ctx = [min(max(t, 0), vs - 1) for t in prompt]
        self.eval()
        generated = []

        ngram_idx: dict = {}
        if no_repeat_ngram > 1:
            n = no_repeat_ngram
            for i in range(len(ctx) - n + 1):
                pref = tuple(ctx[i : i + n - 1])
                ngram_idx.setdefault(pref, set()).add(ctx[i + n - 1])

        kv_caches = [(None, None)] * len(self.blocks)
        chunk = ctx[-self.seq_len :]
        inp = torch.tensor([chunk], dtype=torch.long).clamp(0, vs - 1)
        x = self.drop(self.tok_emb(inp))
        new_kv_list = []
        for i, block in enumerate(self.blocks):
            x, kv = block(x, kv_cache=kv_caches[i])
            new_kv_list.append(kv)
        kv_caches = new_kv_list
        x = self.norm_f(x)

        for step in range(max_new):
            logits = self.head(x)[0, -1, :vs].float()

            if repetition_penalty != 1.0 and ctx:
                seen = set(ctx[-64:])
                for tid in seen:
                    if 0 <= tid < vs:
                        logits[tid] = (
                            logits[tid] / repetition_penalty
                            if logits[tid] > 0
                            else logits[tid] * repetition_penalty
                        )

            if no_repeat_ngram > 1 and len(ctx) >= no_repeat_ngram - 1:
                pref = tuple(ctx[-(no_repeat_ngram - 1) :])
                for tid in ngram_idx.get(pref, set()):
                    if 0 <= tid < vs:
                        logits[tid] = float("-inf")

            logits = logits / max(temperature, 1e-8)

            if top_k > 0:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[-1]
                logits[logits < kth] = float("-inf")

            probs = F.softmax(logits, dim=-1).cpu().numpy().astype(np.float64)
            probs = np.clip(probs, 0, None)

            if top_p < 1.0:
                si = np.argsort(probs)[::-1]
                cum = np.cumsum(probs[si])
                cut = int(np.searchsorted(cum, top_p)) + 1
                mask = np.zeros_like(probs)
                mask[si[:cut]] = probs[si[:cut]]
                s = mask.sum()
                if s > 0:
                    probs = mask / s

            total = probs.sum()
            if total <= 0:
                probs = np.ones(vs, dtype=np.float64) / vs
            else:
                probs = probs / total

            nid = int(np.random.choice(len(probs), p=probs))

            if nid == eos_id and step >= min_new_tokens:
                break

            generated.append(nid)
            ctx.append(nid)

            if no_repeat_ngram > 1 and len(ctx) >= no_repeat_ngram:
                pref = tuple(ctx[-(no_repeat_ngram):-(1)])
                ngram_idx.setdefault(pref, set()).add(nid)

            tok = torch.tensor([[nid]], dtype=torch.long)
            x_new = self.drop(self.tok_emb(tok))
            new_kv_list = []
            for i, block in enumerate(self.blocks):
                x_new, kv = block(x_new, kv_cache=kv_caches[i])
                new_kv_list.append(kv)
            kv_caches = new_kv_list
            x = self.norm_f(x_new)

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


# ══════════════════════════════════════════════════════════════════════════════
#  Legacy architecture — оригинальная MiniGPT (pos_emb, LayerNorm, GELU)
# ══════════════════════════════════════════════════════════════════════════════


class _LegacyAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, seq_len: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        mask = torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len)
        self.register_buffer("mask", mask)

    def forward(self, x, kv_cache=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out), None


class _LegacyBlock(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, dropout: float, seq_len: int
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = _LegacyAttention(d_model, n_heads, seq_len)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, kv_cache=None):
        a, _ = self.attn(self.ln1(x))
        x = x + a
        x = x + self.ff(self.ln2(x))
        return x, None


class _LegacyMiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, dropout, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                _LegacyBlock(d_model, n_heads, d_ff, dropout, seq_len)
                for _ in range(n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x, _ = block(x)
        return self.head(self.ln_f(x))

    @torch.no_grad()
    def generate(
        self,
        prompt,
        max_new,
        temperature=1.0,
        top_k=50,
        top_p=0.92,
        eos_id=2,
        vocab_size=None,
        repetition_penalty=1.3,
        no_repeat_ngram=3,
        min_new_tokens=5,
    ):
        vs = vocab_size or self.tok_emb.num_embeddings
        ctx = [min(max(t, 0), vs - 1) for t in prompt]
        self.eval()
        generated = []
        ngram_idx: dict = {}
        if no_repeat_ngram > 1:
            n = no_repeat_ngram
            for i in range(len(ctx) - n + 1):
                pref = tuple(ctx[i : i + n - 1])
                ngram_idx.setdefault(pref, set()).add(ctx[i + n - 1])

        for step in range(max_new):
            inp = torch.tensor([ctx[-self.seq_len :]], dtype=torch.long).clamp(
                0, vs - 1
            )
            logits = self.forward(inp)[0, -1, :vs].float()

            if repetition_penalty != 1.0 and ctx:
                seen = set(ctx[-64:])
                for tid in seen:
                    if 0 <= tid < vs:
                        logits[tid] = (
                            logits[tid] / repetition_penalty
                            if logits[tid] > 0
                            else logits[tid] * repetition_penalty
                        )

            if no_repeat_ngram > 1 and len(ctx) >= no_repeat_ngram - 1:
                pref = tuple(ctx[-(no_repeat_ngram - 1) :])
                for tid in ngram_idx.get(pref, set()):
                    if 0 <= tid < vs:
                        logits[tid] = float("-inf")

            logits = logits / max(temperature, 1e-8)
            if top_k > 0:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[-1]
                logits[logits < kth] = float("-inf")

            probs = F.softmax(logits, dim=-1).cpu().numpy().astype(np.float64)
            probs = np.clip(probs, 0, None)

            if top_p < 1.0:
                si = np.argsort(probs)[::-1]
                cum = np.cumsum(probs[si])
                cut = int(np.searchsorted(cum, top_p)) + 1
                mask = np.zeros_like(probs)
                mask[si[:cut]] = probs[si[:cut]]
                s = mask.sum()
                if s > 0:
                    probs = mask / s

            total = probs.sum()
            probs = probs / total if total > 0 else np.ones(vs, dtype=np.float64) / vs
            nid = int(np.random.choice(len(probs), p=probs))

            if nid == eos_id and step >= min_new_tokens:
                break

            generated.append(nid)
            ctx.append(nid)

            if no_repeat_ngram > 1 and len(ctx) >= no_repeat_ngram:
                pref = tuple(ctx[-no_repeat_ngram:-1])
                ngram_idx.setdefault(pref, set()).add(nid)

        return generated


def _is_legacy_checkpoint(state_dict: dict) -> bool:
    return "pos_emb.weight" in state_dict or "ln_f.weight" in state_dict


def build_legacy(vocab_size: int, params: dict) -> _LegacyMiniGPT:
    return _LegacyMiniGPT(
        vocab_size=vocab_size,
        d_model=params["d_model"],
        n_heads=params["n_heads"],
        n_layers=params["n_layers"],
        d_ff=params.get("d_ff", params["d_model"] * 4),
        dropout=params.get("dropout", 0.1),
        seq_len=params["seq_len"],
    )


def build_auto(vocab_size: int, params: dict, state_dict: dict):
    """
    Автоматически выбрать архитектуру по state_dict и вернуть загруженную модель.
    strict=False — не падает на буферах (mask, freqs_cis) и мелких расхождениях.
    """
    is_legacy = _is_legacy_checkpoint(state_dict)
    if is_legacy:
        model = build_legacy(vocab_size, params)
    else:
        model = build(vocab_size, params)

    result = model.load_state_dict(state_dict, strict=False)

    # Игнорируем расхождения только по буферам — они пересчитываются при forward
    _BUFFERS = ("freqs_cis", "mask")
    missing = [k for k in result.missing_keys if not any(b in k for b in _BUFFERS)]
    unexpected = [
        k for k in result.unexpected_keys if not any(b in k for b in _BUFFERS)
    ]

    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing keys: {missing}")
        if unexpected:
            details.append(f"unexpected keys: {unexpected}")
        raise RuntimeError(
            "Checkpoint is incompatible with selected MiniGPT architecture: "
            + "; ".join(details)
        )

    model.eval()
    return model
