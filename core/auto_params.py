"""
core/auto_params.py — умная автонастройка параметров.

Ключевая метрика: tokens_per_vocab = n_tokens / vocab_size
  < 5   → модель будет переобучаться мгновенно, нужна крошечная архитектура
  5-15  → осторожно, нужна сильная регуляризация
  15-50 → нормально
  > 50  → можно позволить большую модель

Дополнительно учитывается:
  - avg_sentence_len для seq_len
  - соотношение vocab/tokens для оценки сложности задачи
  - real RAM estimate для batch_size
"""

import math


def _estimate_ram_mb(d_model, n_layers, d_ff, seq_len, batch_size, vocab_size) -> float:
    """Точная оценка RAM с учётом весов + активаций + оптимизатора AdamW."""
    # Параметры модели (float32)
    params = vocab_size * d_model * 2  # embeddings (tied)
    params += seq_len * d_model  # pos embeddings
    params += n_layers * (  # transformer blocks
        3 * d_model * d_model  # QKV
        + d_model * d_model  # proj
        + 2 * d_model * d_ff  # FFN
        + d_model * 6  # LayerNorm (2 × gamma + beta)
    )
    params += d_model * 2  # final LayerNorm
    params_bytes = params * 4  # float32

    # AdamW хранит gradient + m + v = 3 копии
    optimizer_bytes = params_bytes * 3

    # Активации при обучении (backward)
    # seq_len * d_model * n_layers * batch * 4 (fwd) * 2 (bwd)
    act_bytes = batch_size * seq_len * d_model * n_layers * 4 * 8

    # Attention matrix: batch * n_heads * seq^2 * 4
    n_heads_est = max(1, d_model // 64)
    att_bytes = batch_size * n_heads_est * seq_len * seq_len * 4 * n_layers

    total = params_bytes + optimizer_bytes + act_bytes + att_bytes
    return round(total / 1024 / 1024, 1)


def recommend_params(
    n_tokens: int,
    vocab_size: int,
    avg_sentence_len: float = 10.0,
    max_ram_mb: int = 900,
    n_sentences: int = 0,
) -> dict:
    """
    Возвращает оптимальные параметры на основе характеристик корпуса.

    Ключевая формула:
        capacity_ratio = n_tokens / vocab_size

    Смысл: сколько раз в среднем каждое слово встречается в данных.
    Чем меньше — тем меньше должна быть модель, иначе переобучение.
    """

    tpv = n_tokens / max(vocab_size, 1)  # tokens per vocab

    reasons = []
    reasons.append(
        f"tokens/vocab = {tpv:.1f} "
        f"({'критически мало, риск переобучения' if tpv < 7 else 'нормально' if tpv < 20 else 'хорошо'})"
    )

    # ── Архитектура по tokens_per_vocab ──────────────────────
    if tpv < 4:
        # Экстремально мало данных → минимальная модель
        d_model, n_heads, n_layers, d_ff = 64, 4, 1, 128
        reasons.append("Очень мало данных на слово → минимальная модель (64d, 1 слой)")
    elif tpv < 7:
        # Мало данных → крошечная модель с сильным дропаутом
        d_model, n_heads, n_layers, d_ff = 128, 4, 2, 256
        reasons.append("Мало данных на слово → компактная модель (128d, 2 слоя)")
    elif tpv < 15:
        # Умеренно — средняя модель
        d_model, n_heads, n_layers, d_ff = 128, 4, 3, 512
        reasons.append("Умеренный объём данных → небольшая модель (128d, 3 слоя)")
    elif tpv < 30:
        d_model, n_heads, n_layers, d_ff = 256, 8, 4, 1024
        reasons.append("Достаточно данных → стандартная модель (256d, 4 слоя)")
    elif tpv < 80:
        d_model, n_heads, n_layers, d_ff = 256, 8, 6, 1024
        reasons.append("Много данных → глубокая модель (256d, 6 слоёв)")
    else:
        d_model, n_heads, n_layers, d_ff = 512, 8, 6, 2048
        reasons.append("Большой датасет → полная модель (512d, 6 слоёв)")

    # ── seq_len ───────────────────────────────────────────────
    # Берём avg_sentence_len * 2 (достаточно для одного предложения + контекст)
    # но не больше 256 и не меньше 32
    seq_raw = int(avg_sentence_len * 2.5)
    seq_len = max(32, min(256, (seq_raw // 16) * 16))
    seq_len = max(seq_len, 32)
    reasons.append(
        f"seq_len={seq_len} (средняя длина предложения {avg_sentence_len:.1f} слов × 2.5)"
    )

    # ── Dropout: обратно пропорционален tpv ──────────────────
    if tpv < 5:
        dropout = 0.35  # экстремальная регуляризация
    elif tpv < 10:
        dropout = 0.25  # сильная
    elif tpv < 20:
        dropout = 0.2  # умеренная
    elif tpv < 50:
        dropout = 0.15
    else:
        dropout = 0.1

    # ── batch_size ────────────────────────────────────────────
    # Подбираем максимальный batch под ограничение RAM
    for bs_candidate in [64, 32, 16, 8, 4]:
        est = _estimate_ram_mb(
            d_model, n_layers, d_ff, seq_len, bs_candidate, vocab_size
        )
        if est <= max_ram_mb * 0.85:
            batch_size = bs_candidate
            break
    else:
        batch_size = 4

    # ── Learning rate ─────────────────────────────────────────
    # Меньше tpv → меньше lr чтобы не прыгать сразу в переобучение
    if tpv < 7:
        lr = 5e-4
    elif tpv < 20:
        lr = 3e-4
    elif tpv < 50:
        lr = 5e-4
    else:
        lr = 1e-3

    # Дополнительно: чем больше vocab, тем ниже lr
    if vocab_size > 20_000:
        lr = min(lr, 3e-4)
    if vocab_size > 50_000:
        lr = min(lr, 2e-4)

    # ── Проверка n_heads ─────────────────────────────────────
    while d_model % n_heads != 0 and n_heads > 1:
        n_heads -= 1

    # ── Финальная оценка RAM ──────────────────────────────────
    ram_mb = _estimate_ram_mb(d_model, n_layers, d_ff, seq_len, batch_size, vocab_size)

    # ── Предупреждения ────────────────────────────────────────
    warnings = []
    if tpv < 7:
        warnings.append(
            f"⚠️ tokens/vocab={tpv:.1f} — данных очень мало. "
            f"Ожидай переобучение после 1–3 эпох. "
            f"Решение: добавь больше текстов в датасет."
        )
    if n_tokens < 10_000:
        warnings.append(
            f"⚠️ Всего {n_tokens:,} токенов — слишком мало для качественной генерации. "
            f"Рекомендуется минимум 50К токенов."
        )

    reasons.append(f"dropout={dropout:.2f} (регуляризация под tpv={tpv:.1f})")
    reasons.append(f"batch_size={batch_size}, RAM ~{ram_mb} МБ")
    reasons.append(f"lr={lr:.0e}")

    params = {
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "d_ff": d_ff,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "dropout": dropout,
        "lr": lr,
        "name": "Custom (auto)",
    }

    return {
        "params": params,
        "ram_mb": ram_mb,
        "reasons": reasons,
        "warnings": warnings,
        "tpv": round(tpv, 2),
    }


def analyze_corpus(text: str) -> dict:
    """Анализирует текст и возвращает статистику корпуса."""
    import re

    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    words_per_sent = [len(re.findall(r"[а-яёa-zA-ZА-ЯЁ]+", s)) for s in sentences if s]
    total_words = sum(words_per_sent)
    avg_sent_len = (total_words / len(words_per_sent)) if words_per_sent else 10.0

    # Уникальные слова в тексте
    all_words = re.findall(r"[а-яёa-zA-ZА-ЯЁ]+", text.lower())
    unique_ratio = len(set(all_words)) / max(len(all_words), 1)

    # Средняя длина слова (косвенная оценка сложности)
    avg_word_len = sum(len(w) for w in all_words) / max(len(all_words), 1)

    return {
        "n_sentences": len(sentences),
        "avg_sent_len": round(avg_sent_len, 1),
        "min_sent_len": min(words_per_sent) if words_per_sent else 0,
        "max_sent_len": max(words_per_sent) if words_per_sent else 0,
        "total_words": total_words,
        "unique_ratio": round(unique_ratio, 3),
        "avg_word_len": round(avg_word_len, 1),
    }
