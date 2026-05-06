"""
core/postprocess.py — агенты постобработки + DPA v2 + Consciousness Algorithm

Агент 1: clean_generated         — мусор, скобки, спецтокены, двойные знаки
Агент 2: fix_agreement           — грамматика (pymorphy3)
Агент 3: fix_punctuation         — пунктуация
Агент 4: deduplicate_sentences   — точные + near-duplicate (Жаккар)
Агент 5: fix_repeating_words     — повторяющиеся слова и фразы
Агент 6: fix_sentence_fragments  — обрывки, повисшие союзы
Агент 7: normalize_style         — унификация разговорных оборотов
Агент 8: fix_capitalization      — заглавные после знаков препинания
Агент 9: lexical_diversity       — замена частых слов синонимами

DPA v2: эффективная батч-генерация без 25 вызовов per слово
Consciousness: многовариантная генерация + кросс-голосование
"""

import re
import torch
import torch.nn.functional as F

# ══════════════════════════════════════════════════════════════════════════════
#  Агент 1: Очистка мусора
# ══════════════════════════════════════════════════════════════════════════════


def clean_generated(text: str) -> str:
    text = re.sub(r"<(?:PAD|BOS|EOS|UNK)>", "", text, flags=re.IGNORECASE)
    result, depth = [], 0
    for ch in text:
        if ch == "(":
            depth += 1
            result.append(ch)
        elif ch == ")":
            if depth > 0:
                depth -= 1
                result.append(ch)
        else:
            result.append(ch)
    text = "".join(result)
    while text.count("(") > text.count(")"):
        idx = text.rfind("(")
        if idx == -1:
            break
        text = text[:idx] + text[idx + 1 :]
    while text.count("«") > text.count("»"):
        idx = text.rfind("«")
        if idx == -1:
            break
        text = text[:idx] + text[idx + 1 :]
    for _ in range(3):
        text = re.sub(r"\(\s*\)", "", text)
        text = re.sub(r"«\s*»", "", text)
    text = re.sub(r"\s+([.,!?;:»)\]])", r"\1", text)
    text = re.sub(r"([«(\[])\s+", r"\1", text)
    text = re.sub(r"([.!?])\1+", r"\1", text)
    text = re.sub(r"([,;:])\1+", r"\1", text)
    text = re.sub(
        r"([.,;:])([.,;:])",
        lambda m: m.group(1) if m.group(1) in ".!?" else m.group(2),
        text,
    )
    text = re.sub(r"([!?])[,;:.]+", r"\1", text)
    text = re.sub(r"[,;:.]+([!?])", r"\1", text)
    text = re.sub(r"—{2,}", "—", text)
    text = re.sub(r"-{3,}", "—", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    if text:
        text = text[0].upper() + text[1:]
    return text


# ══════════════════════════════════════════════════════════════════════════════
#  Агент 2: Грамматика (pymorphy3)
# ══════════════════════════════════════════════════════════════════════════════

_morph = None
_morph_ok = None


def _get_morph():
    global _morph, _morph_ok
    if _morph_ok is False:
        return None
    if _morph is not None:
        return _morph
    for pkg in ("pymorphy3", "pymorphy2"):
        try:
            mod = __import__(pkg)
            _morph = mod.MorphAnalyzer()
            _morph_ok = True
            return _morph
        except Exception:
            continue
    _morph_ok = False
    return None


def _inflect(morph, word: str, grammemes: set):
    parses = morph.parse(word)
    if not parses:
        return None
    form = parses[0].inflect(grammemes)
    return form.word if form else None


PREP_CASE = {
    "в": "loct",
    "во": "loct",
    "на": "loct",
    "при": "loct",
    "о": "loct",
    "об": "loct",
    "к": "datv",
    "по": "datv",
    "от": "ablt",
    "с": "ablt",
    "за": "ablt",
    "под": "ablt",
    "до": "gent",
    "из": "gent",
    "у": "gent",
    "для": "gent",
    "через": "accs",
    "про": "accs",
}


def fix_agreement(text: str) -> str:
    morph = _get_morph()
    if not morph:
        return text
    tokens = text.split()
    result = list(tokens)
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        clean_tok = re.sub(r"[^\wа-яёА-ЯЁ]", "", tok, flags=re.UNICODE)
        tok_lower = tok.lower().rstrip(".,!?;:")

        # Числительное + существительное
        m = re.fullmatch(r"(\d+)", tok)
        if m and i + 1 < len(tokens):
            num = int(m.group(1))
            nxt_raw = tokens[i + 1]
            nxt_word = re.sub(r"[^\wа-яёА-ЯЁ]", "", nxt_raw, flags=re.UNICODE)
            if nxt_word and re.search(r"[а-яё]", nxt_word, re.IGNORECASE):
                parses = morph.parse(nxt_word)
                if parses and ("NOUN" in parses[0].tag or "ADJF" in parses[0].tag):
                    last2, last1 = num % 100, num % 10
                    if 11 <= last2 <= 19:
                        grms = {"gent", "plur"}
                    elif last1 == 1:
                        grms = {"nomn", "sing"}
                    elif last1 in (2, 3, 4):
                        grms = {"gent", "sing"}
                    else:
                        grms = {"gent", "plur"}
                    nw = _inflect(morph, nxt_word, grms)
                    if nw:
                        result[i + 1] = nw + nxt_raw[len(nxt_word) :]

        # Предлог + существительное
        if tok_lower in PREP_CASE and i + 1 < len(tokens):
            nxt_raw = tokens[i + 1]
            nxt_word = re.sub(r"[^\wа-яёА-ЯЁ]", "", nxt_raw, flags=re.UNICODE)
            if nxt_word and re.search(r"[а-яё]", nxt_word, re.IGNORECASE):
                parses = morph.parse(nxt_word)
                if parses and "NOUN" in parses[0].tag:
                    nw = _inflect(morph, nxt_word, {PREP_CASE[tok_lower]})
                    if nw:
                        result[i + 1] = nw + nxt_raw[len(nxt_word) :]

        # Прилагательное + существительное (согласование рода/числа/падежа)
        if (
            clean_tok
            and re.search(r"[а-яё]", clean_tok, re.IGNORECASE)
            and i + 1 < len(tokens)
        ):
            parses = morph.parse(clean_tok)
            if parses and "ADJF" in parses[0].tag:
                nxt_raw = tokens[i + 1]
                nxt_word = re.sub(r"[^\wа-яёА-ЯЁ]", "", nxt_raw, flags=re.UNICODE)
                if nxt_word:
                    noun_p = morph.parse(nxt_word)
                    if noun_p and "NOUN" in noun_p[0].tag:
                        grms = set()
                        for attr in ("gend", "numb", "case"):
                            v = getattr(noun_p[0].tag, attr, None)
                            if v:
                                grms.add(v)
                        if grms:
                            nw = _inflect(morph, clean_tok, grms)
                            if nw:
                                result[i] = nw + tok[len(clean_tok) :]
        i += 1
    return " ".join(result)


# ══════════════════════════════════════════════════════════════════════════════
#  Агент 3: Пунктуация
# ══════════════════════════════════════════════════════════════════════════════

_COMMA_BEFORE = re.compile(
    r"(?<![,])\s+(но|а|зато|однако|хотя|потому\s+что|так\s+как|поэтому|"
    r"следовательно|если|пока|когда|как\s+только|чтобы)",
    re.IGNORECASE,
)
_INTRODUCTORY = re.compile(
    r"(?<![,])\s+(во-первых|во-вторых|наконец|например|то\s+есть|"
    r"следовательно|таким\s+образом|конечно|кстати|впрочем|однако)(?!\s*,)",
    re.IGNORECASE,
)


def fix_punctuation(text: str) -> str:
    text = re.sub(r"\s+([.,!?;:»)\]])", r"\1", text)
    text = re.sub(r"([«(\[])\s+", r"\1", text)
    text = _COMMA_BEFORE.sub(lambda m: ", " + m.group(1), text)
    text = _INTRODUCTORY.sub(lambda m: ", " + m.group(1).strip() + ",", text)
    text = re.sub(r"([.,!?;:])([а-яёА-ЯЁa-zA-Z0-9])", r"\1 \2", text)
    text = re.sub(
        r"([.!?])\s+([а-яёa-z])", lambda m: m.group(1) + " " + m.group(2).upper(), text
    )
    text = re.sub(r",\s*([.!?…])", r"\1", text)
    text = re.sub(r"(?<=\w)\s+-\s+(?=\w)", " — ", text)
    text = text.strip()
    if text and text[-1] not in ".!?…":
        text += "."
    if text:
        text = text[0].upper() + text[1:]
    return text


# ══════════════════════════════════════════════════════════════════════════════
#  Агент 4: Дедупликация предложений
# ══════════════════════════════════════════════════════════════════════════════


def _split_sentences(text: str) -> list:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _jaccard(a: str, b: str, n: int = 3) -> float:
    def ngrams(s):
        s = s.lower()
        return set(s[i : i + n] for i in range(max(0, len(s) - n + 1)))

    sa, sb = ngrams(a), ngrams(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def deduplicate_sentences(text: str, threshold: float = 0.65) -> str:
    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return text
    seen_exact, seen_fuzzy, result = set(), [], []
    for sent in sentences:
        norm = re.sub(r"\s+", " ", sent.lower().strip(".,!? "))
        if not norm or len(norm) < 3:
            continue
        if norm in seen_exact:
            continue
        if any(_jaccard(norm, prev) >= threshold for prev in seen_fuzzy):
            continue
        seen_exact.add(norm)
        seen_fuzzy.append(norm)
        result.append(sent)
    if not result:
        return text
    joined = " ".join(result)
    if not joined.endswith((".", "!", "?")):
        joined += "."
    return joined


# ══════════════════════════════════════════════════════════════════════════════
#  Агент 5: Повторяющиеся слова
# ══════════════════════════════════════════════════════════════════════════════


def fix_repeating_words(text: str) -> str:
    for _ in range(5):
        text = re.sub(r"\b([а-яёa-zА-ЯЁA-Z]+)\s+\1\b", r"\1", text, flags=re.IGNORECASE)
    for _ in range(3):
        text = re.sub(
            r"\b([а-яёa-zА-ЯЁA-Z]+)\s+и\s+\1\b", r"\1", text, flags=re.IGNORECASE
        )
    for _ in range(3):
        text = re.sub(
            r"\b([а-яёa-zА-ЯЁA-Z]+)\s+и\s+снова\s+\1\b",
            r"\1",
            text,
            flags=re.IGNORECASE,
        )
    text = re.sub(r"\bснова(\s+снова)+\b", "снова", text, flags=re.IGNORECASE)
    text = re.sub(r"\bопять(\s+опять)+\b", "опять", text, flags=re.IGNORECASE)
    text = re.sub(r"\bеще(\s+еще)+\b", "еще", text, flags=re.IGNORECASE)
    text = re.sub(r"\bуже(\s+уже)+\b", "уже", text, flags=re.IGNORECASE)
    text = re.sub(r"((?:[а-яёА-ЯЁ]+\s+){1,4})и\s+\1", r"\1", text, flags=re.IGNORECASE)
    return text


# ══════════════════════════════════════════════════════════════════════════════
#  Агент 6: Фрагменты и обрывки
# ══════════════════════════════════════════════════════════════════════════════

_HANGING_START = re.compile(
    r"(?<=[.!?]\s)(и|а|но|или|да|ни|же|то|ведь|лишь|только)\s+", re.IGNORECASE
)


def fix_sentence_fragments(text: str) -> str:
    sentences = _split_sentences(text)
    result = []
    i = 0
    while i < len(sentences):
        sent = sentences[i]
        words = [w for w in re.split(r"\s+", sent.strip(".,!? ")) if w]
        if len(words) <= 2 and i + 1 < len(sentences) and len(words) >= 1:
            clean = sent.rstrip(".!? ")
            sentences[i + 1] = (
                clean + ", " + sentences[i + 1][0].lower() + sentences[i + 1][1:]
            )
            i += 1
            continue
        result.append(sent)
        i += 1
    text = " ".join(result)
    text = _HANGING_START.sub("", text)
    return text


# ══════════════════════════════════════════════════════════════════════════════
#  Агент 7: Нормализация стиля
# ══════════════════════════════════════════════════════════════════════════════

_PARASITES = [
    (r"\bто\s+же\s+самое\b", "это"),
    (r"\bтем\s+не\s+менее\s+однако\b", "тем не менее"),
    (r"\bкак\s+будто\s+бы\b", "как будто"),
    (r"\bтак\s+же\s+как\s+и\s+раньше\b", "как прежде"),
    (r"\bпо\s+крайней\s+мере\s+хотя\s+бы\b", "хотя бы"),
    (r"\bя\s+я\b", "я"),
    (r"\bон\s+он\b", "он"),
    (r"\bона\s+она\b", "она"),
    (r"(?<=\w)\s*\.\.\.\s*(?=\w)", " "),
    (r"\b(что|как|когда|если|но|и)\s+\1\b", r"\1"),
]


def normalize_style(text: str) -> str:
    for pattern, replacement in _PARASITES:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    text = re.sub(r"\s+(и|а|но|или)\s*([.!?])", r"\2", text, flags=re.IGNORECASE)
    return text


# ══════════════════════════════════════════════════════════════════════════════
#  Агент 8: Капитализация
# ══════════════════════════════════════════════════════════════════════════════


def fix_capitalization(text: str) -> str:
    text = re.sub(
        r"([.!?])\s+([а-яёa-z])",
        lambda m: m.group(1) + " " + m.group(2).upper(),
        text,
    )
    if text:
        text = text[0].upper() + text[1:]
    return text


# ══════════════════════════════════════════════════════════════════════════════
#  Агент 9: Лексическое разнообразие (простые синонимы частых слов)
# ══════════════════════════════════════════════════════════════════════════════

_COMMON_SYNONYMS = {
    r"\bсказал\b": ["произнёс", "ответил", "добавил", "заметил"],
    r"\bпошёл\b": ["направился", "двинулся", "зашагал"],
    r"\bсмотрел\b": ["глядел", "наблюдал", "всматривался"],
    r"\bбольшой\b": ["огромный", "крупный", "немаленький"],
    r"\bмаленький\b": ["небольшой", "крошечный", "миниатюрный"],
    r"\bхороший\b": ["отличный", "прекрасный", "замечательный"],
    r"\bплохой\b": ["скверный", "неудачный", "нехороший"],
}

import random


def lexical_diversity(text: str, rate: float = 0.3) -> str:
    words = text.split()
    result = []
    for word in words:
        replaced = False
        for pattern, synonyms in _COMMON_SYNONYMS.items():
            if re.search(pattern, word, re.IGNORECASE) and random.random() < rate:
                candidate = random.choice(synonyms)
                word_new = re.sub(pattern, candidate, word, flags=re.IGNORECASE)
                result.append(word_new)
                replaced = True
                break
        if not replaced:
            result.append(word)
    return " ".join(result)


# ══════════════════════════════════════════════════════════════════════════════
#  Полный постпроцессинг (все 9 агентов)
# ══════════════════════════════════════════════════════════════════════════════


def postprocess(text: str, corrections: dict = None) -> str:
    text = fix_repeating_words(text)
    text = normalize_style(text)
    text = clean_generated(text)
    text = fix_agreement(text)
    text = fix_punctuation(text)
    text = deduplicate_sentences(text)
    text = fix_sentence_fragments(text)
    text = fix_capitalization(text)
    text = clean_generated(text)
    if corrections:
        for wrong, correct in corrections.items():
            text = re.sub(
                r"\b" + re.escape(wrong) + r"\b", correct, text, flags=re.IGNORECASE
            )
    return text


# ══════════════════════════════════════════════════════════════════════════════
#  postprocess_selective — запускать только выбранные агенты
# ══════════════════════════════════════════════════════════════════════════════

_AGENT_FUNCS = {
    "repeats":   fix_repeating_words,
    "style":     normalize_style,
    "clean":     clean_generated,
    "grammar":   fix_agreement,
    "punct":     fix_punctuation,
    "dedup":     deduplicate_sentences,
    "fragments": fix_sentence_fragments,
    "caps":      fix_capitalization,
    "lexdiv":    lexical_diversity,
}

_AGENT_ORDER = ["repeats", "style", "clean", "grammar", "punct",
                "dedup", "fragments", "caps", "lexdiv"]


def postprocess_selective(text: str, active_agents: list = None,
                          corrections: dict = None) -> str:
    """Запускать только агенты из active_agents (по ключу из _AGENT_FUNCS)."""
    if active_agents is None:
        return postprocess(text, corrections)
    agents = active_agents if active_agents else list(_AGENT_FUNCS.keys())
    for key in _AGENT_ORDER:
        if key in agents and key in _AGENT_FUNCS:
            text = _AGENT_FUNCS[key](text)
    text = clean_generated(text)
    if corrections:
        for wrong, correct in corrections.items():
            text = re.sub(
                r"\b" + re.escape(wrong) + r"\b", correct, text, flags=re.IGNORECASE
            )
    return text


# ══════════════════════════════════════════════════════════════════════════════
#  postprocess_with_diff — постпроцессинг с пошаговым диффом по агентам
# ══════════════════════════════════════════════════════════════════════════════

def _word_diff_summary(before: str, after: str, max_items: int = 8) -> str:
    """Краткое описание изменений (старое→новое) для первых max_items отличий."""
    wb = before.split()
    wa = after.split()
    pairs = []
    for i in range(min(len(wb), len(wa))):
        if wb[i] != wa[i]:
            pairs.append(f"{wb[i]}→{wa[i]}")
            if len(pairs) >= max_items:
                break
    # Добавленные/удалённые слова
    if len(wa) > len(wb):
        extra = wa[len(wb):][:3]
        pairs += [f"+{w}" for w in extra]
    elif len(wb) > len(wa):
        removed = wb[len(wa):][:3]
        pairs += [f"-{w}" for w in removed]
    return ", ".join(pairs) if pairs else ""


def postprocess_with_diff(text: str, active_agents: list = None,
                          corrections: dict = None) -> dict:
    """
    Выполнить постпроцессинг с пошаговым логированием изменений каждого агента.

    Возвращает:
        {
            "text": str,          # итоговый текст
            "agent_diff": [       # список изменений по агентам
                {
                    "name": str,        # название агента
                    "key": str,         # ключ агента
                    "changes": int,     # число изменённых слов
                    "summary": str,     # краткое описание (старое→новое)
                    "changed": bool,    # был ли изменён текст
                },
                ...
            ]
        }
    """
    AGENT_NAMES = {
        "repeats":   "Повторы",
        "style":     "Стиль",
        "clean":     "Очистка",
        "grammar":   "Грамматика",
        "punct":     "Пунктуация",
        "dedup":     "Дедупликация",
        "fragments": "Фрагменты",
        "caps":      "Капитализация",
        "lexdiv":    "Разнообразие",
    }
    agents = active_agents if active_agents is not None else list(_AGENT_FUNCS.keys())
    diff_log = []
    current = text

    for key in _AGENT_ORDER:
        if key not in agents or key not in _AGENT_FUNCS:
            continue
        before = current
        current = _AGENT_FUNCS[key](current)
        # Count word-level changes
        wb, wa = before.split(), current.split()
        changes = sum(1 for a, b in zip(wb, wa) if a != b)
        changes += abs(len(wb) - len(wa))
        summary = _word_diff_summary(before, current) if changes else ""
        diff_log.append({
            "name": AGENT_NAMES.get(key, key),
            "key": key,
            "changes": changes,
            "summary": summary,
            "changed": current != before,
        })

    current = clean_generated(current)
    if corrections:
        before = current
        for wrong, correct in corrections.items():
            current = re.sub(
                r"\b" + re.escape(wrong) + r"\b", correct, current, flags=re.IGNORECASE
            )
        changes = sum(1 for a, b in zip(before.split(), current.split()) if a != b)
        diff_log.append({
            "name": "Замены",
            "key": "corrections",
            "changes": changes,
            "summary": _word_diff_summary(before, current) if changes else "",
            "changed": current != before,
        })

    return {"text": current, "agent_diff": diff_log}


# ══════════════════════════════════════════════════════════════════════════════
#  DPA v2 — Deep Processing Algorithm (батч-генерация, без цикла)
# ══════════════════════════════════════════════════════════════════════════════


def _words_of(text: str) -> list:
    return [w for w in re.split(r"\s+", text.strip()) if w]


def _tok_ids(words, w2i, bos):
    return [w2i.get(w.lower(), bos) for w in words]


def _batch_logits(model, ctx_ids: list, vs: int, device):
    """Получить логиты для следующего токена одним forward pass."""
    inp = torch.tensor([ctx_ids], dtype=torch.long).clamp(0, vs - 1).to(device)
    with torch.no_grad():
        logits = model(inp)[0, -1, :vs].float()
    return F.softmax(logits, dim=-1)


def _score_word(probs, word: str, w2i: dict) -> float:
    idx = w2i.get(word.lower(), -1)
    if idx < 0 or idx >= len(probs):
        return 0.0
    return probs[idx].item()


def deep_process(
    text,
    model,
    vocab,
    temperature=1.0,
    top_k=50,
    top_p=0.92,
    ckpt_vs=None,
    corrections=None,
) -> str:
    words = _words_of(text)
    if len(words) < 30:
        return postprocess(text, corrections)

    w2i = vocab["w2i"]
    i2w = vocab["i2w"]
    bos = w2i.get("<BOS>", 1)
    eos = w2i.get("<EOS>", 2)
    vs = ckpt_vs or len(w2i)

    # Адаптация температуры
    if temperature > 1.1:
        temperature = 0.9
        top_k = min(top_k, 40)
    elif temperature < 0.6:
        temperature = 0.6

    device = next(model.parameters()).device
    sentences = _split_sentences(text)
    result_sents = []
    global_ctx = [bos]

    for sent in sentences:
        sw = _words_of(sent)
        if not sw:
            continue

        refined = []
        ctx = global_ctx[-model.seq_len :]

        for pos, word in enumerate(sw):
            # Получаем распределение вероятностей для следующего слова
            probs = _batch_logits(model, ctx, vs, device)

            # Применяем temperature + top_k к вероятностям
            log_probs = torch.log(probs.clamp(min=1e-10)) / max(temperature, 1e-8)
            if top_k > 0:
                kth = torch.topk(log_probs, min(top_k, log_probs.size(-1))).values[-1]
                log_probs[log_probs < kth] = float("-inf")
            adjusted_probs = F.softmax(log_probs, dim=-1)

            # Генерируем N кандидатов из дискретного распределения
            topk_vals, topk_ids = torch.topk(adjusted_probs, min(15, vs))
            candidates = {}
            for prob_val, tid in zip(topk_vals.tolist(), topk_ids.tolist()):
                w = i2w.get(tid, "")
                if w and re.search(r"[а-яёa-zA-Z]", w):
                    candidates[w] = prob_val

            orig_score = _score_word(adjusted_probs, word, w2i)
            candidates[word] = max(candidates.get(word, 0), orig_score)

            if not candidates:
                chosen = word
            else:
                # Выбираем слово с наибольшей вероятностью среди кандидатов
                # с небольшим бонусом за оригинал (консервативная замена)
                candidates[word] = candidates.get(word, 0) * 1.1
                chosen = max(candidates, key=candidates.get)

            refined.append(chosen)
            tok_id = w2i.get(chosen.lower(), bos)
            ctx = (ctx + [tok_id])[-model.seq_len :]

        result_sents.append(" ".join(refined))
        global_ctx = (global_ctx + _tok_ids(refined, w2i, bos))[-model.seq_len :]

    refined_text = " ".join(result_sents)
    result = postprocess(refined_text, corrections)
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  CONSCIOUSNESS ALGORITHM — только для администратора
#
#  Идея: многовариантная параллельная генерация + кросс-голосование
#  1. Генерируем N гипотез (diverse beam-like с разными температурами)
#  2. Для каждой позиции голосуем: какое слово выбирает большинство гипотез
#  3. Применяем морфологическую согласованность (pymorphy3)
#  4. Оцениваем «осознанность»: разнообразие, связность, отсутствие артефактов
#  5. Итеративно рефинируем слабые предложения
# ══════════════════════════════════════════════════════════════════════════════


class ConsciousnessConfig:
    def __init__(
        self,
        n_hypotheses=5,
        refine_passes=2,
        diversity_weight=0.3,
        coherence_weight=0.5,
        quality_weight=0.2,
    ):
        self.n_hypotheses = n_hypotheses
        self.refine_passes = refine_passes
        self.diversity_weight = diversity_weight
        self.coherence_weight = coherence_weight
        self.quality_weight = quality_weight


def _generate_hypothesis(
    model, vocab, prompt_ids, max_new, temperature, top_k, top_p, ckpt_vs
):
    bos = vocab["w2i"].get("<BOS>", 1)
    eos = vocab["w2i"].get("<EOS>", 2)
    ids = model.generate(
        prompt_ids,
        max_new=max_new,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_id=eos,
        vocab_size=ckpt_vs,
        repetition_penalty=1.35,
        no_repeat_ngram=3,
        min_new_tokens=10,
    )
    return [vocab["i2w"].get(i, "") for i in ids if vocab["i2w"].get(i, "")]


def _score_hypothesis(words: list, w2i: dict) -> dict:
    """Многомерная оценка гипотезы."""
    if not words:
        return {"total": 0.0, "diversity": 0.0, "length": 0.0, "clean": 0.0}

    # Лексическое разнообразие
    unique_ratio = len(set(words)) / max(len(words), 1)

    # Штраф за короткие тексты
    length_score = min(1.0, len(words) / 30.0)

    # Штраф за артефакты (<PAD>, <BOS> и т.п.)
    artifacts = sum(1 for w in words if re.search(r"<[A-Z]+>", w))
    clean_score = 1.0 - artifacts / max(len(words), 1)

    # Знакомость слов (есть в словаре)
    known = sum(1 for w in words if w.lower() in w2i)
    vocab_score = known / max(len(words), 1)

    total = (
        unique_ratio * 0.3 + length_score * 0.2 + clean_score * 0.3 + vocab_score * 0.2
    )
    return {
        "total": total,
        "diversity": unique_ratio,
        "length": length_score,
        "clean": clean_score,
        "vocab": vocab_score,
    }


def _cross_vote_words(hypotheses: list, position: int, window: int = 3) -> dict:
    """
    Голосование за слово на позиции с учётом контекста.
    Каждая гипотеза голосует за своё слово на данной позиции.
    Вес голоса пропорционален качеству гипотезы.
    """
    votes: dict = {}
    for hyp_words, hyp_score in hypotheses:
        if position < len(hyp_words):
            word = hyp_words[position].lower()
            votes[word] = votes.get(word, 0) + hyp_score
    return votes


def _build_consensus_text(hypotheses: list, max_len: int) -> list:
    """
    Построение текста по принципу консенсуса:
    на каждой позиции выбираем слово с наибольшим суммарным весом.
    """
    result = []
    best_len = max(len(h[0]) for h in hypotheses) if hypotheses else 0
    best_len = min(best_len, max_len)

    for pos in range(best_len):
        votes = _cross_vote_words(hypotheses, pos)
        if not votes:
            break
        chosen = max(votes, key=votes.get)
        result.append(chosen)
    return result


def _morpho_refine(words: list) -> list:
    """Финальная морфологическая правка консенсус-текста."""
    morph = _get_morph()
    if not morph:
        return words
    text = " ".join(words)
    text = fix_agreement(text)
    return text.split()


def consciousness_generate(
    model,
    vocab,
    prompt_ids: list,
    max_new: int = 150,
    config: ConsciousnessConfig = None,
    ckpt_vs: int = None,
    corrections: dict = None,
) -> dict:
    """
    Consciousness Algorithm — admin-only.

    Возвращает:
        {
            "text": str,           # итоговый текст
            "hypotheses": list,    # все варианты
            "scores": list,        # оценки вариантов
            "consensus": str,      # консенсус без постпроцессинга
            "best": str,           # лучшая одиночная гипотеза
            "stats": dict,         # статистика
        }
    """
    if config is None:
        config = ConsciousnessConfig()

    vs = ckpt_vs or model.tok_emb.num_embeddings
    bos = vocab["w2i"].get("<BOS>", 1)

    # Параметры N разных гипотез — разные температуры и top_p
    hypothesis_params = [
        {"temperature": 0.6, "top_k": 30, "top_p": 0.85},
        {"temperature": 0.8, "top_k": 40, "top_p": 0.90},
        {"temperature": 1.0, "top_k": 50, "top_p": 0.92},
        {"temperature": 1.1, "top_k": 60, "top_p": 0.95},
        {"temperature": 1.3, "top_k": 80, "top_p": 0.98},
        {"temperature": 0.7, "top_k": 25, "top_p": 0.88},
        {"temperature": 0.9, "top_k": 45, "top_p": 0.91},
        {"temperature": 1.2, "top_k": 55, "top_p": 0.93},
    ][: config.n_hypotheses]

    model.eval()
    raw_hypotheses = []
    for p in hypothesis_params:
        words = _generate_hypothesis(model, vocab, prompt_ids, max_new, **p, ckpt_vs=vs)
        raw_hypotheses.append(words)

    # Оцениваем каждую гипотезу
    scored = []
    for words in raw_hypotheses:
        score_data = _score_hypothesis(words, vocab["w2i"])
        scored.append((words, score_data["total"]))

    scored.sort(key=lambda x: x[1], reverse=True)

    # Строим консенсус
    consensus_words = _build_consensus_text(scored, max_len=max_new)
    consensus_words = _morpho_refine(consensus_words)
    consensus_text = " ".join(consensus_words)

    # Берём лучшую гипотезу
    best_words = scored[0][0] if scored else []
    best_text = " ".join(best_words)

    # Рефинирование консенсуса через DPA (итеративно)
    final_text = consensus_text
    for refine_pass in range(config.refine_passes):
        try:
            final_text = deep_process(
                final_text,
                model,
                vocab,
                temperature=0.7,
                top_k=30,
                top_p=0.88,
                ckpt_vs=vs,
                corrections=corrections,
            )
        except Exception:
            break

    # Финальный постпроцессинг
    final_text = postprocess(final_text, corrections)
    best_text = postprocess(best_text, corrections)

    # Применяем lexical_diversity для обогащения
    final_text = lexical_diversity(final_text, rate=0.2)

    all_hypotheses = []
    all_scores = []
    for words, score in scored:
        h_text = postprocess(" ".join(words), corrections)
        all_hypotheses.append(h_text)
        all_scores.append(round(score, 4))

    return {
        "text": final_text,
        "best": best_text,
        "consensus": postprocess(consensus_text, corrections),
        "hypotheses": all_hypotheses,
        "scores": all_scores,
        "stats": {
            "n_hypotheses": len(scored),
            "best_score": all_scores[0] if all_scores else 0,
            "avg_score": round(sum(all_scores) / max(len(all_scores), 1), 4),
            "consensus_len": len(consensus_words),
            "refine_passes": config.refine_passes,
        },
    }
