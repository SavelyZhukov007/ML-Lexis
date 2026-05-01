"""
core/postprocess.py — агенты постобработки текста.

Агент 1: clean_generated         — мусор, скобки, спецтокены, двойные знаки
Агент 2: fix_agreement           — грамматика (pymorphy3)
Агент 3: fix_punctuation         — пунктуация
Агент 4: deduplicate_sentences   — точные + near-duplicate (Жаккар)
Агент 5: fix_repeating_words     — "кивнул и кивнул" → "кивнул"
Агент 6: fix_sentence_fragments  — обрывки, повисшие союзы
Агент 7: normalize_style         — унификация разговорных оборотов
"""

import re


# ═══════════════════════════════════════════════
#  Агент 1: Очистка мусора
# ═══════════════════════════════════════════════


def clean_generated(text: str) -> str:
    text = re.sub(r"<(?:PAD|BOS|EOS|UNK)>", "", text, flags=re.IGNORECASE)

    # Парные скобки
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

    # Пустые скобки
    for _ in range(3):
        text = re.sub(r"\(\s*\)", "", text)
        text = re.sub(r"«\s*»", "", text)

    # Пробелы вокруг знаков
    text = re.sub(r"\s+([.,!?;:»)\]])", r"\1", text)
    text = re.sub(r"([«(\[])\s+", r"\1", text)

    # Двойные знаки подряд
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


# ═══════════════════════════════════════════════
#  Агент 2: Грамматика (pymorphy3)
# ═══════════════════════════════════════════════

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


def fix_agreement(text: str) -> str:
    morph = _get_morph()
    if not morph:
        return text

    tokens = text.split()
    result = list(tokens)

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

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        clean_tok = re.sub(r"[^\wа-яёА-ЯЁ]", "", tok, flags=re.UNICODE)
        tok_lower = tok.lower().rstrip(".,!?;:")

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

        if tok_lower in PREP_CASE and i + 1 < len(tokens):
            nxt_raw = tokens[i + 1]
            nxt_word = re.sub(r"[^\wа-яёА-ЯЁ]", "", nxt_raw, flags=re.UNICODE)
            if nxt_word and re.search(r"[а-яё]", nxt_word, re.IGNORECASE):
                parses = morph.parse(nxt_word)
                if parses and "NOUN" in parses[0].tag:
                    nw = _inflect(morph, nxt_word, {PREP_CASE[tok_lower]})
                    if nw:
                        result[i + 1] = nw + nxt_raw[len(nxt_word) :]

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
                        tag = noun_p[0].tag
                        grms = set()
                        for attr in ("gend", "numb", "case"):
                            v = getattr(tag, attr, None)
                            if v:
                                grms.add(v)
                        if grms:
                            nw = _inflect(morph, clean_tok, grms)
                            if nw:
                                result[i] = nw + tok[len(clean_tok) :]
        i += 1

    return " ".join(result)


# ═══════════════════════════════════════════════
#  Агент 3: Пунктуация
# ═══════════════════════════════════════════════

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

    def _cap(m):
        return m.group(1) + " " + m.group(2).upper()

    text = re.sub(r"([.!?])\s+([а-яёa-z])", _cap, text)
    text = re.sub(r",\s*([.!?…])", r"\1", text)
    text = re.sub(r"(?<=\w)\s+-\s+(?=\w)", " — ", text)

    text = text.strip()
    if text and text[-1] not in ".!?…":
        text += "."
    if text:
        text = text[0].upper() + text[1:]
    return text


# ═══════════════════════════════════════════════
#  Агент 4: Дедупликация предложений
# ═══════════════════════════════════════════════


def _split_sentences(text: str) -> list[str]:
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

    seen_exact: set[str] = set()
    seen_fuzzy: list[str] = []
    result: list[str] = []

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


# ═══════════════════════════════════════════════
#  Агент 5: Повторяющиеся слова и фразы
# ═══════════════════════════════════════════════


def fix_repeating_words(text: str) -> str:
    """
    Убирает непосредственные повторы слов:
      "кивнул и кивнул и кивнул" → "кивнул"
      "снова снова снова" → "снова"
      "он он посмотрел" → "он посмотрел"

    Также убирает повтор через союз "и":
      "посмотрел и посмотрел" → "посмотрел"
    """
    # 1. Прямые повторы подряд: слово слово → слово
    for _ in range(5):
        text = re.sub(r"\b([а-яёa-zА-ЯЁA-Z]+)\s+\1\b", r"\1", text, flags=re.IGNORECASE)

    # 2. Повтор через "и": "X и X" → "X"
    for _ in range(3):
        text = re.sub(
            r"\b([а-яёa-zА-ЯЁA-Z]+)\s+и\s+\1\b", r"\1", text, flags=re.IGNORECASE
        )

    # 3. Повтор через "и снова": "кивнул и снова кивнул" → "кивнул"
    for _ in range(3):
        text = re.sub(
            r"\b([а-яёa-zА-ЯЁA-Z]+)\s+и\s+снова\s+\1\b",
            r"\1",
            text,
            flags=re.IGNORECASE,
        )

    # 4. "снова снова" → "снова"
    text = re.sub(r"\bснова(\s+снова)+\b", "снова", text, flags=re.IGNORECASE)
    text = re.sub(r"\bопять(\s+опять)+\b", "опять", text, flags=re.IGNORECASE)
    text = re.sub(r"\bеще(\s+еще)+\b", "еще", text, flags=re.IGNORECASE)
    text = re.sub(r"\bуже(\s+уже)+\b", "уже", text, flags=re.IGNORECASE)

    # 5. Повтор глагол + повернулся/кивнул + к + местоимение
    #    "повернулся ко мне и повернулся ко мне" → "повернулся ко мне"
    text = re.sub(r"((?:[а-яёА-ЯЁ]+\s+){1,4})и\s+\1", r"\1", text, flags=re.IGNORECASE)

    return text


# ═══════════════════════════════════════════════
#  Агент 6: Фрагменты и обрывки
# ═══════════════════════════════════════════════

# Союзы в начале предложения (нежелательно как начало отдельного предложения)
_HANGING_START = re.compile(
    r"(?<=[.!?]\s)(и|а|но|или|да|ни|же|то|ведь|лишь|только)\s+", re.IGNORECASE
)
# Предложение-обрывок: только 1-2 слова
_FRAGMENT = re.compile(r"(?<=[.!?]\s)([А-ЯЁA-Z][а-яёa-z]+\.)\s+")


def fix_sentence_fragments(text: str) -> str:
    """
    1. Убирает одиночные слова, оставшиеся как предложения
       ("Да." / "Нет." в одиночестве → остаются, но "он." → присоединяем)
    2. Убирает зависший союз в начале предложения и сливает с предыдущим
    """
    # Сливаем слишком короткие "предложения" (1 слово, не диалог)
    sentences = _split_sentences(text)
    result = []
    i = 0
    while i < len(sentences):
        sent = sentences[i]
        words = [w for w in re.split(r"\s+", sent.strip(".,!? ")) if w]
        # Если предложение из 1-2 слов и следующее есть — присоединить к следующему
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

    # Убираем зависший союз в начале после знака
    text = _HANGING_START.sub("", text)

    return text


# ═══════════════════════════════════════════════
#  Агент 7: Нормализация стиля
# ═══════════════════════════════════════════════

# Паразиты и тавтологии, характерные для плохой генерации
_PARASITES = [
    (r"\bто\s+же\s+самое\b", "это"),
    (r"\bтем\s+не\s+менее\s+однако\b", "тем не менее"),
    (r"\bкак\s+будто\s+бы\b", "как будто"),
    (r"\bтак\s+же\s+как\s+и\s+раньше\b", "как прежде"),
    (r"\bпо\s+крайней\s+мере\s+хотя\s+бы\b", "хотя бы"),
    # "я я знаю" → "я знаю"
    (r"\bя\s+я\b", "я"),
    (r"\bон\s+он\b", "он"),
    (r"\bона\s+она\b", "она"),
    # Троеточие-паразит в середине
    (r"(?<=\w)\s*\.\.\.\s*(?=\w)", " "),
    # "что что" → "что"
    (r"\b(что|как|когда|если|но|и)\s+\1\b", r"\1"),
]


def normalize_style(text: str) -> str:
    for pattern, replacement in _PARASITES:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    # Убираем висящие союзы в конце предложения перед знаком
    text = re.sub(r"\s+(и|а|но|или)\s*([.!?])", r"\2", text, flags=re.IGNORECASE)
    return text


# ═══════════════════════════════════════════════
#  Полный постпроцессинг (все 7 агентов)
# ═══════════════════════════════════════════════


def postprocess(text: str, corrections: dict = None) -> str:
    text = fix_repeating_words(text)  # 5 — сначала, пока ещё есть повторы
    text = normalize_style(text)  # 7
    text = clean_generated(text)  # 1
    text = fix_agreement(text)  # 2
    text = fix_punctuation(text)  # 3
    text = deduplicate_sentences(text)  # 4
    text = fix_sentence_fragments(text)  # 6
    # Финальная очистка после всех агентов
    text = clean_generated(text)
    if corrections:
        for wrong, correct in corrections.items():
            text = re.sub(
                r"\b" + re.escape(wrong) + r"\b", correct, text, flags=re.IGNORECASE
            )
    return text


# ═══════════════════════════════════════════════
#  DPA (Deep Processing Algorithm)
# ═══════════════════════════════════════════════


def _adapt_params(temperature, top_k, top_p):
    if temperature > 1.1:
        return 0.85, min(top_k, 40), min(top_p, 0.9)
    elif temperature < 0.7:
        return temperature, max(top_k, 20), min(top_p, 0.85)
    return temperature, top_k, top_p


def _words_of(text: str) -> list[str]:
    return [w for w in re.split(r"\s+", text.strip()) if w]


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
    import torch

    words = _words_of(text)
    if len(words) < 100:
        return postprocess(text, corrections)

    w2i = vocab["w2i"]
    i2w = vocab["i2w"]
    bos = w2i.get("<BOS>", 1)
    eos = w2i.get("<EOS>", 2)
    vs = ckpt_vs or len(w2i)

    temp, tk, tp = _adapt_params(temperature, top_k, top_p)

    def _tok(word):
        return w2i.get(word.lower(), bos)

    def _gen_candidates(ctx_words, n=10):
        prompt = [_tok(w) for w in ctx_words[-model.seq_len :]] or [bos]
        cands = set()
        for _ in range(25):
            if len(cands) >= n:
                break
            gen = model.generate(
                prompt,
                max_new=1,
                temperature=temp + 0.15,
                top_k=min(tk, 30),
                top_p=tp,
                eos_id=eos,
                vocab_size=vs,
                repetition_penalty=1.4,
                no_repeat_ngram=3,
                min_new_tokens=1,
            )
            if gen:
                w = i2w.get(gen[0], "")
                if w and re.search(r"[а-яёa-z]", w, re.IGNORECASE):
                    cands.add(w)
        return list(cands)

    def _best_by_prob(ctx_words, candidates):
        if not candidates:
            return ""
        prompt = [_tok(w) for w in ctx_words[-model.seq_len :]] or [bos]
        inp = torch.tensor([prompt], dtype=torch.long).clamp(0, vs - 1)
        with torch.no_grad():
            import torch.nn.functional as F

            logits = model(inp)[0, -1, :vs].float()
        probs = F.softmax(logits / max(temp, 1e-8), dim=-1)
        best_w, best_p = "", -1.0
        for c in candidates:
            idx = w2i.get(c.lower(), -1)
            if 0 <= idx < vs:
                p = probs[idx].item()
                if p > best_p:
                    best_p, best_w = p, c
        return best_w or candidates[0]

    sentences = _split_sentences(text)
    result_sents = []
    all_words_so_far: list[str] = []

    for sent in sentences:
        sw = _words_of(sent)
        if not sw:
            continue
        seed_word = sw[0]
        ctx_before = (all_words_so_far + [seed_word])[-10:]
        prompt_ids = [_tok(w) for w in ctx_before[-model.seq_len :]] or [bos]
        gen_ids = model.generate(
            prompt_ids,
            max_new=10,
            temperature=temp,
            top_k=tk,
            top_p=tp,
            eos_id=eos,
            vocab_size=vs,
            repetition_penalty=1.3,
            no_repeat_ngram=3,
            min_new_tokens=3,
        )
        gen_words = [i2w.get(g, "") for g in gen_ids if i2w.get(g, "")]
        orig_check = sw[1:11]
        refined = [seed_word]

        for pos, orig_w in enumerate(orig_check):
            gen_w = gen_words[pos] if pos < len(gen_words) else ""
            if gen_w.lower() == orig_w.lower():
                refined.append(orig_w)
            else:
                left_ctx = all_words_so_far + refined
                candidates = _gen_candidates(left_ctx, n=10)
                if orig_w:
                    candidates.append(orig_w)
                if gen_w:
                    candidates.append(gen_w)
                best = _best_by_prob(left_ctx, list(set(candidates)))
                refined.append(best if best else orig_w)

        if len(sw) > 11:
            refined.extend(sw[11:])

        result_sents.append(" ".join(refined))
        all_words_so_far.extend(refined)

    return postprocess(" ".join(result_sents), corrections)
