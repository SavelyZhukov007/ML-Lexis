"""
dataset.py — Сборщик русскоязычных датасетов для Lexis
═══════════════════════════════════════════════════════
Запуск:  python dataset.py
Зависимости устанавливаются автоматически при первом запуске.

Создаёт файлы в корне проекта:
  wiki.txt      — статьи Википедии (~0.5 ГБ)
  proza.txt     — художественная проза (~0.5 ГБ)
  news.txt      — новостные статьи (~0.5 ГБ)
  science.txt   — научпоп и научные тексты (~0.5 ГБ)
  dialog.txt    — диалоги и субтитры (~0.5 ГБ)
"""

# ══════════════════════════════════════════════════════════════════
#  Авто-установка зависимостей
# ══════════════════════════════════════════════════════════════════
import subprocess, sys, importlib


def _install(*packages):
    for pkg in packages:
        mod = pkg.split("==")[0].replace("-", "_")
        try:
            importlib.import_module(mod)
        except ImportError:
            print(f"  [install] {pkg}...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pkg, "-q"],
                stdout=subprocess.DEVNULL,
            )


_install("datasets", "requests", "tqdm", "beautifulsoup4", "lxml")

# ══════════════════════════════════════════════════════════════════
#  Импорты
# ══════════════════════════════════════════════════════════════════
import re
import os
import json
import time
import random
import hashlib
import unicodedata
from pathlib import Path
from typing import Iterator, Optional

import requests
from tqdm import tqdm

# ══════════════════════════════════════════════════════════════════
#  Конфиг
# ══════════════════════════════════════════════════════════════════
TARGET_GB = 0.5  # целевой размер каждого файла в ГБ
TARGET_BYTES = int(TARGET_GB * 1024**3)
OUT_DIR = Path(".")  # куда писать (текущая папка)
MIN_SENT_LEN = 30  # минимальная длина предложения (символов)
MAX_SENT_LEN = 2000  # максимальная длина предложения
CACHE_DIR = Path(".dataset_cache")  # кэш скачанных данных
CACHE_DIR.mkdir(exist_ok=True)

# ══════════════════════════════════════════════════════════════════
#  Утилиты очистки текста
# ══════════════════════════════════════════════════════════════════
_RU_CHARS = re.compile(r"[а-яёА-ЯЁ]")
_GARBAGE = re.compile(
    r"(?:"
    r"https?://\S+"  # ссылки
    r"|www\.\S+"
    r"|\{[^}]*\}"  # шаблоны wiki
    r"|\[[^\]]*\]"  # сноски [1], [[ссылки]]
    r"|<[^>]+>"  # HTML теги
    r"|&\w+;"  # HTML entities
    r"|={2,}[^=]*={2,}"  # заголовки wiki
    r"|\*{2,}"  # маркеры
    r"|#{1,6}\s"  # markdown заголовки
    r"|^\s*[|!].*$"  # таблицы wiki
    r")",
    re.MULTILINE,
)
_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_MULTI_NL = re.compile(r"\n{3,}")
_BRACKETS_NUM = re.compile(r"\[\d+\]")


def clean(text: str) -> str:
    """Универсальная очистка текста."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = _GARBAGE.sub(" ", text)
    text = _BRACKETS_NUM.sub("", text)
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NL.sub("\n\n", text)
    text = text.strip()
    return text


def is_good_line(line: str) -> bool:
    """Фильтр качества: достаточно русского текста, не мусор."""
    line = line.strip()
    if len(line) < MIN_SENT_LEN or len(line) > MAX_SENT_LEN:
        return False
    ru = len(_RU_CHARS.findall(line))
    total = len(line.replace(" ", ""))
    if total == 0:
        return False
    ratio = ru / total
    if ratio < 0.55:  # меньше 55% русских букв → пропуск
        return False
    # Слишком много цифр — таблица или список
    digits = sum(c.isdigit() for c in line)
    if digits / total > 0.3:
        return False
    return True


def split_to_paragraphs(text: str) -> list[str]:
    """Разбить текст на абзацы, отфильтровать плохие."""
    paragraphs = []
    for block in re.split(r"\n{2,}", text):
        block = block.strip()
        lines = [l for l in block.split("\n") if is_good_line(l)]
        if lines:
            paragraphs.append(" ".join(lines))
    return paragraphs


# ══════════════════════════════════════════════════════════════════
#  Дедупликация через хэши
# ══════════════════════════════════════════════════════════════════
class DedupSet:
    def __init__(self):
        self._seen: set[str] = set()

    def is_new(self, text: str) -> bool:
        h = hashlib.md5(text[:200].encode()).hexdigest()
        if h in self._seen:
            return False
        self._seen.add(h)
        return True


# ══════════════════════════════════════════════════════════════════
#  Писатель файлов с прогрессом
# ══════════════════════════════════════════════════════════════════
class DatasetWriter:
    def __init__(self, path: Path, target_bytes: int, label: str):
        self.path = path
        self.target = target_bytes
        self.label = label
        self.written = 0
        self.count = 0
        self.dedup = DedupSet()
        self._f = open(path, "w", encoding="utf-8")
        self._bar = tqdm(
            total=target_bytes,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=f"  {label:<12}",
            colour="green",
        )

    @property
    def done(self) -> bool:
        return self.written >= self.target

    def write(self, text: str) -> bool:
        """Записать текст если он новый и качественный. Вернуть True если записан."""
        if self.done:
            return False
        text = clean(text).strip()
        if not text or not self.dedup.is_new(text):
            return False
        paras = split_to_paragraphs(text)
        if not paras:
            return False
        block = "\n\n".join(paras) + "\n\n"
        b = block.encode("utf-8")
        self._f.write(block)
        self._f.flush()
        self.written += len(b)
        self.count += 1
        self._bar.update(len(b))
        return True

    def close(self):
        self._bar.close()
        self._f.close()
        mb = self.written / 1024**2
        print(
            f"  ✓ {self.label}: {self.count:,} блоков, {mb:.1f} МБ → {self.path.name}"
        )


# ══════════════════════════════════════════════════════════════════
#  Источник 1: Википедия (через HuggingFace datasets)
# ══════════════════════════════════════════════════════════════════
def source_wikipedia() -> Iterator[str]:
    """Статьи русской Википедии через HuggingFace."""
    try:
        from datasets import load_dataset

        print("  → Загрузка Wikipedia (ru)...")
        ds = load_dataset(
            "wikipedia",
            "20220301.ru",
            split="train",
            trust_remote_code=True,
            cache_dir=str(CACHE_DIR),
        )
        for row in ds:
            txt = row.get("text", "")
            if txt and len(txt) > 200:
                yield txt
    except Exception as e:
        print(f"  [!] Wikipedia HF failed: {e}")
        yield from source_wikipedia_api()


def source_wikipedia_api() -> Iterator[str]:
    """Запасной вариант: Mediawiki API — рандомные статьи."""
    print("  → Fallback: Wikipedia API (random articles)...")
    session = requests.Session()
    session.headers["User-Agent"] = "LexisDataset/1.0 (research)"
    seen_ids: set[int] = set()

    api = "https://ru.wikipedia.org/w/api.php"
    while True:
        try:
            # 20 случайных статей за раз
            r = session.get(
                api,
                params={
                    "action": "query",
                    "format": "json",
                    "list": "random",
                    "rnnamespace": 0,
                    "rnlimit": 20,
                },
                timeout=15,
            )
            pages = r.json()["query"]["random"]
            page_ids = [p["id"] for p in pages if p["id"] not in seen_ids]
            seen_ids.update(page_ids)
            if not page_ids:
                time.sleep(1)
                continue

            # Получить текст страниц
            r2 = session.get(
                api,
                params={
                    "action": "query",
                    "format": "json",
                    "pageids": "|".join(map(str, page_ids)),
                    "prop": "extracts",
                    "explaintext": True,
                    "exsectionformat": "plain",
                },
                timeout=30,
            )
            result = r2.json().get("query", {}).get("pages", {})
            for page in result.values():
                txt = page.get("extract", "")
                if txt and len(txt) > 300:
                    yield txt
            time.sleep(0.5)
        except Exception as e:
            print(f"  [!] Wikipedia API error: {e}")
            time.sleep(3)


# ══════════════════════════════════════════════════════════════════
#  Источник 2: Художественная проза (несколько HF датасетов)
# ══════════════════════════════════════════════════════════════════
def source_proza() -> Iterator[str]:
    """Художественная проза: несколько источников."""
    from datasets import load_dataset

    # 2a. IlyaGusev/ficbook — фанфики и проза с Ficbook.net
    try:
        print("  → Загрузка ficbook (проза)...")
        ds = load_dataset(
            "IlyaGusev/ficbook",
            split="train",
            trust_remote_code=True,
            cache_dir=str(CACHE_DIR),
        )
        for row in ds:
            for field in ("text", "content", "story"):
                txt = row.get(field, "")
                if txt and len(txt) > 300:
                    yield txt
                    break
    except Exception as e:
        print(f"  [!] ficbook: {e}")

    # 2b. Классическая литература через API Lib.ru
    yield from source_libru()

    # 2c. mc4 Russian (разнообразный веб-текст)
    try:
        print("  → Загрузка mc4 (ru, веб-проза)...")
        ds = load_dataset(
            "mc4",
            "ru",
            split="train",
            streaming=True,
            trust_remote_code=True,
            cache_dir=str(CACHE_DIR),
        )
        for row in ds:
            txt = row.get("text", "")
            if txt and len(txt) > 500:
                yield txt
    except Exception as e:
        print(f"  [!] mc4: {e}")


def source_libru() -> Iterator[str]:
    """
    Lib.ru — открытая библиотека классики.
    Простой обход нескольких известных авторов через HTTP.
    """
    print("  → Lib.ru (классика)...")
    # Прямые ссылки на .txt файлы с lib.ru (открытый доступ, без авторских прав)
    books = [
        "http://az.lib.ru/t/tolstoj_lew_nikolaewich/text_0040.shtml",
        "http://az.lib.ru/d/dostoewskij_f_m/text_0060.shtml",
        "http://az.lib.ru/c/chehow_a_p/text_0060.shtml",
        "http://az.lib.ru/g/gogolx_n_w/text_0010.shtml",
        "http://az.lib.ru/t/turgenew_i_s/text_0010.shtml",
        "http://az.lib.ru/p/pushkin_a_s/text_0040.shtml",
        "http://az.lib.ru/b/bulgakow_m_a/text_0040.shtml",
        "http://az.lib.ru/l/leskov_n_s/text_0020.shtml",
        "http://az.lib.ru/k/kuprin_a_i/text_0020.shtml",
        "http://az.lib.ru/b/bunin_i_a/text_0010.shtml",
    ]
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 LexisResearch/1.0"
    for url in books:
        try:
            r = session.get(url, timeout=20)
            r.encoding = "cp1251"
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(r.text, "lxml")
            # Убрать навигацию и скрипты
            for tag in soup(["script", "style", "nav", "header", "footer"]):
                tag.decompose()
            txt = soup.get_text(separator="\n")
            if txt and len(txt) > 1000:
                yield txt
            time.sleep(1)
        except Exception as e:
            print(f"  [!] lib.ru {url}: {e}")


# ══════════════════════════════════════════════════════════════════
#  Источник 3: Новости
# ══════════════════════════════════════════════════════════════════
def source_news() -> Iterator[str]:
    """Новостные тексты из нескольких HF датасетов."""
    from datasets import load_dataset

    # 3a. IlyaGusev/gazeta — датасет газета.ру (саммари + статьи)
    try:
        print("  → Загрузка Gazeta.ru (новости)...")
        for split in ("train", "validation", "test"):
            try:
                ds = load_dataset(
                    "IlyaGusev/gazeta",
                    split=split,
                    trust_remote_code=True,
                    cache_dir=str(CACHE_DIR),
                )
                for row in ds:
                    # Полный текст + резюме вместе
                    parts = []
                    for field in ("text", "article", "summary"):
                        v = row.get(field, "")
                        if v:
                            parts.append(v)
                    if parts:
                        yield "\n\n".join(parts)
            except Exception:
                pass
    except Exception as e:
        print(f"  [!] gazeta: {e}")

    # 3b. Den Rossiiskoi Gazety + RIA через OPUS/HF
    try:
        print("  → Загрузка russian_news (HF)...")
        ds = load_dataset(
            "ilyagusev/ru_turbo_alpaca",
            split="train",
            trust_remote_code=True,
            cache_dir=str(CACHE_DIR),
        )
        for row in ds:
            for field in ("output", "instruction", "input"):
                txt = row.get(field, "")
                if txt and len(txt) > 100:
                    yield txt
    except Exception as e:
        print(f"  [!] ru_turbo_alpaca: {e}")

    # 3c. Lenta.ru датасет
    try:
        print("  → Загрузка Lenta.ru...")
        ds = load_dataset(
            "IlyaGusev/lenta_ru",
            split="train",
            trust_remote_code=True,
            cache_dir=str(CACHE_DIR),
        )
        for row in ds:
            parts = []
            for f in ("text", "title", "tags"):
                v = row.get(f, "")
                if isinstance(v, list):
                    v = ", ".join(v)
                if v:
                    parts.append(v)
            if parts:
                yield " ".join(parts)
    except Exception as e:
        print(f"  [!] lenta_ru: {e}")


# ══════════════════════════════════════════════════════════════════
#  Источник 4: Научные и образовательные тексты
# ══════════════════════════════════════════════════════════════════
def source_science() -> Iterator[str]:
    """Научные статьи, учебники, Викитека."""
    from datasets import load_dataset

    # 4a. Викитека (wikisource) через API
    yield from source_wikisource()

    # 4b. Russian SuperGLUE — академические тексты
    try:
        print("  → Загрузка ruSciBench (научные)...")
        ds = load_dataset(
            "mlsa-iai-msu-lab/ru-scibench",
            split="test",
            trust_remote_code=True,
            cache_dir=str(CACHE_DIR),
        )
        for row in ds:
            for f in ("abstract", "text", "content"):
                txt = row.get(f, "")
                if txt and len(txt) > 200:
                    yield txt
                    break
    except Exception as e:
        print(f"  [!] ruSciBench: {e}")

    # 4c. Wikipedia (более детальные статьи в категориях Наука)
    try:
        print("  → Wikipedia науч. статьи (API)...")
        yield from source_wikipedia_category("Наука")
        yield from source_wikipedia_category("Физика")
        yield from source_wikipedia_category("Биология")
        yield from source_wikipedia_category("Математика")
        yield from source_wikipedia_category("История")
    except Exception as e:
        print(f"  [!] Wikipedia categories: {e}")

    # 4d. Oscar / C4 как запасной вариант
    try:
        print("  → Загрузка OSCAR (ru)...")
        ds = load_dataset(
            "oscar-corpus/OSCAR-2301",
            "ru",
            split="train",
            streaming=True,
            trust_remote_code=True,
            cache_dir=str(CACHE_DIR),
        )
        for i, row in enumerate(ds):
            txt = row.get("text", "")
            if txt and len(txt) > 500:
                yield txt
            if i > 200_000:
                break
    except Exception as e:
        print(f"  [!] OSCAR: {e}")


def source_wikisource() -> Iterator[str]:
    """Викитека — классика и научные тексты (Mediawiki API)."""
    print("  → Загрузка Викитека (wikisource.ru)...")
    api = "https://ru.wikisource.org/w/api.php"
    session = requests.Session()
    session.headers["User-Agent"] = "LexisDataset/1.0"
    cats = [
        "Категория:Научные_статьи",
        "Категория:Философия",
        "Категория:Публицистика",
        "Категория:Естественные_науки",
    ]
    for cat in cats:
        try:
            r = session.get(
                api,
                params={
                    "action": "query",
                    "format": "json",
                    "list": "categorymembers",
                    "cmtitle": cat,
                    "cmlimit": 50,
                    "cmtype": "page",
                },
                timeout=15,
            )
            pages = r.json().get("query", {}).get("categorymembers", [])
            for page in pages[:30]:
                try:
                    r2 = session.get(
                        api,
                        params={
                            "action": "query",
                            "format": "json",
                            "pageids": page["pageid"],
                            "prop": "extracts",
                            "explaintext": True,
                        },
                        timeout=20,
                    )
                    result = r2.json().get("query", {}).get("pages", {})
                    for p in result.values():
                        txt = p.get("extract", "")
                        if txt and len(txt) > 300:
                            yield txt
                    time.sleep(0.3)
                except Exception:
                    pass
        except Exception as e:
            print(f"  [!] wikisource {cat}: {e}")


def source_wikipedia_category(category: str) -> Iterator[str]:
    """Вики-статьи из конкретной категории."""
    api = "https://ru.wikipedia.org/w/api.php"
    session = requests.Session()
    session.headers["User-Agent"] = "LexisDataset/1.0"
    try:
        r = session.get(
            api,
            params={
                "action": "query",
                "format": "json",
                "list": "categorymembers",
                "cmtitle": f"Категория:{category}",
                "cmlimit": 100,
                "cmtype": "page",
            },
            timeout=15,
        )
        pages = r.json().get("query", {}).get("categorymembers", [])
        ids = [str(p["pageid"]) for p in pages]
        for chunk in [ids[i : i + 20] for i in range(0, len(ids), 20)]:
            r2 = session.get(
                api,
                params={
                    "action": "query",
                    "format": "json",
                    "pageids": "|".join(chunk),
                    "prop": "extracts",
                    "explaintext": True,
                },
                timeout=30,
            )
            for page in r2.json().get("query", {}).get("pages", {}).values():
                txt = page.get("extract", "")
                if txt and len(txt) > 300:
                    yield txt
            time.sleep(0.5)
    except Exception as e:
        print(f"  [!] Wikipedia category {category}: {e}")


# ══════════════════════════════════════════════════════════════════
#  Источник 5: Диалоги и субтитры
# ══════════════════════════════════════════════════════════════════
def source_dialog() -> Iterator[str]:
    """Диалоги: субтитры, форумы, интервью."""
    from datasets import load_dataset

    # 5a. OpenSubtitles через HF
    try:
        print("  → Загрузка OpenSubtitles (диалоги)...")
        ds = load_dataset(
            "open_subtitles",
            lang1="en",
            lang2="ru",
            split="train",
            streaming=True,
            trust_remote_code=True,
            cache_dir=str(CACHE_DIR),
        )
        buf = []
        for row in ds:
            txt = row.get("translation", {}).get("ru", "")
            if txt and len(txt) > 15:
                buf.append(txt.strip())
                if len(buf) >= 30:
                    yield " ".join(buf)
                    buf = []
        if buf:
            yield " ".join(buf)
    except Exception as e:
        print(f"  [!] OpenSubtitles: {e}")

    # 5b. Russian conversations / chats
    try:
        print("  → Загрузка ru_conversational (чаты)...")
        ds = load_dataset(
            "Den4ikAI/russian_instructions",
            split="train",
            trust_remote_code=True,
            cache_dir=str(CACHE_DIR),
        )
        for row in ds:
            parts = []
            for f in ("instruction", "input", "output", "response"):
                v = row.get(f, "")
                if v and len(v) > 20:
                    parts.append(v)
            if parts:
                yield "\n".join(parts)
    except Exception as e:
        print(f"  [!] russian_instructions: {e}")

    # 5c. Толстой / Достоевский с диалогами (уже частично из proza)
    # 5d. Форумы через Common Crawl подмножество
    try:
        print("  → Загрузка CulturaX (ru)...")
        ds = load_dataset(
            "uonlp/CulturaX",
            "ru",
            split="train",
            streaming=True,
            trust_remote_code=True,
            cache_dir=str(CACHE_DIR),
        )
        for i, row in enumerate(ds):
            txt = row.get("text", "")
            if txt and len(txt) > 200:
                yield txt
            if i > 300_000:
                break
    except Exception as e:
        print(f"  [!] CulturaX: {e}")


# ══════════════════════════════════════════════════════════════════
#  Главный сборщик
# ══════════════════════════════════════════════════════════════════
DATASETS = [
    ("wiki", "Википедия", source_wikipedia),
    ("proza", "Художественная проза", source_proza),
    ("news", "Новости", source_news),
    ("science", "Наука и образование", source_science),
    ("dialog", "Диалоги", source_dialog),
]


def collect(name: str, label: str, source_fn, target_bytes: int = TARGET_BYTES):
    path = OUT_DIR / f"{name}.txt"
    print(f"\n{'═'*56}")
    print(f"  Сбор: {label}  →  {path.name}")
    print(f"{'═'*56}")

    writer = DatasetWriter(path, target_bytes, label)
    try:
        for text in source_fn():
            if writer.done:
                break
            writer.write(text)
    except KeyboardInterrupt:
        print("\n  [!] Прервано пользователем — сохраняю что есть...")
    finally:
        writer.close()
    return writer.written


def main():
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║     LEXIS Dataset Collector — Russian NLP            ║")
    print(
        f"║     Цель: {TARGET_GB} ГБ × {len(DATASETS)} файлов = {TARGET_GB*len(DATASETS):.1f} ГБ суммарно  ║"
    )
    print("╚══════════════════════════════════════════════════════╝")
    print()
    print("  Зависимости: datasets, requests, tqdm, beautifulsoup4")
    print("  Кэш:  ", CACHE_DIR.resolve())
    print("  Вывод:", OUT_DIR.resolve())
    print()

    totals = {}
    for name, label, source_fn in DATASETS:
        try:
            written = collect(name, label, source_fn)
            totals[name] = written
        except Exception as e:
            print(f"\n  [!!] Критическая ошибка в '{name}': {e}")
            totals[name] = 0

    # Итог
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║  Готово! Итоги:                                      ║")
    for name, label, _ in DATASETS:
        mb = totals.get(name, 0) / 1024**2
        bar = "█" * int(mb / (TARGET_GB * 1024 / 20))
        print(f"║  {name:<10} {bar:<20} {mb:6.1f} МБ    ║")
    total_gb = sum(totals.values()) / 1024**3
    print(f"║  {'ВСЕГО':<10} {'':20} {total_gb:6.2f} ГБ    ║")
    print("╠══════════════════════════════════════════════════════╣")
    print("║  Следующий шаг:                                      ║")
    print("║  → Скопируй нужные .txt в storage/datasets/          ║")
    print("║  → Запусти обучение в интерфейсе Lexis               ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()


if __name__ == "__main__":
    main()
