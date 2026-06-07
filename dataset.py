#!/usr/bin/env python3
"""
Максимально быстрый сборщик русских книг в один файл text.txt
────────────────────────────────────────────────────────────────
Использует асинхронные запросы (aiohttp) для скачивания тысяч книг
с открытых библиотек: Lib.ru, Flibusta, Project Gutenberg (ru),
Русская фантастика и т.д.

Запуск: python fast_books.py

Зависимости: aiohttp, aiofiles, beautifulsoup4, tqdm
Устанавливаются автоматически.
"""

import asyncio
import time
import re
import hashlib
import unicodedata
from pathlib import Path
import sys
import subprocess
import importlib


# ===== Автоустановка зависимостей =====
def auto_install():
    required = []
    try:
        import aiohttp
    except ImportError:
        required.append("aiohttp")
    try:
        import aiofiles
    except ImportError:
        required.append("aiofiles")
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        required.append("beautifulsoup4")
    try:
        import lxml
    except ImportError:
        required.append("lxml")
    try:
        from tqdm.asyncio import tqdm
    except ImportError:
        required.append("tqdm")
    if required:
        print(f"[install] {', '.join(required)} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *required, "-q"])
        print("✓ Зависимости установлены.")


def load_runtime_deps():
    global aiohttp, aiofiles, tqdm, BeautifulSoup
    import aiohttp as _aiohttp
    import aiofiles as _aiofiles
    from tqdm.asyncio import tqdm as _tqdm
    from bs4 import BeautifulSoup as _BeautifulSoup

    aiohttp = _aiohttp
    aiofiles = _aiofiles
    tqdm = _tqdm
    BeautifulSoup = _BeautifulSoup


try:
    load_runtime_deps()
except ImportError:
    aiohttp = None
    aiofiles = None
    tqdm = None
    BeautifulSoup = None

# ===== Конфигурация =====
OUT_FILE = Path("text.txt")  # единый выходной файл
MAX_CONCURRENT = 50  # количество одновременных загрузок
MAX_RETRIES = 3  # повторы при ошибке
TIMEOUT = aiohttp.ClientTimeout(total=60, connect=15)
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0"
)

# Источники: библиотеки, откуда будем брать прямые ссылки на .txt, .fb2, .html
# (приоритет отдаём .txt, затем .fb2, затем .html)
SOURCES = {
    # Lib.ru – зеркало без рекламы, быстрые ссылки на txt
    "libru_txt": [
        "http://lib.ru/LITRA/",
        "http://lib.ru/RUSSLIT/",
        "http://lib.ru/DETEKTIWY/",
        "http://lib.ru/FANTAST/",
        "http://lib.ru/POEZIQ/",
        "http://lib.ru/RUSSLIT/PUSHKIN/",
        "http://lib.ru/RUSSLIT/TOLSTOY/",
        "http://lib.ru/RUSSLIT/DOSTOEVSKIJ/",
        "http://lib.ru/RUSSLIT/GOGOL/",
        "http://lib.ru/RUSSLIT/TURGENEV/",
        "http://lib.ru/RUSSLIT/CHEKHOV/",
        "http://lib.ru/RUSSLIT/BULGAKOV/",
        "http://lib.ru/RUSSLIT/NABOKOV/",
        "http://lib.ru/RUSSLIT/LERMONTOV/",
        "http://lib.ru/RUSSLIT/SOLOGUB/",
        "http://lib.ru/RUSSLIT/BUNIN/",
        "http://lib.ru/RUSSLIT/PLATONOV/",
    ],
    # Flibusta – зеркало с fb2 (быстрое)
    "flibusta_fb2": [
        "http://flibusta.is/b/",  # книга по ID, будем перебирать ID
    ],
    # Gutenberg Russia (переводы)
    "gutenberg_txt": [
        "https://www.gutenberg.org/ebooks/",  # + ID, текст в /files/{id}/{id}-0.txt
    ],
}

# ===== Утилиты очистки текста =====
_GARBAGE = re.compile(
    r"(?:"
    r"<[^>]+>"  # HTML теги
    r"|\{.*?\}"  # шаблоны
    r"|\[[^\]]+\]"  # сноски [1]
    r"|https?://\S+"  # ссылки
    r"|www\.\S+"
    r"|\*\*.*?\*\*"  # жирный текст markdown
    r"|={2,}[^=]*={2,}"  # заголовки wiki
    r")",
    re.DOTALL | re.MULTILINE,
)
_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_MULTI_NL = re.compile(r"\n{3,}")
_RU_CHECK = re.compile(r"[а-яёА-ЯЁ]")
_MARKDOWN_BOLD = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_MARKDOWN_ITALIC = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", re.DOTALL)
_UNDERSCORES = re.compile(r"_+")


def clean_text(raw: str) -> str:
    """Очищает HTML/мусор, нормализует пробелы, оставляет только читабельные абзацы."""
    if not raw:
        return ""
    # Убираем управляющие символы
    raw = unicodedata.normalize("NFC", raw)
    raw = _MARKDOWN_BOLD.sub(r"\1", raw)
    raw = _MARKDOWN_ITALIC.sub(r"\1", raw)
    raw = _UNDERSCORES.sub(" ", raw)
    # Удаляем мусор
    raw = _GARBAGE.sub(" ", raw)
    raw = _MULTI_SPACE.sub(" ", raw)
    raw = _MULTI_NL.sub("\n\n", raw)
    # Разбиваем на строки, фильтруем
    lines = []
    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Проверка на русский язык (>40% русских букв)
        ru_chars = len(_RU_CHECK.findall(line))
        if ru_chars == 0 or len(line) < 30:
            continue
        ru_ratio = ru_chars / len(line.replace(" ", ""))
        if ru_ratio < 0.4:
            continue
        lines.append(line)
    return "\n\n".join(lines)


# ===== Дедупликация по хэшу =====
class Dedup:
    def __init__(self):
        self.seen = set()

    def add(self, text):
        h = hashlib.md5(text[:500].encode("utf-8", errors="ignore")).hexdigest()
        if h in self.seen:
            return False
        self.seen.add(h)
        return True


# ===== Асинхронный загрузчик =====
async def fetch(session, url, semaphore, retries=MAX_RETRIES):
    async with semaphore:
        for attempt in range(retries):
            try:
                async with session.get(url, timeout=TIMEOUT) as resp:
                    if resp.status == 200:
                        return await resp.text(errors="ignore")
                    elif resp.status == 404:
                        return None
            except Exception as e:
                if attempt == retries - 1:
                    pass  # лог молча, чтобы не спамить
                else:
                    await asyncio.sleep(0.5 * (attempt + 1))
        return None


async def fetch_binary(session, url, semaphore):
    """Для скачивания бинарных FB2 (потом распарсим)"""
    async with semaphore:
        try:
            async with session.get(url, timeout=TIMEOUT) as resp:
                if resp.status == 200:
                    return await resp.read()
        except:
            pass
    return None


# ===== Парсер ссылок на книги =====
async def get_links_libru(session, base_url, semaphore):
    """Из каталога Lib.ru собирает все ссылки на .txt и .fb2 файлы"""
    links = []
    html = await fetch(session, base_url, semaphore)
    if not html:
        return links
    soup = BeautifulSoup(html, "lxml")
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.endswith(".txt") or href.endswith(".fb2") or href.endswith(".html"):
            if not href.startswith("http"):
                href = base_url.rstrip("/") + "/" + href.lstrip("/")
            links.append(href)
    return links


async def get_links_flibusta(session, start_id=1, end_id=100000, semaphore=None):
    """Flibusta: перебираем ID книг (огромное количество)"""
    links = []
    for book_id in range(start_id, end_id + 1):
        # fb2 прямой доступ: http://flibusta.is/b/{id}/fb2
        url = f"http://flibusta.is/b/{book_id}/fb2"
        links.append(url)
        if book_id % 5000 == 0:
            # небольшая пауза, чтобы не банили
            await asyncio.sleep(0.2)
    return links


# ===== Извлечение текста из FB2 =====
def extract_fb2_text(fb2_bytes):
    """Парсит fb2, возвращает очищенный текст."""
    try:
        import xml.etree.ElementTree as ET

        root = ET.fromstring(fb2_bytes)
        ns = {"fb": "http://www.gribuser.ru/xml/fictionbook/2.0"}
        paragraphs = []
        for p in root.findall(".//fb:p", ns):
            if p.text:
                paragraphs.append(p.text.strip())
        return "\n\n".join(paragraphs)
    except:
        return None


# ===== Извлечение текста из HTML (Lib.ru) =====
def extract_html_text(html):
    soup = BeautifulSoup(html, "lxml")
    # Удаляем скрипты и стили
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()
    # Ищем основной контент – часто в <pre> или <div class="text">
    main = soup.find("pre")
    if not main:
        main = soup.find("div", class_="text")
    if not main:
        main = soup.find("body")
    if not main:
        return None
    text = main.get_text(separator="\n")
    return text


# ===== Основной сборщик =====
async def collect_books():
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    connector = aiohttp.TCPConnector(
        limit=MAX_CONCURRENT, limit_per_host=20, ttl_dns_cache=300
    )
    async with aiohttp.ClientSession(
        headers={"User-Agent": USER_AGENT}, connector=connector
    ) as session:

        # 1. Собираем все ссылки на книги из Lib.ru
        print("[1/3] Сбор ссылок на книги из Lib.ru...")
        libru_links = []
        for base in SOURCES["libru_txt"]:
            links = await get_links_libru(session, base, sem)
            libru_links.extend(links)
            print(f"    {base} → {len(links)} ссылок")
            await asyncio.sleep(0.1)
        print(f"  Всего ссылок из Lib.ru: {len(libru_links)}")

        # 2. Flibusta – генерируем ID (первые 200 000 книг)
        print("[2/3] Генерация ссылок Flibusta (ID 1..200000)...")
        flibusta_links = await get_links_flibusta(session, 1, 200000, sem)
        print(f"  Сгенерировано {len(flibusta_links)} ссылок на FB2")

        # Объединяем все ссылки
        all_links = libru_links + flibusta_links
        total_links = len(all_links)
        print(
            f"[3/3] Начинаем загрузку {total_links} книг (макс. {MAX_CONCURRENT} одновременно)..."
        )

        # Прогресс-бар и запись в файл
        pbar = tqdm(total=total_links, desc="Загрузка", unit="книга", colour="cyan")
        dedup = Dedup()
        total_bytes = 0
        start_time = time.time()

        async def process_link(url):
            nonlocal total_bytes
            # Обработка в зависимости от расширения
            ext = url.split(".")[-1].lower()
            if ext == "txt":
                html = await fetch(session, url, sem)
                if html:
                    text = clean_text(html)
                    if text and dedup.add(text):
                        async with aiofiles.open(OUT_FILE, "a", encoding="utf-8") as f:
                            await f.write(text + "\n\n---\n\n")
                        total_bytes += len(text.encode("utf-8"))
            elif ext == "fb2":
                data = await fetch_binary(session, url, sem)
                if data:
                    text = extract_fb2_text(data)
                    if text:
                        clean = clean_text(text)
                        if clean and dedup.add(clean):
                            async with aiofiles.open(
                                OUT_FILE, "a", encoding="utf-8"
                            ) as f:
                                await f.write(clean + "\n\n---\n\n")
                            total_bytes += len(clean.encode("utf-8"))
            elif ext == "html" or ext == "htm":
                html = await fetch(session, url, sem)
                if html:
                    raw = extract_html_text(html)
                    if raw:
                        clean = clean_text(raw)
                        if clean and dedup.add(clean):
                            async with aiofiles.open(
                                OUT_FILE, "a", encoding="utf-8"
                            ) as f:
                                await f.write(clean + "\n\n---\n\n")
                            total_bytes += len(clean.encode("utf-8"))
            pbar.update(1)
            # Обновление скорости каждые 5 секунд
            if pbar.n % 50 == 0:
                elapsed = time.time() - start_time
                speed = total_bytes / elapsed / 1024 if elapsed > 0 else 0
                pbar.set_postfix(
                    speed=f"{speed:.1f} KB/s", size=f"{total_bytes/1024**2:.1f} MB"
                )

        # Асинхронный запуск всех задач
        tasks = [asyncio.create_task(process_link(url)) for url in all_links]
        await asyncio.gather(*tasks)
        pbar.close()

        # Итог
        elapsed = time.time() - start_time
        mb_per_sec = (total_bytes / 1024 / 1024) / elapsed if elapsed > 0 else 0
        print("\n✓ Сбор завершён!")
        print(f"  Файл: {OUT_FILE.resolve()}")
        print(f"  Размер: {total_bytes / 1024**2:.1f} МБ")
        print(
            f"  Уникальных книг: {pbar.n - (total_links - len(tasks))} (приблизительно)"
        )
        print(f"  Средняя скорость: {mb_per_sec:.1f} МБ/с")
        print(f"  Время: {elapsed:.0f} сек")


def main():
    auto_install()
    load_runtime_deps()

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║   Быстрый сборщик книг → text.txt                     ║")
    print("║   Асинхронная загрузка, дедупликация, очистка       ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    # Очистка старого файла, если есть
    if OUT_FILE.exists():
        OUT_FILE.unlink()

    asyncio.run(collect_books())


if __name__ == "__main__":
    main()
