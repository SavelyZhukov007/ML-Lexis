"""backend/server.py — Lexis v5"""

import json, time, sys, os, re, threading, logging, secrets
from pathlib import Path
from datetime import timedelta
from functools import wraps

import psutil
from flask import Flask, Response, request, jsonify, send_from_directory, g

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from core.config import (
    load,
    save,
    active_params,
    PROFILES,
    LOG_DIR,
    CKPT_DIR,
    DATASETS_DIR,
    MODELS_DIR,
    model_dir,
    model_ckpt_dir,
    model_vocab_file,
    model_tokens_file,
    model_logs_dir,
    STORAGE_DIR,
)
from core.db import (
    init_db,
    register,
    login,
    logout,
    get_user_by_token,
    get_chats,
    create_chat,
    delete_chat,
    rename_chat,
    set_chat_model,
    get_messages,
    add_message,
    add_word_correction,
    get_corrections,
    get_all_corrections,
    add_user_dataset,
    get_user_datasets,
    get_all_dataset_paths,
    create_model,
    get_model_by_id,
    get_models,
    rename_model,
    create_model_version,
    get_model_versions,
    update_model_version,
    get_all_versions_flat,
    get_model_version_by_id,
    get_epochs_for_version,
    get_all_epochs_from_db,
    get_model_samples,
    add_model_sample,
    save_model_sample,
    delete_model_sample,
    queue_add,
    queue_status,
    queue_next_waiting,
    queue_set_processing,
    queue_set_done,
    queue_set_error,
    queue_leave,
    queue_cleanup,
    queue_count_waiting,
)
from core.tokenizer import (
    build_and_save,
    load_vocab,
    load_tokens,
    tokenize_text,
    extend_vocab_with_words,
    VOCAB_FILE,
)
from core.postprocess import (
    postprocess,
    deep_process,
    consciousness_generate,
    ConsciousnessConfig,
)
from core.state import (
    init as st_init,
    read as st_read,
    request_stop,
    set_error,
    get_resume_point,
    get_all_epochs_from_db as st_epochs,
)

FRONTEND = ROOT_DIR / "frontend"
app = Flask(__name__, static_folder=str(FRONTEND))
app.secret_key = os.environ.get("LEXIS_SECRET", secrets.token_hex(32))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-10s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_DIR / "server.log"), encoding="utf-8"),
    ],
)
logging.getLogger("werkzeug").setLevel(logging.ERROR)
log = logging.getLogger("server")

_trainer_thread = None
_tokenize_thread = None
_tokenize_progress = {
    "step": "",
    "pct": 0,
    "done": False,
    "error": None,
    "auto_params": None,
}
_trainer_lock = threading.Lock()
_current_version_id = None

# ── Online users tracking ────────────────────────────────────────────────────
_online_lock = threading.Lock()
_online_users: dict = {}  # {user_id: last_heartbeat_ts}
ONLINE_TTL = 30  # секунд без пинга — считаем оффлайн


def _touch_online(user_id: int):
    with _online_lock:
        _online_users[user_id] = time.time()


def _online_count() -> int:
    now = time.time()
    with _online_lock:
        return sum(1 for ts in _online_users.values() if now - ts < ONLINE_TTL)


def _online_list() -> list:
    now = time.time()
    with _online_lock:
        return [uid for uid, ts in _online_users.items() if now - ts < ONLINE_TTL]


# ── Limits ───────────────────────────────────────────────────────────────────
USER_WORD_LIMIT = 200
ADMIN_WORD_LIMIT = 2000


def _word_limit(role: str) -> int:
    return ADMIN_WORD_LIMIT if role == "admin" else USER_WORD_LIMIT


# ── Queue worker ─────────────────────────────────────────────────────────────
_queue_worker_running = False


def _start_queue_worker():
    global _queue_worker_running
    if _queue_worker_running:
        return
    _queue_worker_running = True
    t = threading.Thread(target=_queue_loop, daemon=True, name="queue-worker")
    t.start()


def _queue_loop():
    while True:
        try:
            task = queue_next_waiting()
            if task:
                queue_set_processing(task["request_id"])
                try:
                    params = json.loads(task["params"]) if task["params"] else {}
                    result = _do_generate(params, task["prompt"])
                    queue_set_done(
                        task["request_id"], json.dumps(result, ensure_ascii=False)
                    )
                except Exception as e:
                    log.error(f"Queue task error: {e}", exc_info=True)
                    queue_set_error(task["request_id"], str(e))
            queue_cleanup(max_age_sec=300)
        except Exception as e:
            log.error(f"Queue loop error: {e}")
        time.sleep(0.5)


init_db()
MAX_FILE_MB = 50
_start_queue_worker()


# ── Auth helpers ─────────────────────────────────────────────────────────────


def get_current_user():
    token = request.cookies.get("token") or request.headers.get("X-Token")
    user = get_user_by_token(token)
    if user:
        _touch_online(user["id"])
    return user


def auth_required(f):
    @wraps(f)
    def w(*a, **kw):
        user = get_current_user()
        if not user:
            return jsonify({"error": "Не авторизован", "auth": False}), 401
        g.user = user
        return f(*a, **kw)

    return w


def admin_required(f):
    @wraps(f)
    def w(*a, **kw):
        user = get_current_user()
        if not user:
            return jsonify({"error": "Не авторизован"}), 401
        if user.get("role") != "admin":
            return jsonify({"error": "Только для администратора"}), 403
        g.user = user
        return f(*a, **kw)

    return w


def _get_ip():
    return (
        request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
        or request.headers.get("X-Real-IP", "")
        or request.remote_addr
        or ""
    )


# ── Pages ────────────────────────────────────────────────────────────────────


@app.route("/")
def index():
    return (FRONTEND / "pages" / "index.html").read_text("utf-8")


@app.route("/<page>.html")
def page(page):
    p = FRONTEND / "pages" / f"{page}.html"
    return (
        p.read_text("utf-8")
        if p.exists()
        else (FRONTEND / "pages" / "index.html").read_text("utf-8")
    )


@app.route("/css/<path:fn>")
def css(fn):
    return send_from_directory(FRONTEND / "css", fn)


@app.route("/js/<path:fn>")
def js_r(fn):
    return send_from_directory(FRONTEND / "js", fn)


# ── Auth API ─────────────────────────────────────────────────────────────────


@app.route("/api/auth/register", methods=["POST"])
def api_register():
    d = request.json or {}
    u = d.get("username", "").strip()
    p = d.get("password", "")
    if len(u) < 2:
        return jsonify({"error": "Имя минимум 2 символа"}), 400
    if len(p) < 4:
        return jsonify({"error": "Пароль минимум 4 символа"}), 400
    result = register(u, p, ip=_get_ip())
    if not result["ok"]:
        return jsonify(result), 400
    lr = login(u, p, ip=_get_ip())
    resp = jsonify({"ok": True, "username": u, "role": lr["role"]})
    resp.set_cookie(
        "token", lr["token"], max_age=86400 * 30, httponly=True, samesite="Lax"
    )
    return resp


@app.route("/api/auth/login", methods=["POST"])
def api_login():
    d = request.json or {}
    ip = _get_ip()
    result = login(d.get("username", ""), d.get("password", ""), ip=ip)
    if not result["ok"]:
        return jsonify(result), 401
    resp = jsonify({"ok": True, "username": result["username"], "role": result["role"]})
    resp.set_cookie(
        "token", result["token"], max_age=86400 * 30, httponly=True, samesite="Lax"
    )
    return resp


@app.route("/api/auth/logout", methods=["POST"])
def api_logout():
    token = request.cookies.get("token")
    if token:
        logout(token)
    user = get_current_user()
    if user:
        with _online_lock:
            _online_users.pop(user["id"], None)
    resp = jsonify({"ok": True})
    resp.delete_cookie("token")
    return resp


@app.route("/api/auth/me")
def api_me():
    user = get_current_user()
    if not user:
        return jsonify({"auth": False})
    return jsonify(
        {
            "auth": True,
            "username": user["username"],
            "user_id": user["id"],
            "role": user["role"],
            "word_limit": _word_limit(user["role"]),
        }
    )


@app.route("/api/auth/heartbeat", methods=["POST"])
def api_heartbeat():
    user = get_current_user()
    if user:
        _touch_online(user["id"])
        return jsonify({"ok": True, "online": _online_count()})
    return jsonify({"ok": False}), 401


# ── Online API ───────────────────────────────────────────────────────────────


@app.route("/api/online")
def api_online():
    return jsonify({"count": _online_count(), "users": _online_list()})


# ── Models API ───────────────────────────────────────────────────────────────


@app.route("/api/models")
def api_models_list():
    models = get_models()
    for m in models:
        m["versions"] = get_model_versions(m["id"])
    return jsonify(models)


@app.route("/api/models", methods=["POST"])
@admin_required
def api_model_create():
    d = request.json or {}
    name = d.get("name", "").strip()
    if not name:
        return jsonify({"error": "Укажи имя модели"}), 400
    mid = create_model(name)
    return jsonify({"ok": True, "id": mid, "name": name})


@app.route("/api/models/<int:mid>/rename", methods=["POST"])
@admin_required
def api_model_rename(mid):
    name = (request.json or {}).get("name", "").strip()
    if not name:
        return jsonify({"error": "Имя не может быть пустым"}), 400
    rename_model(mid, name)
    return jsonify({"ok": True})


@app.route("/api/models/versions")
def api_all_versions():
    return jsonify(get_all_versions_flat())


@app.route("/api/models/versions/<int:vid>/epochs")
def api_version_epochs(vid):
    return jsonify(get_epochs_for_version(vid))


@app.route("/api/models/versions/<int:vid>/samples")
def api_version_samples(vid):
    epoch = request.args.get("epoch")
    try:
        epoch_num = int(epoch) if epoch is not None else None
    except ValueError:
        epoch_num = None
    return jsonify(get_model_samples(vid, epoch=epoch_num))


@app.route("/api/model-samples/<int:sid>/save", methods=["POST"])
@auth_required
def api_save_sample(sid):
    name = (request.json or {}).get("name", "").strip()
    if not name:
        return jsonify({"error": "Укажи имя для сохранения"}), 400
    save_model_sample(sid, name)
    return jsonify({"ok": True})


@app.route("/api/model-samples/<int:sid>", methods=["DELETE"])
@auth_required
def api_delete_sample(sid):
    delete_model_sample(sid)
    return jsonify({"ok": True})


@app.route("/api/models/versions/<int:vid>/load", methods=["POST"])
@admin_required
def api_load_version(vid):
    import torch, shutil
    from core.db import get_conn

    conn = get_conn()
    row = conn.execute("SELECT * FROM model_versions WHERE id=?", (vid,)).fetchone()
    conn.close()
    if not row:
        return jsonify({"error": "Версия не найдена"}), 404
    ckpt_path = row["checkpoint_path"]
    if not ckpt_path or not Path(ckpt_path).exists():
        return jsonify({"error": "Файл чекпоинта не найден"}), 404
    shutil.copy2(ckpt_path, str(CKPT_DIR / "best_model.pt"))
    return jsonify({"ok": True, "message": f"Версия {row['version_num']} загружена"})


# ── Chats ────────────────────────────────────────────────────────────────────


@app.route("/api/chats")
@auth_required
def api_chats():
    return jsonify(get_chats(g.user["id"]))


@app.route("/api/chats", methods=["POST"])
@auth_required
def api_chat_create():
    d = request.json or {}
    vid = d.get("model_version_id")
    cid = create_chat(g.user["id"], d.get("title", "Новый чат"), vid)
    return jsonify({"id": cid, "title": d.get("title", "Новый чат")})


@app.route("/api/chats/<int:cid>", methods=["DELETE"])
@auth_required
def api_chat_delete(cid):
    delete_chat(cid, g.user["id"])
    return jsonify({"ok": True})


@app.route("/api/chats/<int:cid>/rename", methods=["POST"])
@auth_required
def api_chat_rename(cid):
    title = (request.json or {}).get("title", "Новый чат")
    rename_chat(cid, g.user["id"], title)
    return jsonify({"ok": True})


@app.route("/api/chats/<int:cid>/set_model", methods=["POST"])
@auth_required
def api_chat_set_model(cid):
    vid = (request.json or {}).get("version_id")
    if vid:
        set_chat_model(cid, g.user["id"], vid)
    return jsonify({"ok": True})


@app.route("/api/chats/<int:cid>/messages")
@auth_required
def api_messages(cid):
    return jsonify(get_messages(cid))


# ── Model loading ─────────────────────────────────────────────────────────────


def _load_model(checkpoint_path: str = None):
    import torch
    from core.model import build as build_model

    path = Path(checkpoint_path) if checkpoint_path else CKPT_DIR / "best_model.pt"
    if not path.exists():
        raise FileNotFoundError("Модель не найдена. Сначала обучите.")
    ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
    params = ckpt["params"]
    ckpt_vs = ckpt["vocab_size"]
    model = build_model(ckpt_vs, params)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    vocab_dir = path.parent.parent if path.parent.name == "checkpoints" else None
    vocab = load_vocab(base_dir=vocab_dir)
    return model, vocab, ckpt_vs


# ── Core generation logic ─────────────────────────────────────────────────────


def _do_generate(params: dict, prompt_text: str) -> dict:
    """
    Центральная функция генерации. Используется и напрямую, и из очереди.
    Режимы: basic | dpa | extra_thinking
    После DPA и extra_thinking применяются все 9 агентов.
    Гарантирует сохранение количества слов близко к target_words.
    """
    seed = prompt_text or ""
    n_words = params.get("num_words", 60)
    temp = float(params.get("temperature", 1.0))
    top_k = int(params.get("top_k", 50))
    top_p = float(params.get("top_p", 0.92))
    mode = params.get("mode", "basic")
    rep_pen = float(params.get("repetition_penalty", 1.3))
    no_rep = int(params.get("no_repeat_ngram", 3))
    ckpt_path = params.get("checkpoint_path")
    version_id = params.get("version_id")

    # Extra thinking params
    et_hypotheses = int(params.get("et_hypotheses", 5))
    et_passes = int(params.get("et_passes", 2))

    if version_id is not None and ckpt_path is None:
        version = get_model_version_by_id(int(version_id))
        if version and version.get("checkpoint_path"):
            ckpt_path = version["checkpoint_path"]

    model, vocab, ckpt_vs = _load_model(ckpt_path)
    w2i = vocab["w2i"]
    i2w = vocab["i2w"]
    bos = w2i.get("<BOS>", 1)
    eos = w2i.get("<EOS>", 2)

    seed_tokens = tokenize_text(seed) if seed else []
    new_words = [t for t in seed_tokens if t not in w2i]
    base_dir = Path(ckpt_path).parent.parent if ckpt_path else None
    if new_words:
        extend_vocab_with_words(new_words, base_dir=base_dir)
        vocab = load_vocab(base_dir=base_dir)
        w2i = vocab["w2i"]
        i2w = vocab["i2w"]

    prompt = [w2i.get(t, bos) for t in seed_tokens] or [bos]
    corrections = get_all_corrections()

    def _gen_ids(p_tokens, count):
        # Генерируем с 30% запасом чтобы после постобработки осталось нужное кол-во
        safe_count = max(count, int(count * 1.35))
        return model.generate(
            p_tokens,
            max_new=safe_count,
            temperature=temp,
            top_k=top_k,
            top_p=top_p,
            eos_id=eos,
            vocab_size=ckpt_vs,
            repetition_penalty=rep_pen,
            no_repeat_ngram=no_rep,
            min_new_tokens=min(20, count),
        )

    def _ids_to_raw(ids, prefix_tokens):
        words = [i2w.get(i, "") for i in ids if i2w.get(i, "")]
        prefix_str = " ".join(prefix_tokens) + " " if prefix_tokens else ""
        return (prefix_str + " ".join(words)).strip()

    def _trim_to_words(text: str, target: int) -> str:
        """Обрезает текст до target слов, заканчивая на границе предложения."""
        wds = text.split()
        if len(wds) <= target:
            return text
        # Ищем конец предложения ближайший к target
        sub = " ".join(wds[: target + 10])
        parts = re.split(r"(?<=[.!?])\s+", sub)
        result = []
        count = 0
        for part in parts:
            pw = len(part.split())
            if count + pw > target and count >= int(target * 0.8):
                break
            result.append(part)
            count += pw
        if not result:
            result = [" ".join(wds[:target])]
        out = " ".join(result)
        if out and out[-1] not in ".!?":
            out += "."
        return out

    def _ensure_word_count(text: str, target: int, p_tokens, max_tries=4) -> str:
        """
        Дополняет текст если слов меньше target.
        Никогда не усекает ниже 80% от target.
        """
        for _ in range(max_tries):
            wc = len(text.split())
            if wc >= int(target * 0.85):
                break
            deficit = target - wc + int(target * 0.2)
            ctx_tokens = [w2i.get(t, bos) for t in tokenize_text(text)][
                -model.seq_len :
            ] or [bos]
            more_ids = model.generate(
                ctx_tokens,
                max_new=deficit,
                temperature=temp,
                top_k=top_k,
                top_p=top_p,
                eos_id=eos,
                vocab_size=ckpt_vs,
                repetition_penalty=rep_pen,
                no_repeat_ngram=no_rep,
                min_new_tokens=max(5, deficit // 2),
            )
            more_words = [i2w.get(i, "") for i in more_ids if i2w.get(i, "")]
            if not more_words:
                break
            text = (text.rstrip(".!?") + " " + " ".join(more_words)).strip()
            text = postprocess(text, corrections)
        return text

    # ── Режимы ───────────────────────────────────────────────────────────────

    extra_data = {}

    if mode == "extra_thinking":
        cfg = ConsciousnessConfig(
            n_hypotheses=et_hypotheses,
            refine_passes=et_passes,
        )
        result_data = consciousness_generate(
            model,
            vocab,
            prompt,
            max_new=int(n_words * 1.5),
            config=cfg,
            ckpt_vs=ckpt_vs,
            corrections=corrections,
        )
        raw_text = result_data["text"]
        # После Extra thinking — применяем все 9 агентов
        text = postprocess(raw_text, corrections)
        text = _ensure_word_count(text, n_words, prompt)
        text = _trim_to_words(text, int(n_words * 1.1))
        extra_data = {
            "hypotheses": result_data.get("hypotheses", []),
            "scores": result_data.get("scores", []),
            "best": result_data.get("best", ""),
            "consensus": result_data.get("consensus", ""),
            "stats": result_data.get("stats", {}),
        }
        raw = result_data.get("best", raw_text)

    elif mode == "dpa":
        gen_ids = _gen_ids(prompt, n_words)
        raw = _ids_to_raw(gen_ids, seed_tokens)
        # DPA
        if len(raw.split()) >= 20:
            dpa_text = deep_process(
                raw,
                model,
                vocab,
                temperature=temp,
                top_k=top_k,
                top_p=top_p,
                ckpt_vs=ckpt_vs,
                corrections=corrections,
            )
        else:
            dpa_text = postprocess(raw, corrections)
        # После DPA — все 9 агентов сверху
        text = postprocess(dpa_text, corrections)
        text = _ensure_word_count(text, n_words, prompt)
        text = _trim_to_words(text, int(n_words * 1.1))

    else:
        # basic — только агенты
        gen_ids = _gen_ids(prompt, n_words)
        raw = _ids_to_raw(gen_ids, seed_tokens)
        text = postprocess(raw, corrections)
        text = _ensure_word_count(text, n_words, prompt)
        text = _trim_to_words(text, int(n_words * 1.1))

    return {
        "ok": True,
        "text": text,
        "raw": raw,
        "word_count": len(text.split()),
        "target_words": n_words,
        **extra_data,
    }


# ── Generate API ──────────────────────────────────────────────────────────────


@app.route("/api/chats/<int:cid>/generate", methods=["POST"])
@auth_required
def api_generate(cid):
    d = request.json or {}
    role = g.user.get("role", "user")
    limit = _word_limit(role)
    n_words = max(3, min(limit, int(d.get("num_words", 60))))

    mode = d.get("mode", "basic")

    # Extra thinking доступен только админу
    if mode == "extra_thinking" and role != "admin":
        return jsonify({"error": "Extra Thinking доступен только администратору"}), 403

    params = {
        "num_words": n_words,
        "temperature": float(d.get("temperature", 1.0)),
        "top_k": int(d.get("top_k", 50)),
        "top_p": float(d.get("top_p", 0.92)),
        "mode": mode,
        "repetition_penalty": float(d.get("repetition_penalty", 1.3)),
        "no_repeat_ngram": int(d.get("no_repeat_ngram", 3)),
        "checkpoint_path": d.get("checkpoint_path"),
        "version_id": d.get("version_id"),
        "et_hypotheses": int(d.get("et_hypotheses", 5)),
        "et_passes": int(d.get("et_passes", 2)),
    }
    seed = d.get("seed", "").strip()

    # Сразу пробуем генерацию (очередь строится через polling)
    try:
        result = _do_generate(params, seed)
        if seed:
            add_message(cid, "user", seed)
        bot_mid = add_message(cid, "bot", result["text"])
        result["msg_id"] = bot_mid
        return jsonify(result)
    except Exception as e:
        log.error(f"Generate error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/generate/queue", methods=["POST"])
@auth_required
def api_queue_generate():
    """
    Поставить запрос в очередь. Возвращает request_id.
    Клиент поллит /api/generate/status/<request_id>.
    """
    d = request.json or {}
    role = g.user.get("role", "user")
    limit = _word_limit(role)
    n_words = max(3, min(limit, int(d.get("num_words", 60))))
    mode = d.get("mode", "basic")
    if mode == "extra_thinking" and role != "admin":
        return jsonify({"error": "Extra Thinking доступен только администратору"}), 403

    params = {
        "num_words": n_words,
        "temperature": float(d.get("temperature", 1.0)),
        "top_k": int(d.get("top_k", 50)),
        "top_p": float(d.get("top_p", 0.92)),
        "mode": mode,
        "repetition_penalty": float(d.get("repetition_penalty", 1.3)),
        "no_repeat_ngram": int(d.get("no_repeat_ngram", 3)),
        "checkpoint_path": d.get("checkpoint_path"),
        "version_id": d.get("version_id"),
        "et_hypotheses": int(d.get("et_hypotheses", 5)),
        "et_passes": int(d.get("et_passes", 2)),
    }
    rid = secrets.token_urlsafe(16)
    seed = d.get("seed", "").strip()
    pos = queue_add(g.user["id"], rid, seed, params)
    return jsonify(
        {
            "ok": True,
            "request_id": rid,
            "position": pos,
            "queue_size": queue_count_waiting(),
        }
    )


@app.route("/api/generate/status/<rid>")
@auth_required
def api_queue_status(rid):
    s = queue_status(rid)
    if not s.get("found"):
        return jsonify({"error": "Задача не найдена"}), 404
    if s["status"] == "done" and s.get("result"):
        try:
            s["result"] = json.loads(s["result"])
        except Exception:
            pass
    s["queue_total"] = queue_count_waiting()
    return jsonify(s)


@app.route("/api/generate/leave/<rid>", methods=["DELETE"])
@auth_required
def api_queue_leave(rid):
    queue_leave(rid, g.user["id"])
    return jsonify({"ok": True})


# ── Word alternatives (batched) ───────────────────────────────────────────────


@app.route("/api/word/alternatives", methods=["POST"])
@auth_required
def api_word_alternatives():
    import torch, torch.nn.functional as F

    d = request.json or {}
    context = d.get("context", "").strip()
    word = d.get("word", "").strip()
    n_alts = int(d.get("n", 6))
    try:
        model, vocab, ckpt_vs = _load_model()
        w2i = vocab["w2i"]
        i2w = vocab["i2w"]
        bos = w2i.get("<BOS>", 1)
        ctx_tokens = tokenize_text(context) if context else []
        prompt = [w2i.get(t, bos) for t in ctx_tokens[-40:]] or [bos]

        # Батч-генерация: один forward pass → topK токенов
        inp = torch.tensor([prompt], dtype=torch.long).clamp(0, ckpt_vs - 1)
        with torch.no_grad():
            logits = model(inp)[0, -1, :ckpt_vs].float()
        logits = logits / 1.3  # temperature
        topk_v, topk_i = torch.topk(logits, min(100, ckpt_vs))
        probs = F.softmax(topk_v, dim=-1)

        alts = []
        seen = {word.lower()}
        for prob_v, tid in zip(probs.tolist(), topk_i.tolist()):
            w = i2w.get(tid, "")
            if w and w.lower() not in seen and re.match(r"[а-яёa-z]", w, re.IGNORECASE):
                alts.append({"word": w, "prob": round(prob_v, 4)})
                seen.add(w.lower())
            if len(alts) >= n_alts:
                break

        return jsonify({"ok": True, "alternatives": alts, "word": word})
    except Exception as e:
        log.error(f"Alternatives: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ── Corrections ───────────────────────────────────────────────────────────────


@app.route("/api/corrections")
@auth_required
def api_corrections_get():
    return jsonify(get_corrections(g.user["id"]))


@app.route("/api/corrections", methods=["POST"])
@auth_required
def api_corrections_add():
    d = request.json or {}
    wrong = d.get("wrong", "").strip()
    correct = d.get("correct", "").strip()
    context = d.get("context", "")
    msg_id = d.get("msg_id")
    if not wrong or not correct:
        return jsonify({"error": "Нужно заполнить оба поля"}), 400
    add_word_correction(g.user["id"], wrong, correct, context, msg_id)
    if VOCAB_FILE.exists():
        extend_vocab_with_words(tokenize_text(correct))
    return jsonify({"ok": True})


# ── Datasets ──────────────────────────────────────────────────────────────────


@app.route("/api/datasets/upload", methods=["POST"])
@auth_required
def api_dataset_upload():
    if "file" not in request.files:
        return jsonify({"error": "Нет файла"}), 400
    f = request.files["file"]
    if not f.filename.endswith(".txt"):
        return jsonify({"error": "Только .txt файлы"}), 400
    content = f.read()
    if len(content) > MAX_FILE_MB * 1024 * 1024:
        return jsonify({"error": f"Файл > {MAX_FILE_MB} МБ"}), 400
    text = content.decode("utf-8", errors="ignore")
    words = len(text.split())
    filepath = str(DATASETS_DIR / f"u{g.user['id']}_{int(time.time())}_{f.filename}")
    Path(filepath).write_text(text, "utf-8")
    did = add_user_dataset(g.user["id"], f.filename, filepath, words)
    added = extend_vocab_with_words(tokenize_text(text)) if VOCAB_FILE.exists() else 0
    return jsonify(
        {
            "ok": True,
            "id": did,
            "filename": f.filename,
            "words": words,
            "new_vocab_words": added,
        }
    )


@app.route("/api/datasets")
@auth_required
def api_datasets_list():
    return jsonify(get_user_datasets(g.user["id"]))


# ── Upload .pt ────────────────────────────────────────────────────────────────


@app.route("/api/model/upload", methods=["POST"])
@admin_required
def api_model_upload():
    import torch, shutil

    if "file" not in request.files:
        return jsonify({"error": "Нет файла"}), 400
    f = request.files["file"]
    if not f.filename.endswith(".pt"):
        return jsonify({"error": "Только .pt файлы"}), 400
    content = f.read()
    tmp_path = CKPT_DIR / "upload_tmp.pt"
    tmp_path.write_bytes(content)
    try:
        ckpt = torch.load(str(tmp_path), map_location="cpu", weights_only=True)
        if "model" not in ckpt or "params" not in ckpt or "vocab_size" not in ckpt:
            tmp_path.unlink()
            return jsonify({"error": "Файл не содержит нужных ключей"}), 400
        shutil.copy2(str(tmp_path), str(CKPT_DIR / "best_model.pt"))
        tmp_path.unlink(missing_ok=True)
        return jsonify(
            {
                "ok": True,
                "epoch": ckpt.get("epoch", "?"),
                "vocab_size": ckpt["vocab_size"],
                "params": ckpt["params"],
            }
        )
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        return jsonify({"error": f"Ошибка чтения: {e}"}), 400


# ── Tokenize ──────────────────────────────────────────────────────────────────


@app.route("/api/tokenize/start", methods=["POST"])
@admin_required
def api_tokenize_start():
    global _tokenize_thread, _tokenize_progress
    if _tokenize_thread and _tokenize_thread.is_alive():
        return jsonify({"error": "Токенизация уже идёт"}), 400
    cfg = load()
    path = cfg.get("text_file", "")
    if not Path(path).exists():
        return jsonify({"error": f"Файл не найден: {path}"}), 400
    _tokenize_progress = {
        "step": "Инициализация...",
        "pct": 0,
        "done": False,
        "error": None,
        "auto_params": None,
    }

    request_data = request.json or {}
    model_id = request_data.get("model_id")
    model_base_dir = None
    if model_id is not None:
        model_info = get_model_by_id(int(model_id))
        if model_info:
            model_base_dir = model_dir(model_info["slug"])

    def _run():
        try:
            _tokenize_progress.update({"step": "Читаю датасеты...", "pct": 10})
            extra = get_all_dataset_paths()
            _tokenize_progress.update({"step": "Строю словарь...", "pct": 40})
            stats = build_and_save(path, extra, base_dir=model_base_dir)
            _tokenize_progress.update({"step": "Анализирую корпус...", "pct": 80})
            from core.auto_params import recommend_params, analyze_corpus

            main_text = Path(path).read_text(encoding="utf-8", errors="ignore")
            corpus_stats = analyze_corpus(main_text)
            rec = recommend_params(
                n_tokens=stats["n_tokens"],
                vocab_size=stats["vocab_size"],
                avg_sentence_len=corpus_stats["avg_sent_len"],
            )
            _tokenize_progress.update(
                {
                    "step": "Готово",
                    "pct": 100,
                    "done": True,
                    "stats": stats,
                    "corpus_stats": corpus_stats,
                    "auto_params": rec,
                    "warnings": rec.get("warnings", []),
                    "tpv": rec.get("tpv", 0),
                }
            )
        except Exception as e:
            _tokenize_progress["error"] = str(e)
            log.error(f"Tokenize: {e}", exc_info=True)

    _tokenize_thread = threading.Thread(target=_run, daemon=True, name="tokenizer")
    _tokenize_thread.start()
    return jsonify({"ok": True})


@app.route("/api/tokenize/status")
def api_tokenize_status():
    return jsonify(
        {
            **_tokenize_progress,
            "running": bool(_tokenize_thread and _tokenize_thread.is_alive()),
        }
    )


# ── Train ─────────────────────────────────────────────────────────────────────


def _trainer_body(params, vocab_size, n_tokens, n_batches, version_id, model_slug):
    st_init(params, vocab_size, n_tokens, n_batches)
    try:
        from core.trainer import run

        run(params, version_id=version_id, model_slug=model_slug)
    except Exception as e:
        set_error(str(e))
        log.error(f"Trainer error: {e}", exc_info=True)


@app.route("/api/train/start", methods=["POST"])
@admin_required
def api_train_start():
    global _trainer_thread, _current_version_id
    with _trainer_lock:
        if _trainer_thread and _trainer_thread.is_alive():
            return jsonify({"error": "Обучение уже запущено"}), 400
        import shutil

        d = request.json or {}
        model_id = d.get("model_id")
        model_name = d.get("model_name", "").strip()
        notes = d.get("notes", "")

        if not model_id:
            if not model_name:
                model_name = f"Модель {int(time.time())}"
            model_id = create_model(model_name)

        model_info = get_model_by_id(model_id)
        if not model_info:
            return jsonify({"error": "Модель не найдена"}), 404
        model_slug = model_info["slug"]
        mdir = model_dir(model_slug)
        model_logs_dir(model_slug)

        root_vocab = STORAGE_DIR / "vocab.json"
        root_tokens = STORAGE_DIR / "tokens.npy"
        model_vocab = model_vocab_file(model_slug)
        model_tokens = model_tokens_file(model_slug)
        if not model_vocab.exists() or not model_tokens.exists():
            if root_vocab.exists() and root_tokens.exists():
                shutil.copy2(str(root_vocab), str(model_vocab))
                shutil.copy2(str(root_tokens), str(model_tokens))
            else:
                return jsonify({"error": "Сначала токенизируйте текст"}), 400

        cfg = load()
        params = active_params(cfg)
        vocab = load_vocab(base_dir=mdir)
        tokens = load_tokens(base_dir=mdir)
        bs = params["batch_size"]
        sl = params["seq_len"]
        nb = max(1, (int(len(tokens) * 0.9) - sl) // bs)

        version_id = create_model_version(model_id, params, vocab["vocab_size"], notes)
        _current_version_id = version_id

        _trainer_thread = threading.Thread(
            target=_trainer_body,
            args=(params, vocab["vocab_size"], len(tokens), nb, version_id, model_slug),
            daemon=True,
            name="trainer",
        )
        _trainer_thread.start()
    return jsonify({"ok": True, "version_id": version_id, "model_id": model_id})


@app.route("/api/train/stop", methods=["POST"])
@admin_required
def api_train_stop():
    request_stop()
    return jsonify({"ok": True, "message": "Сигнал остановки отправлен"})


@app.route("/api/train/reset", methods=["POST"])
@admin_required
def api_train_reset():
    if _trainer_thread and _trainer_thread.is_alive():
        return jsonify({"error": "Сначала остановите обучение"}), 400
    for p in CKPT_DIR.glob("epoch_*.pt"):
        try:
            p.unlink()
        except Exception:
            pass
    from core.config import STATE_FILE

    try:
        STATE_FILE.unlink()
    except Exception:
        pass
    try:
        from core.db import get_conn

        conn = get_conn()
        conn.execute("DELETE FROM training_epochs")
        conn.commit()
        conn.close()
    except Exception:
        pass
    return jsonify({"ok": True})


@app.route("/api/train/state")
def api_train_state():
    state = st_read()
    mem = psutil.virtual_memory()
    state.update(
        {
            "ram_used": round(mem.used / 1024**3, 2),
            "ram_total": round(mem.total / 1024**3, 2),
            "ram_pct": round(mem.percent, 1),
            "cpu_pct": round(psutil.cpu_percent(interval=None), 1),
            "cpu_count": psutil.cpu_count(logical=True),
            "storage_gb": round(
                sum(f.stat().st_size for f in CKPT_DIR.rglob("*") if f.is_file())
                / 1024**3,
                2,
            ),
            "trainer_alive": bool(_trainer_thread and _trainer_thread.is_alive()),
            "current_version_id": _current_version_id,
            "online_count": _online_count(),
            "queue_size": queue_count_waiting(),
        }
    )
    if _current_version_id is not None:
        version = get_model_version_by_id(_current_version_id)
        if version:
            state["current_model_name"] = version.get("model_name")
            state["current_version_num"] = version.get("version_num")
            state["current_model_id"] = version.get("model_id")
    return jsonify(state)


@app.route("/api/train/epochs")
def api_train_epochs():
    return jsonify(get_all_epochs_from_db())


@app.route("/api/admin/logs")
@admin_required
def api_admin_logs():
    log_file = LOG_DIR / "server.log"
    if not log_file.exists():
        return jsonify({"lines": []})
    lines = log_file.read_text("utf-8", errors="ignore").splitlines()
    n = int(request.args.get("n", 200))
    return jsonify({"lines": lines[-n:]})


# ── Config ────────────────────────────────────────────────────────────────────


@app.route("/api/config")
def api_config_get():
    cfg = load()
    return jsonify({**cfg, "profiles": PROFILES})


@app.route("/api/config", methods=["POST"])
@admin_required
def api_config_set():
    cfg = load()
    data = request.json or {}
    cfg.update(data)
    save(cfg)
    return jsonify({"ok": True})


# ── SSE ───────────────────────────────────────────────────────────────────────


def _fmt(sec) -> str:
    if sec is None:
        return "—"
    td = timedelta(seconds=int(sec))
    h, r = divmod(td.seconds, 3600)
    m, s = divmod(r, 60)
    if td.days:
        return f"{td.days}д {h:02d}:{m:02d}:{s:02d}"
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _sse_gen():
    psutil.cpu_percent(interval=0.3)
    while True:
        try:
            state = st_read()
            mem = psutil.virtual_memory()
            nb = max(state.get("n_train_batches", 1), 1)
            cb = state.get("current_batch", 0)
            payload = {
                **state,
                "ram_used": round(mem.used / 1024**3, 2),
                "ram_total": round(mem.total / 1024**3, 2),
                "ram_pct": round(mem.percent, 1),
                "cpu_pct": round(psutil.cpu_percent(interval=None), 1),
                "cpu_count": psutil.cpu_count(logical=True),
                "storage_gb": round(
                    sum(f.stat().st_size for f in CKPT_DIR.rglob("*") if f.is_file())
                    / 1024**3,
                    2,
                ),
                "elapsed_fmt": _fmt(state.get("elapsed_sec")),
                "eta_fmt": _fmt(state.get("eta_sec")),
                "epoch_eta_fmt": _fmt(state.get("epoch_eta_sec")),
                "epoch_pct": round(cb / nb * 100, 1),
                "trainer_alive": bool(_trainer_thread and _trainer_thread.is_alive()),
                "current_version_id": _current_version_id,
                "online_count": _online_count(),
                "queue_size": queue_count_waiting(),
            }
            if _current_version_id is not None:
                version = get_model_version_by_id(_current_version_id)
                if version:
                    payload["current_model_name"] = version.get("model_name")
                    payload["current_version_num"] = version.get("version_num")
                    payload["current_model_id"] = version.get("model_id")
            yield f"data: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        time.sleep(1.5)


@app.route("/api/stream")
def stream():
    return Response(
        _sse_gen(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Epoch management ──────────────────────────────────────────────────────────

from core.db import (
    validate_epoch,
    delete_epoch_ckpt,
    get_epochs_for_chat,
    get_all_epochs_rich,
    get_epoch_ckpt_path,
    get_models_with_versions,
)


@app.route("/api/models/versions/<int:vid>/epochs/rich")
def api_epochs_rich(vid):
    return jsonify(get_all_epochs_rich(vid))


@app.route(
    "/api/models/versions/<int:vid>/epochs/<int:epoch>/validate", methods=["POST"]
)
@admin_required
def api_validate_epoch(vid, epoch):
    d = request.json or {}
    name = d.get("name", "").strip()
    if not name:
        return jsonify({"error": "Укажи имя для валидированной эпохи"}), 400
    result = validate_epoch(vid, epoch, name)
    return jsonify({"ok": True, "epoch": result})


@app.route(
    "/api/models/versions/<int:vid>/epochs/<int:epoch>/delete_ckpt", methods=["POST"]
)
@admin_required
def api_delete_epoch_ckpt(vid, epoch):
    result = delete_epoch_ckpt(vid, epoch)
    return jsonify(result)


@app.route("/api/models/versions/<int:vid>/epochs/<int:epoch>/samples")
def api_epoch_samples(vid, epoch):
    return jsonify(get_model_samples(vid, epoch=epoch))


@app.route("/api/models/families")
def api_model_families():
    return jsonify(get_models_with_versions())


@app.route("/api/models/versions/<int:vid>/epochs/<int:epoch>/ckpt_path")
def api_epoch_ckpt_path(vid, epoch):
    path = get_epoch_ckpt_path(vid, epoch)
    return jsonify({"path": path})


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    cfg = load()
    port = cfg.get("server_port", 5000)
    psutil.cpu_percent(interval=0.3)
    print(f"\n  ╔══════════════════════════════════════════╗")
    print(f"  ║         LEXIS v5 — Text Generation       ║")
    print(f"  ╠══════════════════════════════════════════╣")
    print(f"  ║  http://localhost:{port:<5}                  ║")
    print(f"  ║  Локальная сеть: http://<ваш_IP>:{port}  ║")
    print(f"  ║  Ctrl+C для выхода                       ║")
    print(f"  ╚══════════════════════════════════════════╝\n")
    try:
        import webbrowser

        webbrowser.open(f"http://localhost:{port}")
    except Exception:
        pass
    app.run(host="0.0.0.0", port=port, threaded=True, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
