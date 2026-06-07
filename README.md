# ML-Lexis

ML-Lexis is a local text-generation lab for Russian prose. It starts with a bundled trained model, so you can open the chat and try generation right away. If you want your own style, vocabulary, or domain, you can upload texts, tokenize them, and train a new model version inside the same interface.

## What Is Inside

- Flask backend with chats, training API, generation queue, model/version history, and corrections.
- MiniGPT-style PyTorch model with modernized blocks: RMSNorm, SwiGLU, RoPE, KV cache for generation, plus legacy checkpoint loading helpers.
- Russian tokenizer and vocabulary builder without `<UNK>`.
- Postprocessing pipeline: cleanup, punctuation, repeated-word removal, sentence deduplication, simple grammar fixes, capitalization, and optional deeper generation modes.
- Web UI for chat, model selection, dataset upload, training dashboard, checkpoints, samples, and settings.
- Bundled trained model in `model27k/storage` for the default first run.

## Product Idea

The project should be presented as:

> A local Russian text-generation sandbox: it works out of the box on a bundled model, and then lets you train your own small language model on a personal corpus.

That framing is stronger than “train a model from scratch”, because the user gets value immediately:

1. Open Lexis.
2. Generate text with the ready model.
3. Add corrections or datasets.
4. Train a personal model version.
5. Compare epochs and keep the best checkpoint.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run.py
```

Then open:

```text
http://localhost:5000
```

Lexis opens directly in local mode.

## Storage Behavior

By default, Lexis uses the bundled trained model from:

```text
model27k/storage
```

If you want a separate clean workspace, set:

```text
LEXIS_STORAGE_DIR=storage
```

The app will then create and use `storage/` for your own database, checkpoints, datasets, and models.

## Training Your Own Model

1. Go to `Обучение`.
2. Set the path to a `.txt` corpus.
3. Click `Токенизировать`.
4. Review auto-selected parameters.
5. Create or select a model family.
6. Start training.
7. Use the chat page to compare best, validated, and selected epochs.

Small corpora overfit quickly, so the training dashboard focuses on train/validation loss, perplexity, and early stopping signals.

## Dataset Parser

`dataset.py` can collect and clean Russian books into `text.txt`. The parser removes HTML, links, markdown emphasis, repeated whitespace, and underscore artifacts before saving text. This keeps tokens like `some_word` from leaking into the vocabulary as underscore-shaped noise.

## Notes For Cleanup

- Large model files are currently stored in the repository. For a public release, move checkpoints to Git LFS, Releases, or an external artifact store.
- `.env`, runtime logs, generated datasets, and new local training output should stay out of git.
- The inspector page already has UI for selective postprocessing and deeper modes; the backend route should be aligned with those extra inspector parameters next.
