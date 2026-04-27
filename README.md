# PIXEL vs BPE Notebook

This repository contains a self-contained `marimo` notebook that illustrates one core idea from
the paper *Language Modelling with Pixels*: token-based models have a vocabulary bottleneck,
while PIXEL can operate directly on rendered text.

## Files

- `pixel_notebook.py`: the interactive notebook
- `notebook_assets/bert-base-cased-vocab.txt`: bundled WordPiece vocabulary for offline tokenization
- `pixel/`: minimal local PIXEL renderer sources and font asset needed by the notebook

## Run locally

Open the notebook with marimo:

```bash
marimo edit pixel_notebook.py
```

The notebook includes inline dependency metadata for reproducible environments on molab.

## Open in molab

After pushing this repository to GitHub, open:

`https://molab.marimo.io/https://github.com/<owner>/<repo>/blob/main/pixel_notebook.py`
