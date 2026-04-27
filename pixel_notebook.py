# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "marimo==0.23.3",
#   "numpy>=1.26,<3",
#   "matplotlib>=3.9,<4",
#   "transformers==4.17.0",
#   "pillow>=10,<13",
#   "pycairo==1.29.0",
#   "PyGObject==3.50.0",
#   "manimpango==0.6.1",
#   "fonttools>=4.0",
# ]
# ///

import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import sys
    import types

    import matplotlib.pyplot as plt
    import numpy as np

    # marimo 0.23.x registers a transformers formatter that expects
    # TextIteratorStreamer, which is unavailable in transformers 4.17.
    # This shim keeps notebook imports working while preserving the pinned
    # transformers version used by the local PIXEL codebase.
    from marimo._output.formatters import formatters as mo_formatters

    transformers_factory = mo_formatters.THIRD_PARTY_FACTORIES.get("transformers")
    if transformers_factory is not None and not getattr(transformers_factory, "_pixel_safe_patch", False):
        original_register = transformers_factory.register

        def _safe_register():
            try:
                original_register()
            except ImportError:
                return

        transformers_factory.register = _safe_register
        setattr(transformers_factory, "_pixel_safe_patch", True)

    notebook_root = Path(__file__).resolve().parent
    local_pixel_package = notebook_root / "pixel" / "src" / "pixel"

    if local_pixel_package.exists():
        for module_name in list(sys.modules):
            if module_name == "pixel" or module_name.startswith("pixel."):
                del sys.modules[module_name]

        namespace_paths = {
            "pixel": local_pixel_package,
            "pixel.data": local_pixel_package / "data",
            "pixel.data.rendering": local_pixel_package / "data" / "rendering",
            "pixel.utils": local_pixel_package / "utils",
        }
        for module_name, module_path in namespace_paths.items():
            namespace_module = types.ModuleType(module_name)
            namespace_module.__path__ = [str(module_path)]
            sys.modules[module_name] = namespace_module

    from pixel.data.rendering.pangocairo_renderer import PangoCairoTextRenderer
    from transformers import BertTokenizer

    return BertTokenizer, PangoCairoTextRenderer, mo, notebook_root, np, plt


@app.cell
def _(mo):
    mo.md("""
    # PIXEL vs BPE: Language Modelling with Pixels

    This notebook is a focused, interactive reconstruction of one core idea from
    *Language Modelling with Pixels*.

    ## Why this notebook exists

    The paper argues that token-based language models suffer from a **vocabulary bottleneck**:
    they must map text into a fixed inventory of subword units before any reasoning begins.
    PIXEL takes a different route. It renders text as an image and reasons over visual patches,
    which lets it exploit **orthographic similarity** directly.

    In the spirit of the molab competition, this notebook does **not** try to reproduce the
    full multilingual benchmark suite from the paper. Instead, it turns one central contribution
    into something tangible, reproducible, and interactive:

    - small spelling changes can disrupt BPE/WordPiece badly
    - visually similar strings can still look almost the same to PIXEL
    - cross-lingual and unseen-character examples expose the vocabulary bottleneck clearly

    ## How to use this notebook

    1. Read each section intro first.
    2. Interact with the controls and compare both representations.
    3. Focus on when tokenization breaks but visual structure remains stable.

    ## Roadmap

    - Section A: Tokenizer Sandbox and Typo Robustness (current section)
    - Section B: Cross-Lingual and Unseen Characters

    By the end, you should have an intuitive understanding of orthographic similarity and
    why a pixel-based encoder can be robust to character-level noise.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Section A: Tokenizer Sandbox and Typo Robustness

    Goal: contrast WordPiece tokenization against PIXEL rendering, then connect that view
    directly to typo robustness.

    Left panel: subword decomposition from a bundled `bert-base-cased` WordPiece vocabulary.
    Right panel: rendered text with a 16x16 patch grid, matching PIXEL's ViT input.

    We start with a misspelling example from Wikipedia's
    [commonly misspelled English words](https://en.wikipedia.org/wiki/Commonly_misspelled_English_words):
    `absence - absense, abcense`.

    The key intuition: a small spelling change can leave the word visually very close to the
    original, but force BPE/WordPiece into a more fragmented decomposition with different token
    identities. Section 4 of the PIXEL paper shows why this matters: under orthographic noise,
    performance can drop sharply for token-based models, while PIXEL is often more stable because
    it still sees nearly the same shape on the page.
    """)
    return


@app.cell
def _(BertTokenizer, PangoCairoTextRenderer, mo, notebook_root):
    tokenizer_name = "bert-base-cased (bundled vocab)"
    vocab_path = notebook_root / "notebook_assets" / "bert-base-cased-vocab.txt"
    if not vocab_path.exists():
        raise FileNotFoundError(
            "Could not locate notebook_assets/bert-base-cased-vocab.txt. "
            "This file is bundled so the notebook can run on molab without "
            "downloading tokenizer assets at runtime."
        )
    tokenizer = BertTokenizer(vocab_file=str(vocab_path), do_lower_case=False)

    font_path = (notebook_root / "pixel" / "configs" / "renderers" / "noto_renderer" / "GoNotoCurrent.ttf").resolve()

    if not font_path.exists():
        raise FileNotFoundError(
            "Could not locate GoNotoCurrent.ttf. Expected at "
            "pixel/configs/renderers/noto_renderer/GoNotoCurrent.ttf relative to the notebook."
        )

    renderer = PangoCairoTextRenderer(font_file=str(font_path), rgb=False)

    status = mo.callout(
        mo.md(
            f"""
            **Assets loaded successfully**

            - WordPiece tokenizer: {tokenizer_name}
            - Bundled tokenizer vocab: {vocab_path.name}
            - PIXEL font file: {font_path.name}
            - PIXEL renderer source: local `pixel/src`
            - Patch size: {renderer.pixels_per_patch} x {renderer.pixels_per_patch} pixels
            """
        ),
        kind="success",
    )
    return renderer, status, tokenizer


@app.cell(hide_code=True)
def _(mo):
    text_input = mo.ui.text(
        value="absence - absense, abcense",
        placeholder="Try: langauge, language modelling, apples",
        label="Enter a word or phrase",
        debounce=150,
        full_width=True,
    )
    return (text_input,)


@app.cell(hide_code=True)
def _(np, renderer, text_input, tokenizer):
    text = text_input.value.strip() or "language"

    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    encoding = renderer(text)
    pixel_values = np.asarray(encoding.pixel_values)
    num_text_patches = int(getattr(encoding, "num_text_patches", 0))
    return num_text_patches, pixel_values, text, token_ids, tokens


@app.cell(hide_code=True)
def _(mo, text, token_ids, tokens):
    if tokens:
        rows = "\n".join(
            f"| {idx} | {tok} | {tid} |"
            for idx, (tok, tid) in enumerate(zip(tokens, token_ids), start=1)
        )
        token_table = f"""
    ### WordPiece tokenization

    Input: **{text}**

    Total WordPiece tokens: **{len(tokens)}**

    | # | Token | Token ID |
    |---:|:------|---:|
    {rows}
    """
    else:
        token_table = """
    ### WordPiece tokenization

    No tokens produced for the current input.
    """

    token_panel = mo.md(token_table)
    return (token_panel,)


@app.cell(hide_code=True)
def _(mo, np, num_text_patches, pixel_values, plt, renderer, text):
    height, width = pixel_values.shape
    patch = renderer.pixels_per_patch

    # Show a human-readable crop of the actually used text patches.
    # Add one extra patch so the end of the rendered text is not cramped.
    if num_text_patches > 0:
        active_width = min(width, (num_text_patches + 1) * patch)
    else:
        active_width = min(width, 12 * patch)

    visible = pixel_values[:, :active_width]
    visible_patches = max(1, active_width // patch)
    tick_stride = patch * max(1, visible_patches // 10)

    fig_width = float(np.clip(visible_patches * 0.45, 8.0, 20.0))
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(fig_width, 4.8),
        dpi=140,
        gridspec_kw={"height_ratios": [3.0, 1.2]},
    )
    ax_zoom, ax_full = axes

    ax_zoom.imshow(
        visible,
        cmap="gray",
        vmin=0,
        vmax=255,
        interpolation="nearest",
        aspect="auto",
    )

    for x in range(0, active_width + 1, patch):
        ax_zoom.axvline(x - 0.5, color="#00A3A3", alpha=0.45, linewidth=0.9)
    for y in range(0, height + 1, patch):
        ax_zoom.axhline(y - 0.5, color="#00A3A3", alpha=0.65, linewidth=0.9)

    ax_zoom.set_title(f'PIXEL rendering (zoomed) for: "{text}"')
    ax_zoom.set_xlabel("x position in pixels (patch boundaries shown)")
    ax_zoom.set_ylabel("y")
    ax_zoom.set_xticks(np.arange(0, active_width + 1, tick_stride))
    ax_zoom.set_yticks([0, patch - 1])
    ax_zoom.set_yticklabels(["0", str(patch - 1)])
    ax_zoom.set_xlim(-0.5, active_width - 0.5)
    ax_zoom.set_ylim(height - 0.5, -0.5)

    # Full-canvas overview keeps model context visible while highlighting
    # the portion used by the current text.
    ax_full.imshow(
        pixel_values,
        cmap="gray",
        vmin=0,
        vmax=255,
        interpolation="nearest",
        aspect="auto",
    )
    ax_full.axvspan(
        -0.5,
        active_width - 0.5,
        facecolor="#00A3A3",
        alpha=0.12,
        edgecolor="none",
    )
    ax_full.set_title("Full model canvas (highlighted = text-used region)")
    ax_full.set_xlabel("full PIXEL canvas width")
    ax_full.set_ylabel("y")
    ax_full.set_xticks(np.arange(0, width + 1, patch * 32))
    ax_full.set_yticks([0, patch - 1])
    ax_full.set_yticklabels(["0", str(patch - 1)])
    ax_full.set_xlim(-0.5, width - 0.5)
    ax_full.set_ylim(height - 0.5, -0.5)

    fig.tight_layout()

    image_panel = mo.vstack(
        [
            mo.md("### PIXEL renderer output"),
            mo.as_html(fig),
            mo.md(
                f"Zoomed view: **{height} x {active_width}** pixels "
                f"(~**{visible_patches}** patches)."
            ),
            mo.md(
                f"Full model canvas: **{height} x {width}** pixels with "
                f"**{patch} x {patch}** patch size."
            ),
        ],
        gap=0.5,
    )

    plt.close(fig)
    return (image_panel,)


@app.cell
def _(image_panel, mo, status, text_input, token_panel):
    mo.vstack(
        [
            status,
            text_input,
            mo.hstack(
                [token_panel, image_panel],
                widths=[0.38, 0.62],
                align="start",
                gap=1.0,
            ),
        ],
        gap=0.9,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Section B: Cross-Lingual and Unseen Characters

    The paper also argues that PIXEL can transfer across languages because it never commits to a
    fixed token vocabulary before reading the input. If two strings share visual structure,
    PIXEL can reuse that structure even when the tokenizer vocabulary becomes awkward, fragmented,
    or incomplete.

    This section is a small mechanism demo rather than a full benchmark reproduction. The goal is
    to make the paper's claim visible: tokenization is a discrete bottleneck, while rendering is a
    continuous visual process.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    preset_pairs = {
        "English vs umlaut": ("apple", "Äpfel"),
        "Accent variation": ("cafe", "café"),
        "Latin vs Cyrillic": ("Penguins", "Пингвины"),
        "Latin vs Japanese": ("hello", "こんにちは"),
        "Latin vs Arabic": ("hello", "مرحبا"),
    }
    pair_choice = mo.ui.dropdown(
        options=list(preset_pairs.keys()),
        value="English vs umlaut",
        label="Choose a comparison pair",
        full_width=True,
    )
    return pair_choice, preset_pairs


@app.cell
def _(mo, pair_choice, preset_pairs):
    default_left, default_right = preset_pairs[pair_choice.value]
    left_text_input = mo.ui.text(
        value=default_left,
        label="Left example",
        debounce=150,
        full_width=True,
    )
    right_text_input = mo.ui.text(
        value=default_right,
        label="Right example",
        debounce=150,
        full_width=True,
    )
    mo.vstack(
        [
            pair_choice,
            mo.hstack(
                [left_text_input, right_text_input],
                widths=[0.5, 0.5],
                gap=1.0,
                align="start",
            ),
        ],
        gap=0.8,
    )
    return left_text_input, right_text_input


@app.cell(hide_code=True)
def _(left_text_input, np, renderer, right_text_input, tokenizer):
    def collect_representation(text):
        normalized = text.strip() or "text"
        tokens = tokenizer.tokenize(normalized)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        encoding = renderer(normalized)
        pixel_values = np.asarray(encoding.pixel_values)
        num_text_patches = int(getattr(encoding, "num_text_patches", 0))
        return {
            "text": normalized,
            "tokens": tokens,
            "token_ids": token_ids,
            "pixel_values": pixel_values,
            "num_text_patches": num_text_patches,
        }

    left_repr = collect_representation(left_text_input.value)
    right_repr = collect_representation(right_text_input.value)
    return left_repr, right_repr


@app.cell(hide_code=True)
def _(left_repr, mo, np, plt, renderer, right_repr):
    section_b_patch = renderer.pixels_per_patch

    def build_token_panel(title, payload):
        tokens = payload["tokens"]
        token_ids = payload["token_ids"]
        unk_count = sum(tok == "[UNK]" for tok in tokens)
        if tokens:
            rows = "\n".join(
                f"| {idx} | {tok} | {tid} |"
                for idx, (tok, tid) in enumerate(zip(tokens, token_ids), start=1)
            )
            body = f"""
    ### {title}

    Input: **{payload["text"]}**

    - WordPiece tokens: **{len(tokens)}**
    - `[UNK]` tokens: **{unk_count}**
    - PIXEL text patches: **{payload["num_text_patches"]}**

    | # | Token | Token ID |
    |---:|:------|---:|
    {rows}
    """
        else:
            body = f"""
    ### {title}

    Input: **{payload["text"]}**

    No tokens produced for the current input.
    """
        return mo.md(body)

    def build_render_panel(title, payload):
        pixel_values = payload["pixel_values"]
        height, width = pixel_values.shape
        num_text_patches = payload["num_text_patches"]

        if num_text_patches > 0:
            active_width = min(width, (num_text_patches + 1) * section_b_patch)
        else:
            active_width = min(width, 10 * section_b_patch)

        visible = pixel_values[:, :active_width]
        visible_patches = max(1, active_width // section_b_patch)
        fig_width = float(np.clip(visible_patches * 0.45, 5.5, 10.5))

        fig, ax = plt.subplots(1, 1, figsize=(fig_width, 2.2), dpi=140)
        ax.imshow(
            visible,
            cmap="gray",
            vmin=0,
            vmax=255,
            interpolation="nearest",
            aspect="auto",
        )
        for x in range(0, active_width + 1, section_b_patch):
            ax.axvline(x - 0.5, color="#00A3A3", alpha=0.45, linewidth=0.9)
        for y in range(0, height + 1, section_b_patch):
            ax.axhline(y - 0.5, color="#00A3A3", alpha=0.65, linewidth=0.9)

        ax.set_title(title)
        ax.set_xlabel("patch-aligned pixel strip")
        ax.set_ylabel("y")
        ax.set_xticks([])
        ax.set_yticks([0, section_b_patch - 1])
        ax.set_yticklabels(["0", str(section_b_patch - 1)])
        ax.set_xlim(-0.5, active_width - 0.5)
        ax.set_ylim(height - 0.5, -0.5)
        fig.tight_layout()

        panel = mo.vstack(
            [
                mo.as_html(fig),
                mo.md(
                    f"Visible text region: **{visible_patches}** patches "
                    f"for **{payload['text']}**."
                ),
            ],
            gap=0.4,
        )
        plt.close(fig)
        return panel

    left_tokens = len(left_repr["tokens"])
    right_tokens = len(right_repr["tokens"])
    left_unk = sum(tok == "[UNK]" for tok in left_repr["tokens"])
    right_unk = sum(tok == "[UNK]" for tok in right_repr["tokens"])

    if left_tokens == right_tokens:
        token_message = (
            "Both inputs use the same number of WordPiece tokens here, but the token identities "
            "can still shift substantially."
        )
    elif left_tokens > right_tokens:
        token_message = (
            f"The left input is more fragmented for WordPiece: **{left_tokens}** tokens versus "
            f"**{right_tokens}**."
        )
    else:
        token_message = (
            f"The right input is more fragmented for WordPiece: **{right_tokens}** tokens versus "
            f"**{left_tokens}**."
        )

    if left_unk or right_unk:
        unk_message = (
            f"`[UNK]` appears in this comparison (left: **{left_unk}**, right: **{right_unk}**), "
            "which makes the vocabulary bottleneck especially visible."
        )
    else:
        unk_message = (
            "This pair does not trigger `[UNK]`, which is useful too: the bottleneck can show up "
            "as fragmentation and altered token identities even when coverage is not completely lost."
        )

    summary_panel = mo.callout(
        mo.md(
            f"""
            **What to notice**

            - {token_message}
            - {unk_message}
            - PIXEL renders both strings directly as patch grids, so it does not need a separate
              vocabulary entry before it can represent the text.
            """
        ),
        kind="info",
    )

    left_token_panel = build_token_panel("Left-side tokenization", left_repr)
    right_token_panel = build_token_panel("Right-side tokenization", right_repr)
    left_render_panel = build_render_panel("Left-side PIXEL rendering", left_repr)
    right_render_panel = build_render_panel("Right-side PIXEL rendering", right_repr)

    mo.vstack(
        [
            summary_panel,
            mo.hstack(
                [left_token_panel, right_token_panel],
                widths=[0.5, 0.5],
                gap=1.0,
                align="start",
            ),
            mo.hstack(
                [left_render_panel, right_render_panel],
                widths=[0.5, 0.5],
                gap=1.0,
                align="start",
            ),
        ],
        gap=0.9,
    )
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            """
            ## Conclusion

            This notebook intentionally zooms in on one core contribution from the paper rather than
            trying to replay the full experimental pipeline.

            Section A makes **typo robustness** concrete: small orthographic changes can create a large
            change in WordPiece segmentation, even when the rendered text still looks almost the same.

            Section B makes the **vocabulary bottleneck** concrete: cross-lingual variants, diacritics,
            and unfamiliar scripts can push token-based models into awkward segmentations or `[UNK]`,
            while PIXEL still receives a directly rendered visual signal.

            That is the main takeaway from this competition-style notebook: the paper's argument becomes
            easier to trust once you can poke at it yourself, change the text, and watch the bottleneck
            appear in real time.
            """
        ),
        kind="success",
    )
    return


if __name__ == "__main__":
    app.run()
