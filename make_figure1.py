#!/usr/bin/env python3
"""Generate Figure 1 (end-to-end pipeline) for APIN-D-26-05359.

Run:  python3 make_figure1.py [-o OUTPUT.png] [--dpi 200]

Every number drawn here is a named constant below, each traceable to the source of truth:

  * hidden size 384, fused width 1152, 70.7M backbone / 71.3M total
        -> research_paper_revised.tex Sec. 3.4 ("dimension 384", "V in R^{1152}",
           "71.3M parameters, of which 70.7M reside in the shared encoder")
  * logit order [z_A, z_B, z_tie]
        -> notebook_src.py CFG.LABEL2NAME = {0: winner_model_a, 1: winner_model_b, 2: winner_tie}
  * separators " [TURN] " and " [RESPONSE] "
        -> notebook_src.py CFG.TURN_SEP / CFG.RESP_SEP, and greedy_multi_turn_context(), which
           builds each block as TURN_SEP + prompt + RESP_SEP + response
  * split sizes 44,477 / 2,000 / 3,000 / 8,000 of 57,477
        -> results/paper_results.json ["split_sizes"]
  * T = 1.682, log loss 1.0378, ECE 0.0712 -> 0.0116, flip consistency 0.8235 -> 0.9203
        -> results/paper_results.json ["calibration"], ["position_bias"]

This file exists because the previous diagram was drawn by hand and had drifted from the text: it
showed 768/2304 instead of 384/1152, labelled the encoders with the *total* parameter count rather
than the backbone's, ordered the logits [z_A, z_tie, z_B], and omitted the [RESPONSE] separator.
Generating it from constants keeps figure and manuscript in step.
"""

import argparse
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ----------------------------------------------------------------- source-of-truth constants
N_TOTAL = "57,477"
N_TRAIN_POOL = "44,477"
N_DEV = "2,000"
N_CALIB = "3,000"
N_EVAL = "8,000"

SEQ_LEN = 256
HIDDEN_DIM = 384           # DeBERTa-v3-extra-small hidden size
FUSED_DIM = 3 * HIDDEN_DIM  # [e_A ; e_B ; |e_A - e_B|] = 1152
BACKBONE_PARAMS = "70.7M"   # shared encoder only
TOTAL_PARAMS = "71.3M"      # encoder + classification head
CLS_HIDDEN = 512
DROPOUT = 0.1

TRAIN_ROWS = "12k"
AUG_ROWS = "24k"

TEMP = "1.682"
LOG_LOSS = "1.0378"
ECE_BEFORE, ECE_AFTER = "0.0712", "0.0116"
ECE_DELTA = "-83.7\\%"
FLIP_BEFORE, FLIP_AFTER = "0.8235", "0.9203"
FLIP_DELTA = "+9.7"          # full model vs unaugmented baseline, in percentage points

# ----------------------------------------------------------------- palette
DATA_EC, DATA_FC = "#6b6b5e", "#f0efe9"      # dataset / partition stages
TRAIN_EC, TRAIN_FC = "#1f5fa8", "#e8f1fb"    # training-data construction
HOLD_EC, HOLD_FC = "#b8860b", "#fdf6e3"      # calibration split
EVAL_EC, EVAL_FC = "#2e7d4f", "#eaf5ee"      # evaluation split
ENC_EC, ENC_FC = "#5b4b9a", "#eeebf8"        # encoder / pooling
HEAD_EC, HEAD_FC = "#1b6b52", "#e7f4ef"      # fusion / classifier

TITLE_FS, BODY_FS, SMALL_FS = 12.5, 10.0, 9.2


def box(ax, x, y, w, h, title, lines, ec, fc, dashed=False, title_fs=TITLE_FS):
    """Rounded box with a bold title and optional detail lines beneath it."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.10,rounding_size=0.18",
        linewidth=1.7, edgecolor=ec, facecolor=fc,
        linestyle=(0, (5, 3)) if dashed else "solid",
        mutation_aspect=1.0, zorder=2,
    ))
    cx = x + w / 2.0
    if lines:
        ax.text(cx, y + h * 0.68, title, ha="center", va="center",
                fontsize=title_fs, fontweight="bold", color=ec, zorder=3)
        step = h * 0.30 / max(len(lines), 1)
        y0 = y + h * 0.36
        for i, ln in enumerate(lines):
            ax.text(cx, y0 - i * step, ln, ha="center", va="center",
                    fontsize=SMALL_FS, color="#333333", zorder=3)
    else:
        ax.text(cx, y + h / 2.0, title, ha="center", va="center",
                fontsize=title_fs, fontweight="bold", color=ec, zorder=3)
    return SimpleNamespace(x=x, y=y, w=w, h=h, cx=cx, top=y + h, bot=y)


def arrow(ax, p, q, dashed=False, color="#2b2b2b", label=None, lx=0.0, ly=0.0):
    ax.add_patch(FancyArrowPatch(
        p, q, arrowstyle="-|>", mutation_scale=15,
        linewidth=1.5, color=color, zorder=1,
        linestyle=(0, (5, 3)) if dashed else "solid",
        shrinkA=0, shrinkB=0,
    ))
    if label:
        ax.text((p[0] + q[0]) / 2 + lx, (p[1] + q[1]) / 2 + ly, label,
                ha="center", va="center", fontsize=SMALL_FS, color=color, zorder=3)


def elbow(ax, p, q, dashed=False, color="#2b2b2b"):
    """Right-angled connector: horizontal from p, then vertical into q."""
    mid = (q[0], p[1])
    ax.plot([p[0], mid[0]], [p[1], mid[1]], color=color, linewidth=1.5,
            linestyle=(0, (5, 3)) if dashed else "solid", zorder=1)
    arrow(ax, mid, q, dashed=dashed, color=color)


def build(out_path, dpi):
    fig, ax = plt.subplots(figsize=(12.6, 13.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 108)
    ax.axis("off")

    # ---------------------------------------------------------------- 1. dataset
    b_data = box(ax, 20, 100, 60, 6.0, "LMSYS Chatbot Arena benchmark",
                 [f"{N_TOTAL} multi-turn interactions"], DATA_EC, DATA_FC)

    # ---------------------------------------------------------------- 2. partition
    b_part = box(ax, 20, 91.5, 60, 6.0, "Partition by interaction ID",
                 ["disjoint splits, applied before any augmentation"], DATA_EC, DATA_FC)
    arrow(ax, (b_data.cx, b_data.bot), (b_part.cx, b_part.top))

    # ---------------------------------------------------------------- 3. four splits
    sw, sy, sh = 21.5, 81.0, 6.4
    b_pool = box(ax, 2.0, sy, sw, sh, "Train pool", [N_TRAIN_POOL], TRAIN_EC, TRAIN_FC)
    b_dev = box(ax, 26.0, sy, sw, sh, "Dev split", [f"{N_DEV}, monitoring only"],
                DATA_EC, DATA_FC, dashed=True)
    b_cal = box(ax, 50.0, sy, sw, sh, "Calibration", [f"{N_CALIB}, held out"],
                HOLD_EC, HOLD_FC, dashed=True)
    b_ev = box(ax, 74.0, sy, sw, sh, "Held-out eval", [f"{N_EVAL}, held out"],
               EVAL_EC, EVAL_FC, dashed=True)

    ax.plot([b_part.cx, b_part.cx], [b_part.bot, 88.6], color="#2b2b2b", lw=1.5, zorder=1)
    ax.plot([b_pool.cx, b_ev.cx], [88.6, 88.6], color="#2b2b2b", lw=1.5, zorder=1)
    for b in (b_pool, b_dev, b_cal, b_ev):
        arrow(ax, (b.cx, 88.6), (b.cx, b.top))

    # ---------------------------------------------------------------- 4. swap augmentation
    b_swap = box(ax, 20, 71.0, 60, 6.4, "Swap augmentation",
                 [r"$(P, R_A, R_B) \leftrightarrow (P, R_B, R_A)$ with mirrored label",
                  f"train pool only:  {TRAIN_ROWS} rows $\\rightarrow$ {AUG_ROWS} training pairs"],
                 TRAIN_EC, TRAIN_FC)
    elbow(ax, (b_pool.cx, b_pool.bot - 1.2), (b_swap.cx, b_swap.top))
    ax.plot([b_pool.cx, b_pool.cx], [b_pool.bot, b_pool.bot - 1.2],
            color="#2b2b2b", lw=1.5, zorder=1)
    ax.text(b_pool.cx + 1.0, b_pool.bot - 2.4, "train pool only",
            ha="left", va="center", fontsize=SMALL_FS, color=TRAIN_EC)

    # ---------------------------------------------------------------- 5. greedy context
    b_ctx = box(ax, 20, 62.5, 60, 6.0, "Greedy multi-turn context selection",
                [f"newest turn first, never dropped;  $L \\leq {SEQ_LEN}$ tokens"],
                TRAIN_EC, TRAIN_FC)
    arrow(ax, (b_swap.cx, b_swap.bot), (b_ctx.cx, b_ctx.top))

    # ---------------------------------------------------------------- 6. tokenized pairs
    # The right-hand column stops at x=86 so the two dashed feedback lines (x=90, x=94.5)
    # have a clear channel and do not cut through any box.
    pw, py, ph = 34.0, 53.0, 6.6
    LX, RX = 8.0, 52.0
    tmpl_a = r"[CLS]  [TURN] $P$  [RESPONSE] $R_A$  [SEP]"
    tmpl_b = r"[CLS]  [TURN] $P$  [RESPONSE] $R_B$  [SEP]"
    b_ia = box(ax, LX, py, pw, ph, "Input A", [tmpl_a, "literal text markers, not special tokens"],
               ENC_EC, ENC_FC)
    b_ib = box(ax, RX, py, pw, ph, "Input B", [tmpl_b, "literal text markers, not special tokens"],
               ENC_EC, ENC_FC)
    ax.plot([b_ctx.cx, b_ctx.cx], [b_ctx.bot, 61.0], color="#2b2b2b", lw=1.5, zorder=1)
    ax.plot([b_ia.cx, b_ib.cx], [61.0, 61.0], color="#2b2b2b", lw=1.5, zorder=1)
    for b in (b_ia, b_ib):
        arrow(ax, (b.cx, 61.0), (b.cx, b.top))

    # ---------------------------------------------------------------- 7. shared encoder
    ey, eh = 43.0, 7.4
    enc_lines = [f"shared weights $\\theta$  ·  {BACKBONE_PARAMS} backbone",
                 f"$H_A \\in \\mathbb{{R}}^{{L \\times {HIDDEN_DIM}}}$"]
    enc_lines_b = [f"shared weights $\\theta$  ·  {BACKBONE_PARAMS} backbone",
                   f"$H_B \\in \\mathbb{{R}}^{{L \\times {HIDDEN_DIM}}}$"]
    b_ea = box(ax, LX, ey, pw, eh, "DeBERTa-v3-extra-small encoder", enc_lines, ENC_EC, ENC_FC)
    b_eb = box(ax, RX, ey, pw, eh, "DeBERTa-v3-extra-small encoder", enc_lines_b, ENC_EC, ENC_FC)
    arrow(ax, (b_ia.cx, b_ia.bot), (b_ea.cx, b_ea.top))
    arrow(ax, (b_ib.cx, b_ib.bot), (b_eb.cx, b_eb.top))

    # weight-tying link
    ax.add_patch(FancyArrowPatch(
        (b_ea.x + b_ea.w, ey + eh / 2), (b_eb.x, ey + eh / 2),
        arrowstyle="<|-|>", mutation_scale=13, linewidth=1.4,
        color=ENC_EC, zorder=1, shrinkA=0, shrinkB=0))
    ax.text((b_ea.x + b_ea.w + b_eb.x) / 2, ey + eh / 2 + 1.5, "tied", ha="center", va="center",
            fontsize=SMALL_FS, color=ENC_EC, style="italic")

    # ---------------------------------------------------------------- 8. pooling
    poy, poh = 34.5, 6.2
    b_pa = box(ax, LX, poy, pw, poh, "Masked mean pooling",
               [f"$e_A \\in \\mathbb{{R}}^{{{HIDDEN_DIM}}}$  (non-padding only)"],
               ENC_EC, ENC_FC)
    b_pb = box(ax, RX, poy, pw, poh, "Masked mean pooling",
               [f"$e_B \\in \\mathbb{{R}}^{{{HIDDEN_DIM}}}$  (non-padding only)"],
               ENC_EC, ENC_FC)
    arrow(ax, (b_ea.cx, b_ea.bot), (b_pa.cx, b_pa.top))
    arrow(ax, (b_eb.cx, b_eb.bot), (b_pb.cx, b_pb.top))

    # ---------------------------------------------------------------- 9. fusion
    b_fuse = box(ax, 18, 25.0, 64, 6.6, "Symmetric feature fusion",
                 [r"$V = [\, e_A \,;\, e_B \,;\, |e_A - e_B| \,]"
                  f" \\in \\mathbb{{R}}^{{{FUSED_DIM}}}$",
                  r"the $|e_A - e_B|$ term is invariant under exchange of the two responses"],
                 HEAD_EC, HEAD_FC)
    elbow(ax, (b_pa.cx, b_pa.bot - 1.6), (b_fuse.cx, b_fuse.top))
    elbow(ax, (b_pb.cx, b_pb.bot - 1.6), (b_fuse.cx, b_fuse.top))
    for b in (b_pa, b_pb):
        ax.plot([b.cx, b.cx], [b.bot, b.bot - 1.6], color="#2b2b2b", lw=1.5, zorder=1)

    # ---------------------------------------------------------------- 10. classifier
    b_cls = box(ax, 18, 15.5, 64, 6.6, "Classification head",
                [f"Dense({CLS_HIDDEN}, ReLU) $\\rightarrow$ dropout({DROPOUT}) $\\rightarrow$ "
                 r"softmax logits $[\, z_A ,\; z_B ,\; z_{\mathrm{tie}} \,]$",
                 f"total model {TOTAL_PARAMS} parameters"],
                HEAD_EC, HEAD_FC)
    arrow(ax, (b_fuse.cx, b_fuse.bot), (b_cls.cx, b_cls.top))

    # ---------------------------------------------------------------- 11. calibration + results
    b_cal2 = box(ax, 18, 5.5, 64, 7.2, "Post-hoc temperature calibration",
                 [f"$T = {TEMP}$ fitted on the calibration split, applied unchanged to the eval split",
                  f"log loss {LOG_LOSS}  ·  ECE {ECE_BEFORE} $\\rightarrow$ {ECE_AFTER} "
                  f"(${ECE_DELTA}$)  ·  flip consistency {FLIP_BEFORE} $\\rightarrow$ {FLIP_AFTER} "
                  f"(${FLIP_DELTA}$ pp)"],
                 HOLD_EC, HOLD_FC)
    arrow(ax, (b_cls.cx, b_cls.bot), (b_cal2.cx, b_cal2.top))

    # Dashed feedback, routed down the clear channel to the right of the encoder column
    # (which now ends at x=86): the calibration split fits T, the eval split is only ever scored.
    CAL_CH, EVAL_CH = 90.0, 95.0

    ax.plot([b_cal.cx, b_cal.cx], [b_cal.bot, 79.0], color=HOLD_EC, lw=1.4,
            linestyle=(0, (5, 3)), zorder=1)
    ax.plot([b_cal.cx, CAL_CH], [79.0, 79.0], color=HOLD_EC, lw=1.4,
            linestyle=(0, (5, 3)), zorder=1)
    ax.plot([CAL_CH, CAL_CH], [79.0, 9.1], color=HOLD_EC, lw=1.4,
            linestyle=(0, (5, 3)), zorder=1)
    arrow(ax, (CAL_CH, 9.1), (b_cal2.x + b_cal2.w, 9.1), dashed=True, color=HOLD_EC)
    ax.text(CAL_CH - 1.3, 44.0, "fit $T$", rotation=90, ha="center", va="center",
            fontsize=SMALL_FS, color=HOLD_EC)

    ax.plot([b_ev.cx, b_ev.cx], [b_ev.bot, 76.0], color=EVAL_EC, lw=1.4,
            linestyle=(0, (5, 3)), zorder=1)
    ax.plot([b_ev.cx, EVAL_CH], [76.0, 76.0], color=EVAL_EC, lw=1.4,
            linestyle=(0, (5, 3)), zorder=1)
    ax.plot([EVAL_CH, EVAL_CH], [76.0, 7.2], color=EVAL_EC, lw=1.4,
            linestyle=(0, (5, 3)), zorder=1)
    arrow(ax, (EVAL_CH, 7.2), (b_cal2.x + b_cal2.w, 7.2), dashed=True, color=EVAL_EC)
    ax.text(EVAL_CH - 1.3, 42.0, "score only", rotation=90, ha="center", va="center",
            fontsize=SMALL_FS, color=EVAL_EC)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(out_path, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"wrote {out_path} at {dpi} dpi")
    print(f"  hidden={HIDDEN_DIM}  fused={FUSED_DIM}  backbone={BACKBONE_PARAMS}  "
          f"total={TOTAL_PARAMS}  logits=[z_A, z_B, z_tie]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", default="pipeline_diagram.png")
    ap.add_argument("--dpi", type=int, default=200)
    a = ap.parse_args()
    build(a.out, a.dpi)
