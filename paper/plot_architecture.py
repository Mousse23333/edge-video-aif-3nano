#!/usr/bin/env python3
"""
System architecture diagram for the AIF edge scheduling paper.
Generates figures/architecture.pdf using matplotlib patches and annotations.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

# ── colour palette ──────────────────────────────────────────────────
C_ORIN_BG   = "#E8F0FE"
C_NANO_BG   = "#FFF3E0"
C_AIF_BG    = "#E8F5E9"
C_ENGINE    = "#BBDEFB"
C_STREAM    = "#C8E6C9"
C_CTRL      = "#A5D6A7"
C_NANO_SRV  = "#FFE0B2"
C_VIDEO     = "#F5F5F5"
C_OBS       = "#FFF9C4"
C_DATA      = "#1565C0"
C_CTRL_ARR  = "#C62828"
C_NET       = "#6A1B9A"


def rbox(ax, x, y, w, h, fc="#fff", ec="#333", lw=1.2, zorder=2, ls="-"):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008",
                         facecolor=fc, edgecolor=ec, linewidth=lw,
                         linestyle=ls, zorder=zorder)
    ax.add_patch(box)
    return box


def txt(ax, x, y, s, **kw):
    defaults = dict(ha="center", va="center", fontsize=7.5,
                    fontweight="bold", family="sans-serif", zorder=8)
    defaults.update(kw)
    ax.text(x, y, s, **defaults)


def stxt(ax, x, y, s, **kw):
    defaults = dict(ha="center", va="center", fontsize=5.5,
                    family="sans-serif", color="#555", zorder=8)
    defaults.update(kw)
    ax.text(x, y, s, **defaults)


def arr(ax, x1, y1, x2, y2, color="#333", lw=1.2, rad=0, zorder=5):
    conn = f"arc3,rad={rad}" if rad else "arc3,rad=0"
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                connectionstyle=conn, shrinkA=1, shrinkB=1),
                zorder=zorder)


def alabel(ax, x, y, s, color="#333", fontsize=5.5):
    ax.text(x, y, s, ha="center", va="center", fontsize=fontsize,
            color=color, fontstyle="italic", zorder=10,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9))


def main():
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    # ================================================================
    # BACKGROUND REGIONS
    # ================================================================
    # Orin
    rbox(ax, 0.015, 0.23, 0.60, 0.75, fc=C_ORIN_BG, ec="#1565C0",
         lw=2.0, ls="--", zorder=0)
    txt(ax, 0.315, 0.96, "Coordinator — NVIDIA Jetson AGX Orin",
        fontsize=10, color="#0D47A1")

    # Nano cluster
    rbox(ax, 0.66, 0.38, 0.325, 0.60, fc=C_NANO_BG, ec="#E65100",
         lw=2.0, ls="--", zorder=0)
    txt(ax, 0.822, 0.96, "Offload Nodes", fontsize=10, color="#BF360C")

    # ================================================================
    # 1) VIDEO SOURCE
    # ================================================================
    vx, vy = 0.035, 0.87
    vw, vh = 0.12, 0.055
    rbox(ax, vx, vy, vw, vh, fc=C_VIDEO, ec="#616161")
    txt(ax, vx + vw/2, vy + vh/2 + 0.008, "Video Streams")
    stxt(ax, vx + vw/2, vy + vh/2 - 0.012, "720p traffic × N")

    # ================================================================
    # 2) STREAM MANAGER
    # ================================================================
    sm_x, sm_y, sm_w, sm_h = 0.035, 0.74, 0.27, 0.095
    rbox(ax, sm_x, sm_y, sm_w, sm_h, fc="#E8F5E9", ec="#388E3C", lw=1.5, zorder=1)
    txt(ax, sm_x + sm_w/2, sm_y + sm_h - 0.014, "StreamManager",
        fontsize=8, color="#2E7D32")

    n_sw = 5
    bw = 0.042
    gap = 0.008
    total = n_sw * bw + (n_sw - 1) * gap
    sx0 = sm_x + (sm_w - total) / 2
    for i in range(n_sw):
        bx = sx0 + i * (bw + gap)
        rbox(ax, bx, sm_y + 0.012, bw, 0.042, fc=C_STREAM, ec="#388E3C", lw=0.8)
        txt(ax, bx + bw/2, sm_y + 0.033 + 0.005, f"SW{i}", fontsize=6.5)
        stxt(ax, bx + bw/2, sm_y + 0.018 + 0.003, f"strm{i}", fontsize=5)

    # Video → StreamManager
    arr(ax, vx + vw/2, vy, sm_x + sm_w/2, sm_y + sm_h, color=C_DATA, lw=1.5)

    # ================================================================
    # 3) MODE ROUTER
    # ================================================================
    mr_x, mr_y, mr_w, mr_h = 0.08, 0.655, 0.20, 0.048
    rbox(ax, mr_x, mr_y, mr_w, mr_h, fc="#E3F2FD", ec="#1565C0")
    txt(ax, mr_x + mr_w/2, mr_y + mr_h/2 + 0.007, "Mode Router", fontsize=8)
    stxt(ax, mr_x + mr_w/2, mr_y + mr_h/2 - 0.010, "FULL  |  LITE  |  SKIP  |  OFFLOAD")

    # SM → Router
    arr(ax, sm_x + sm_w/2, sm_y, mr_x + mr_w/2, mr_y + mr_h, color=C_DATA, lw=1.3)

    # ================================================================
    # 4) INFERENCE ENGINES + SKIP
    # ================================================================
    ey = 0.52
    ew, eh = 0.115, 0.085

    # FULL
    fe_x = 0.085
    rbox(ax, fe_x, ey, ew, eh, fc=C_ENGINE, ec="#0D47A1")
    txt(ax, fe_x + ew/2, ey + eh/2 + 0.01, "BatchInference", fontsize=7)
    stxt(ax, fe_x + ew/2, ey + eh/2 - 0.01, "FULL  (imgsz=640)")
    # GPU badge
    txt(ax, fe_x + ew/2, ey + eh + 0.013, "GPU", fontsize=6, color="#0D47A1",
        bbox=dict(boxstyle="round,pad=0.06", fc="#E3F2FD", ec="#0D47A1", lw=0.7))

    # LITE
    le_x = 0.225
    rbox(ax, le_x, ey, ew, eh, fc=C_ENGINE, ec="#0D47A1")
    txt(ax, le_x + ew/2, ey + eh/2 + 0.01, "BatchInference", fontsize=7)
    stxt(ax, le_x + ew/2, ey + eh/2 - 0.01, "LITE  (imgsz=320)")
    txt(ax, le_x + ew/2, ey + eh + 0.013, "GPU", fontsize=6, color="#0D47A1",
        bbox=dict(boxstyle="round,pad=0.06", fc="#E3F2FD", ec="#0D47A1", lw=0.7))

    # SKIP
    sk_x, sk_w, sk_h = 0.365, 0.09, 0.055
    sk_y = ey + (eh - sk_h) / 2
    rbox(ax, sk_x, sk_y, sk_w, sk_h, fc="#FFECB3", ec="#F57F17")
    txt(ax, sk_x + sk_w/2, sk_y + sk_h/2 + 0.007, "SKIP", fontsize=7)
    stxt(ax, sk_x + sk_w/2, sk_y + sk_h/2 - 0.010, "passthrough")

    # Router → engines
    arr(ax, mr_x + 0.03, mr_y, fe_x + ew/2, ey + eh, color=C_DATA, lw=1.2)
    arr(ax, mr_x + mr_w/2, mr_y, le_x + ew/2, ey + eh, color=C_DATA, lw=1.2)
    arr(ax, mr_x + mr_w - 0.03, mr_y, sk_x + sk_w/2, sk_y + sk_h, color=C_DATA, lw=1.2)

    # ================================================================
    # 5) OFFLOAD PATH → Nanos
    # ================================================================
    arr(ax, mr_x + mr_w, mr_y + mr_h/2, 0.685, 0.78,
        color=C_NET, lw=2.0, rad=-0.15)
    alabel(ax, 0.50, 0.735, "HTTP POST\n(JPEG frames)", color=C_NET, fontsize=6)

    # Nano 2
    n2_x, n2_y, n2_w, n2_h = 0.685, 0.71, 0.275, 0.12
    rbox(ax, n2_x, n2_y, n2_w, n2_h, fc=C_NANO_SRV, ec="#E65100", lw=1.3)
    txt(ax, n2_x + n2_w/2, n2_y + n2_h - 0.018,
        "Nano 2  (192.168.1.52)", fontsize=8, color="#BF360C")
    rbox(ax, n2_x + 0.02, n2_y + 0.015, n2_w - 0.04, 0.052,
         fc="#FFF8E1", ec="#E65100", lw=0.8)
    txt(ax, n2_x + n2_w/2, n2_y + 0.041 + 0.007, "nano_server.py",
        fontsize=7, family="monospace")
    stxt(ax, n2_x + n2_w/2, n2_y + 0.041 - 0.012, "YOLOv8n · CUDA 12.6")

    # Nano 3
    n3_y = 0.55
    rbox(ax, n2_x, n3_y, n2_w, n2_h, fc=C_NANO_SRV, ec="#E65100", lw=1.3)
    txt(ax, n2_x + n2_w/2, n3_y + n2_h - 0.018,
        "Nano 3  (192.168.1.53)", fontsize=8, color="#BF360C")
    rbox(ax, n2_x + 0.02, n3_y + 0.015, n2_w - 0.04, 0.052,
         fc="#FFF8E1", ec="#E65100", lw=0.8)
    txt(ax, n2_x + n2_w/2, n3_y + 0.041 + 0.007, "nano_server.py",
        fontsize=7, family="monospace")
    stxt(ax, n2_x + n2_w/2, n3_y + 0.041 - 0.012, "YOLOv8n · CUDA 12.6")

    # Return arrows
    arr(ax, n2_x, n2_y + 0.03, 0.46, 0.465, color=C_NET, lw=1.2, rad=0.2)
    arr(ax, n2_x, n3_y + 0.03, 0.46, 0.455, color=C_NET, lw=1.2, rad=0.12)
    alabel(ax, 0.575, 0.52, "JSON {detections}", color=C_NET, fontsize=5.5)

    # GbE label
    txt(ax, 0.645, 0.65, "GbE\nLAN", fontsize=7, color=C_NET, rotation=0,
        bbox=dict(boxstyle="round,pad=0.08", fc="#F3E5F5", ec=C_NET, lw=0.8))

    # ================================================================
    # 6) OBSERVATION COLLECTOR
    # ================================================================
    oc_x, oc_y, oc_w, oc_h = 0.035, 0.40, 0.40, 0.065
    rbox(ax, oc_x, oc_y, oc_w, oc_h, fc=C_OBS, ec="#F9A825", lw=1.3)
    txt(ax, oc_x + oc_w/2, oc_y + oc_h/2 + 0.009, "ObservationCollector",
        fontsize=8)
    stxt(ax, oc_x + oc_w/2, oc_y + oc_h/2 - 0.012,
         "FPS · P95 latency · GPU util · mode counts  →  {GOOD, MARGINAL, BAD}")

    # Engines → ObsCollector
    arr(ax, fe_x + ew/2, ey, oc_x + 0.10, oc_y + oc_h, color=C_DATA, lw=1.0)
    arr(ax, le_x + ew/2, ey, oc_x + 0.20, oc_y + oc_h, color=C_DATA, lw=1.0)
    arr(ax, sk_x + sk_w/2, sk_y, oc_x + 0.32, oc_y + oc_h, color=C_DATA, lw=1.0)

    # ================================================================
    # 7) CONTROLLER BAR
    # ================================================================
    cb_x, cb_y, cb_w, cb_h = 0.035, 0.275, 0.56, 0.075
    rbox(ax, cb_x, cb_y, cb_w, cb_h, fc=C_CTRL, ec="#2E7D32", lw=1.6, zorder=2)
    txt(ax, cb_x + 0.055, cb_y + cb_h/2, "Controller",
        fontsize=9, color="#1B5E20")

    # Pluggable controllers
    cnames = ["AIF", "Heuristic", "Myopic", "DQN", "No-Op"]
    ccolors = [("#C8E6C9", "#2E7D32"), ("#E1F5FE", "#0277BD"),
               ("#FFF3E0", "#E65100"), ("#FCE4EC", "#AD1457"),
               ("#EEEEEE", "#616161")]
    c_bw = 0.076
    c_gap = 0.012
    c_total = len(cnames) * c_bw + (len(cnames)-1) * c_gap
    c_x0 = cb_x + 0.12
    for i, (name, (cfc, cec)) in enumerate(zip(cnames, ccolors)):
        cx = c_x0 + i * (c_bw + c_gap)
        rbox(ax, cx, cb_y + 0.015, c_bw, 0.045, fc=cfc, ec=cec, lw=1.0)
        txt(ax, cx + c_bw/2, cb_y + 0.015 + 0.0225, name, fontsize=7, color=cec)

    # ObsCollector → Controller
    arr(ax, oc_x + oc_w/2, oc_y, cb_x + cb_w/2, cb_y + cb_h, color=C_DATA, lw=1.5)
    alabel(ax, oc_x + oc_w/2 + 0.05, oc_y - 0.015, "obs(t)", color=C_DATA)

    # Controller → StreamManager (control loop — routed outside left edge)
    # Use a multi-segment path: controller left → far left → stream manager left
    ctrl_out_x = cb_x - 0.005
    ctrl_out_y = cb_y + cb_h/2
    sm_in_x = sm_x - 0.005
    sm_in_y = sm_y + sm_h/2
    mid_x = 0.022
    ax.annotate("", xy=(sm_in_x + 0.01, sm_y),
                xytext=(cb_x, cb_y + cb_h),
                arrowprops=dict(arrowstyle="-|>", color=C_CTRL_ARR, lw=2.0,
                                connectionstyle="arc3,rad=-0.55",
                                shrinkA=1, shrinkB=1),
                zorder=5)
    alabel(ax, 0.025, 0.53, "set_mode\n(stream_i,\nmode)", color=C_CTRL_ARR, fontsize=5.5)

    # 1 Hz loop annotation
    txt(ax, 0.50, 0.385, "← 1 Hz control loop →", fontsize=7, color="#666",
        fontstyle="italic", fontweight="normal")

    # ================================================================
    # 8) AIF INTERNALS (bottom panel)
    # ================================================================
    ai_x, ai_y, ai_w, ai_h = 0.035, 0.025, 0.94, 0.215
    rbox(ax, ai_x, ai_y, ai_w, ai_h, fc=C_AIF_BG, ec="#2E7D32",
         lw=1.5, ls="--", zorder=0)
    txt(ax, ai_x + ai_w/2, ai_y + ai_h - 0.015,
        "AIF Controller — Internal Architecture", fontsize=9, color="#1B5E20")

    # Boxes inside AIF
    boxes_aif = [
        (0.06,  "Belief\nQ(s)", "[P(LOW), P(MED),\n  P(HIGH)]", "#C8E6C9"),
        (0.23,  "Likelihood\nA(o|s,m)", "per-mode 3×3\nobservation model", "#B9F6CA"),
        (0.40,  "Transition\nB(s'|s,a)", "promote / demote\n/ no-op dynamics", "#B9F6CA"),
        (0.57,  "EFE\nG(π)", "pragmatic (pref.)\n+ epistemic (info.)", "#81C784"),
        (0.74,  "Softmax\nσ(−βG)", "precision β = 4.0\naction selection", "#66BB6A"),
    ]
    bw_aif, bh_aif = 0.14, 0.095
    by_aif = ai_y + 0.065
    for bx_off, label, sub, fc in boxes_aif:
        rbox(ax, bx_off, by_aif, bw_aif, bh_aif, fc=fc, ec="#1B5E20", lw=1.0)
        txt(ax, bx_off + bw_aif/2, by_aif + bh_aif/2 + 0.015, label,
            fontsize=7, color="#1B5E20")
        stxt(ax, bx_off + bw_aif/2, by_aif + bh_aif/2 - 0.020, sub, fontsize=5.5)

    # Arrows between AIF boxes
    for i in range(len(boxes_aif) - 1):
        x1 = boxes_aif[i][0] + bw_aif
        x2 = boxes_aif[i+1][0]
        cy = by_aif + bh_aif/2
        arr(ax, x1, cy, x2, cy, color="#2E7D32", lw=1.2)

    # Input / output labels
    txt(ax, 0.04, by_aif + bh_aif/2, "o(t)\n→", fontsize=7, color="#1B5E20",
        fontweight="normal")
    txt(ax, 0.90, by_aif + bh_aif/2, "→\na(t)", fontsize=7, color="#1B5E20",
        fontweight="normal")

    # Preference C
    pref_x = 0.50
    rbox(ax, pref_x, ai_y + 0.012, 0.14, 0.035, fc="#E8F5E9", ec="#66BB6A", lw=0.8)
    txt(ax, pref_x + 0.07, ai_y + 0.030 + 0.003, "Preference C",
        fontsize=6.5, color="#2E7D32")
    stxt(ax, pref_x + 0.07, ai_y + 0.030 - 0.012,
         "[0.85, 0.12, 0.03]", fontsize=5.5, family="monospace")
    arr(ax, pref_x + 0.07, ai_y + 0.047, 0.64, by_aif, color="#66BB6A", lw=0.8)

    # Anti-oscillation
    ao_x = 0.15
    rbox(ax, ao_x, ai_y + 0.012, 0.28, 0.035, fc="#E8F5E9", ec="#66BB6A", lw=0.8)
    txt(ax, ao_x + 0.14, ai_y + 0.030,
        "Anti-oscillation:  cooldown τc = 3  ·  ASAP  ·  reversal penalty",
        fontsize=5.5, fontweight="normal", color="#333")

    # ================================================================
    # 9) LEGEND
    # ================================================================
    lx, ly0 = 0.68, 0.46
    leg = [(C_DATA, "Data flow (frames / metrics)"),
           (C_CTRL_ARR, "Control flow (mode assignment)"),
           (C_NET, "Network (HTTP offload / response)")]
    for i, (c, t) in enumerate(leg):
        ly = ly0 - i * 0.028
        ax.plot([lx, lx + 0.04], [ly, ly], color=c, lw=2.5, zorder=6)
        ax.annotate("", xy=(lx + 0.04, ly), xytext=(lx + 0.035, ly),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=1.5), zorder=6)
        txt(ax, lx + 0.05, ly, t, ha="left", fontsize=6, fontweight="normal", color="#333")

    # ================================================================
    fig.savefig(os.path.join(os.path.dirname(__file__), "figures", "architecture.pdf"),
                dpi=300, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(os.path.join(os.path.dirname(__file__), "figures", "architecture.png"),
                dpi=300, bbox_inches="tight", pad_inches=0.05)
    print("Saved architecture.pdf + architecture.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
