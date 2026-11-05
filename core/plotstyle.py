#!/usr/bin/env python3
"""Shared publication style for all paper figures (Physical Review Research, APS).

APS widths: one column 8.6 cm, two columns 17.8 cm. Figures are built at their final printed width
so text renders at the intended size: 8 pt bold panel letters, 7 pt labels, 6.5 pt ticks and
annotations, Arial, vector PDF with embedded text. Call apply() at the top of a figure script,
panel(ax, "a") for panel letters, and save(fig, stem) to write PDF + PNG.
"""
import matplotlib

MM = 1.0 / 25.4
COL = 86 * MM      # one column  (inches) -> \includegraphics[width=\columnwidth]
FULL = 178 * MM    # two columns (inches) -> figure* with width=\textwidth
ICON = 30          # default scatter marker area for dataset icons


def apply():
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
        "font.size": 7.0,
        "axes.titlesize": 7.0,
        "axes.titleweight": "normal",
        "axes.titlepad": 4.0,
        "axes.labelsize": 7.0,
        "axes.labelpad": 2.5,
        "axes.labelcolor": "#2B2B2B",
        "axes.edgecolor": "#2B2B2B",
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "xtick.color": "#2B2B2B",
        "ytick.color": "#2B2B2B",
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.major.pad": 2.0,
        "ytick.major.pad": 2.0,
        "legend.fontsize": 6.5,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.borderaxespad": 0.3,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.2,
        "lines.markersize": 4.0,
        "errorbar.capsize": 0,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def panel(ax, letter, x=-0.16, y=1.06):
    """Bold lower-case panel letter at a fixed offset from the axes' top-left corner."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=8, fontweight="bold",
            va="bottom", ha="left", color="#2B2B2B")


def save(fig, stem):
    """Write <stem>.pdf (vector, for submission) and <stem>.png (for visual QC)."""
    fig.savefig(stem + ".pdf")
    fig.savefig(stem + ".png", dpi=300)
