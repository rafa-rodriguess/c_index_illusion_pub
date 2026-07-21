"""SciencePlots matplotlib setup (artefatos.md §0)."""

from __future__ import annotations

from pathlib import Path

# Maximum practical export DPI (vector PDF + any rasterized artists / viewer previews)
EXPORT_DPI = 1200
# Physical size multiplier so IDE/PDF raster previews stay sharp; LaTeX scales down.
SIZE_SCALE = 2.0


def figsize(width_in: float, height_in: float) -> tuple[float, float]:
    """Journal logical inches × SIZE_SCALE for max-res export."""
    return (width_in * SIZE_SCALE, height_in * SIZE_SCALE)


def apply_paper_style() -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import scienceplots  # noqa: F401 — registers styles

    plt.style.use(["science", "no-latex"])
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "figure.dpi": EXPORT_DPI,
            "savefig.dpi": EXPORT_DPI,
            "savefig.facecolor": "white",
            "savefig.transparent": False,
            "savefig.bbox": "tight",
            "axes.grid": False,
            # Crisp vector paths / embedded fonts (no Type-3 bitmaps)
            "pdf.fonttype": 42,  # TrueType
            "ps.fonttype": 42,
            "pdf.compression": 0,
            "path.simplify": False,
            "agg.path.chunksize": 0,
            # Thicker strokes / larger type at scaled canvas
            "lines.linewidth": 1.2,
            "axes.linewidth": 0.8,
            "font.size": 10,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )


def save_pdf(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        format="pdf",
        dpi=EXPORT_DPI,
        bbox_inches="tight",
        pad_inches=0.02,
        metadata={"Creator": "cross-domain-survival Block P"},
    )
    return path
