"""Digitize Fig. 6 of Ge-Chong-Stone, PRA 82, 063824 (arXiv:1008.0628).

Regenerates ``data/ge_fig6_digitized.csv``, the reference data that
``compare_to_pra_fig6.py`` overlays on the netsalt solvers. Open symbols in the
figure are the exact SALT (numerical solution of the paper's Eq. 28), solid
lines the single-pole approximation (Eqs. 40, 44-45): red squares = dominant
mode, blue circles = second mode, black triangles = total.

Method: download the arXiv PDF (if not already next to this script), render the
Fig. 6 region at 600 dpi with ``pdftoppm`` (poppler), color-separate the
markers from the lines, and calibrate the axes from the tick-label text
centroids (MATLAB centers labels on ticks; calibration residual ~0.3%).
Isolated blobs of a color are markers; the long thin component is the SPA line
-- markers riding *on* the line (single-mode regime) merge into it, so there
the ``spa_*`` series carries small bumps and the exact symbols are only
recovered where they separate from the line. The crop box is pinned to the v1
arXiv PDF (Fig. 6 is on page 12).

Requires: poppler (``pdftoppm``, ``pdftotext``), scipy, matplotlib.
"""

import subprocess
import urllib.request
from pathlib import Path

import numpy as np
from matplotlib.image import imread
from scipy import ndimage

HERE = Path(__file__).resolve().parent
PDF = HERE / "data" / "arxiv_1008.0628.pdf"
OUT = HERE / "data" / "ge_fig6_digitized.csv"
ARXIV_URL = "https://arxiv.org/pdf/1008.0628"
# Fig. 6 crop on its page, in 600-dpi pixels (pinned to the v1 arXiv PDF)
CROP = {"x": 2900, "y": 300, "W": 2200, "H": 1500}


def fig6_page(pdf):
    """Page number holding the Fig. 6 caption."""
    txt = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True, text=True).stdout
    for i, page in enumerate(txt.split("\f"), 1):
        if "FIG. 6" in page:
            return i
    raise RuntimeError("FIG. 6 not found in PDF text")


def render(pdf, page, png_prefix):
    subprocess.run(
        ["pdftoppm", "-png", "-r", "600", "-f", str(page), "-l", str(page)]
        + ["-x", str(CROP["x"]), "-y", str(CROP["y"]), "-W", str(CROP["W"]), "-H", str(CROP["H"])]
        + [str(pdf), str(png_prefix)],
        check=True,
    )
    (out,) = png_prefix.parent.glob(png_prefix.name + "*.png")
    return out


def blob_centroids(mask, min_pix=40):
    lab, n = ndimage.label(mask)
    out = []
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        if len(xs) >= min_pix:
            out.append((xs.mean(), ys.mean()))
    return out


def cluster_1d(vals, gap):
    vals = sorted(vals)
    groups, cur = [], [vals[0]]
    for v in vals[1:]:
        if v - cur[-1] <= gap:
            cur.append(v)
        else:
            groups.append(float(np.mean(cur)))
            cur = [v]
    groups.append(float(np.mean(cur)))
    return groups


def digitize(png):
    img = imread(png)[..., :3]
    H, _W = img.shape[:2]
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    red = (r > 0.6) & (g < 0.35) & (b < 0.35)
    blue = (b > 0.6) & (r < 0.35) & (g < 0.35)
    black = (r < 0.25) & (g < 0.25) & (b < 0.25)

    # plot frame = longest black vertical/horizontal runs
    vert = np.where(black.sum(0) > 0.5 * H)[0]
    x_left, x_right = int(vert.min()), int(vert.max())
    horiz = np.where(black[:, x_left:x_right].sum(1) > 0.4 * (x_right - x_left))[0]
    y_top, y_bot = int(horiz.min()), int(horiz.max())

    # axis calibration from the tick-label text centroids
    # label band only (the axis title sits lower and must not join the clusters)
    xs_lab = cluster_1d([c[0] for c in blob_centroids(black[y_bot + 5 : y_bot + 90, :])], gap=60)
    assert len(xs_lab) == 4, f"expected x labels 0.6 0.8 1.0 1.2, got {len(xs_lab)}"
    cx = np.polyfit(xs_lab, [0.6, 0.8, 1.0, 1.2], 1)
    left = black[:, max(0, x_left - 130) : x_left - 8]
    ys_lab = cluster_1d([c[1] for c in blob_centroids(left) if c[1] < y_bot + 40], gap=40)
    assert len(ys_lab) == 4, f"expected y labels 0.3 0.2 0.1 0, got {len(ys_lab)}"
    cy = np.polyfit(ys_lab, [0.3, 0.2, 0.1, 0.0], 1)

    inside = np.zeros_like(red)
    inside[y_top + 3 : y_bot - 2, x_left + 3 : x_right - 2] = True

    def extract(mask):
        lab, n = ndimage.label(mask & inside)
        markers, line_pts = [], []
        for i in range(1, n + 1):
            ys, xs = np.where(lab == i)
            w, h = np.ptp(xs) + 1, np.ptp(ys) + 1
            if w > 120:  # the SPA line (with any markers riding on it merged in)
                line_pts.extend((c, np.median(ys[xs == c])) for c in np.unique(xs))
            elif 12 < w < 90 and 12 < h < 90:  # an isolated open marker
                markers.append((xs.mean(), ys.mean()))
        return np.array(markers).reshape(-1, 2), np.array(line_pts).reshape(-1, 2)

    series = {}
    for _color, mask, exact, spa in [
        ("red", red, "exact_dominant", "spa_dominant"),
        ("blue", blue, "exact_second", "spa_second"),
        ("black", black, "exact_total", "exact_total"),
    ]:
        markers, line = extract(mask)
        for name, pts in [(exact, markers), (spa, line)]:
            if not len(pts):
                continue
            d = np.column_stack([np.polyval(cx, pts[:, 0]), np.polyval(cy, pts[:, 1])])
            d = d[np.argsort(d[:, 0])]
            if name.startswith("spa") or name == "exact_total":
                d = d[:: max(1, len(d) // 80)]  # thin the per-column line samples
            series.setdefault(name, []).append(d)
    return {k: np.vstack(v) for k, v in series.items()}


def main():
    PDF.parent.mkdir(exist_ok=True)
    if not PDF.exists():
        print(f"downloading {ARXIV_URL} ...")
        urllib.request.urlretrieve(ARXIV_URL, PDF)
    page = fig6_page(PDF)
    png = render(PDF, page, HERE / "data" / "ge_fig6")
    series = digitize(png)

    with open(OUT, "w") as f:
        f.write("# Ge-Chong-Stone PRA 82, 063824 (arXiv:1008.0628) Fig. 6, digitized\n")
        f.write("# exact_* : symbols (numerical solution of Eq. 28, exact SALT)\n")
        f.write("# spa_*   : solid lines (single-pole approximation, Eqs. 40, 44-45)\n")
        f.write("# NB: in the single-mode regime the symbols ride on the lines, so the\n")
        f.write("# spa_* series carry small bumps where merged markers were not separable.\n")
        f.write("series,D0,intensity\n")
        for name in sorted(series):
            for D0, inten in series[name]:
                f.write(f"{name},{D0:.4f},{inten:.4f}\n")
    print(f"wrote {OUT} ({sum(len(v) for v in series.values())} rows)")


if __name__ == "__main__":
    main()
