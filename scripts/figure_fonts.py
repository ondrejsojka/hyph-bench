"""Typeset matplotlib figures in the paper's fonts: TeX Gyre Termes and Cursor.

The paper is set in TeX Gyre Termes (text) and TeX Gyre Cursor (monospace).
TeX Gyre ships as CFF-flavoured OpenType, which matplotlib cannot embed as
Type 42; embedding it directly yields either Type 3 fonts or a PDF whose font
type disagrees with the embedded program. The outlines are therefore converted
once to TrueType (cubic to quadratic) and cached, after which matplotlib embeds
them as ordinary CID TrueType subsets with extractable text.

TeX Gyre is distributed under the GUST Font License; the converted files are a
local build cache and only glyph subsets leave this machine, inside the PDFs.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

FAMILIES = ("texgyretermes", "texgyrecursor")
CACHE_DIR = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "pat-gen-opt" / "fonts"

# Where TeX distributions keep the OpenType files.
OTF_SEARCH_GLOBS = (
    "/usr/share/texmf/fonts/opentype/public/tex-gyre/*.otf",  # Debian fonts-texgyre
    "/usr/share/texlive/texmf-dist/fonts/opentype/public/tex-gyre/*.otf",
    "/usr/local/texlive/*/texmf-dist/fonts/opentype/public/tex-gyre/*.otf",
    str(Path.home() / ".cache/Tectonic/bundles/data/*/texgyre*.otf"),
)


def find_otf_files() -> dict[str, Path]:
    found: dict[str, Path] = {}
    for pattern in OTF_SEARCH_GLOBS:
        for path in glob.glob(pattern):
            name = Path(path).name
            if name.startswith(FAMILIES) and name not in found:
                found[name] = Path(path)
    return found


def otf_to_ttf(src: Path, dst: Path) -> None:
    from fontTools.pens.cu2quPen import Cu2QuPen
    from fontTools.pens.ttGlyphPen import TTGlyphPen
    from fontTools.ttLib import TTFont, newTable

    font = TTFont(str(src))
    glyph_set = font.getGlyphSet()
    order = font.getGlyphOrder()
    glyf = newTable("glyf")
    glyf.glyphs = {}
    glyf.glyphOrder = order
    for name in order:
        pen = TTGlyphPen(glyph_set)
        glyph_set[name].draw(Cu2QuPen(pen, 1.0, reverse_direction=True))
        glyf.glyphs[name] = pen.glyph()
    font["glyf"] = glyf
    font["loca"] = newTable("loca")
    del font["CFF "]
    maxp = font["maxp"]
    maxp.tableVersion = 0x00010000
    for field in (
        "maxPoints", "maxContours", "maxCompositePoints", "maxCompositeContours",
        "maxTwilightPoints", "maxStorage", "maxFunctionDefs", "maxInstructionDefs",
        "maxStackElements", "maxSizeOfInstructions", "maxComponentElements", "maxComponentDepth",
    ):
        setattr(maxp, field, 0)
    maxp.maxZones = 1
    post = font["post"]
    post.formatType = 2.0
    post.extraNames = []
    post.mapping = {}
    font.sfntVersion = "\x00\x01\x00\x00"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".ttf.tmp")
    font.save(str(tmp))
    tmp.replace(dst)


def register_paper_fonts() -> bool:
    """Make TeX Gyre Termes/Cursor available to matplotlib; False if not found."""
    otfs = find_otf_files()
    if not otfs:
        return False
    for name, src in sorted(otfs.items()):
        ttf = CACHE_DIR / (Path(name).stem + ".ttf")
        if not ttf.is_file() or ttf.stat().st_mtime < src.stat().st_mtime:
            otf_to_ttf(src, ttf)
        fm.fontManager.addfont(str(ttf))
    return True


def use_paper_fonts() -> None:
    """Point matplotlib at the paper's fonts, with DejaVu as glyph fallback.

    Text: TeX Gyre Termes. Monospace: TeX Gyre Cursor. Math: STIX, a Times-
    compatible design (Termes has no math companion in TrueType form). DejaVu
    Sans stays in the family list only for glyphs Termes lacks, such as the
    star marker.
    """
    if register_paper_fonts():
        plt.rcParams["font.family"] = ["TeX Gyre Termes", "DejaVu Sans"]
        plt.rcParams["font.monospace"] = ["TeX Gyre Cursor", "DejaVu Sans Mono"]
    else:
        plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"
