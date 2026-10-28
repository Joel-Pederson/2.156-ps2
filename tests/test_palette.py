"""The district colours stay distinguishable (docs/decisions.md, 25 October, colours).

Every pair of district colours is compared, because any two districts can sit side by side.
Colour difference is ΔE: the distance between two colours in OKLab, a colour space built so
that equal distances look about equally different (scaled ×100). Red-green colour blindness
is simulated with the Machado, Oliveira and Fernandes (2009) model at full severity, applied
in linear RGB, for both types (protan and deutan). Contrast is the WCAG contrast ratio
against the page. These are the calculations of the palette validator used to choose the
colours, so the numbers quoted in docs/decisions.md can be reproduced here.
"""

import itertools

import numpy as np

import plots

# Machado, Oliveira & Fernandes (2009), severity 1.0, applied to linear RGB.
MACHADO = {
    "protan": [
        [0.152286, 1.052583, -0.204868],
        [0.114503, 0.786281, 0.099216],
        [-0.003882, -0.048116, 1.051998],
    ],
    "deutan": [
        [0.367322, 0.860646, -0.227968],
        [0.280085, 0.672501, 0.047413],
        [-0.011820, 0.042940, 0.968881],
    ],
}
# Linear RGB -> LMS, and cube-rooted LMS -> OKLab (Björn Ottosson's published matrices).
TO_LMS = [
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
]
TO_OKLAB = [
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.8086757660],
]
PAIRS = list(itertools.combinations(plots.DISTRICT_COLOURS, 2))


def _linear(hex_colour):
    """'#rrggbb' -> linear RGB, each 0..1 (undoing the sRGB gamma)."""
    c = np.array([int(hex_colour[i : i + 2], 16) / 255 for i in (1, 3, 5)])
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _oklab(rgb):
    return np.array(TO_OKLAB) @ np.cbrt(np.array(TO_LMS) @ rgb)


def _delta_e(c1, c2, kind=None):
    """ΔE ×100 between two colours, as seen with normal vision or simulated colour blindness."""

    def seen(c):
        rgb = _linear(c)
        if kind:
            rgb = np.clip(np.array(MACHADO[kind]) @ rgb, 0, 1)
        return _oklab(rgb)

    return 100 * float(np.linalg.norm(seen(c1) - seen(c2)))


def _contrast(c1, c2):
    """WCAG contrast ratio: (lighter + 0.05) / (darker + 0.05) of relative luminance."""
    lum = sorted(
        float(np.array([0.2126, 0.7152, 0.0722]) @ _linear(c)) for c in (c1, c2)
    )
    return (lum[1] + 0.05) / (lum[0] + 0.05)


def test_every_pair_of_district_colours_differs_under_normal_vision():
    worst = min(_delta_e(a, b) for a, b in PAIRS)
    assert worst >= 15  # the validator's minimum
    assert round(worst, 1) == 18.0  # quoted in docs/decisions.md


def test_every_pair_differs_under_simulated_red_green_colour_blindness():
    worst = min(_delta_e(a, b, kind) for a, b in PAIRS for kind in MACHADO)
    assert worst >= 8  # the validator's target
    assert round(worst, 1) == 13.3  # quoted in docs/decisions.md


def test_only_pink_and_amber_fall_below_3_to_1_contrast():
    # Hence the district letter on every cell.
    low = {c for c in plots.DISTRICT_COLOURS if _contrast(c, plots.SURFACE) < 3}
    assert low == {"#f48cae", "#eba002"}
