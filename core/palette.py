"""Shared figure palette .

One chromatic order for every record, and the same order everywhere it is listed:
  * colour: the hue sweeps violet -> blue -> teal -> green -> yellow -> orange -> red,
  * lightness: rises monotonically along the same sequence, from the darkest record to the brightest.
So the list DATASET_ORDER is simultaneously the hue order and the dark-to-bright order.
Sequential ramps (replicate counts, branching ratios, sizes) run dark -> bright with increasing
value. The two mechanisms keep their own hues (NODE, TIE). Old names stay as aliases.
"""
import os


# mechanism hues (schematic, confound, spreading and dynamics figures)
NODE = "#3F6FA0"        # node self-excitation: steel blue
NODE_LIGHT = "#A9C1DA"
TIE = "#8B3A6B"         # tie reinforcement: plum
TIE_LIGHT = "#D7B3C8"
POISSON = "#8C8C8C"     # memoryless reference
INK = "#2B2B2B"         # axes, text
REF = "#9A9A9A"         # reference lines (truth, nulls)

# chromatic order, darkest first; (name, family, hex, marker, short label)
_DATA = [
    ("SNN (bursting)",            "neural",        "#37005B", "v", "Spiking network"),
    ("Cortex",                    "neural",        "#440F83", "h", "Cortex"),
    ("Stack Overflow",            "communication", "#1B3EA5", "*", "Stack Overflow"),
    ("Email (EU institution)",    "communication", "#005BA6", "D", "Email"),
    ("CollegeMsg (Direct Msgs)",  "communication", "#0074A7", "<", "CollegeMsg"),
    ("Wikibooks",                 "communication", "#008AAE", "p", "Wikibooks"),
    ("Wikibooks bots",            "algorithmic",   "#00A599", "X", "Wikibooks bots"),
    ("Microservices",             "algorithmic",   "#5BBB52", "d", "Microservices"),
    ("Baboons",                   "proximity",     "#C3B500", "P", "Baboons"),
    ("Conference (SFHH)",         "proximity",     "#EFB301", "^", "Conference"),
    ("High school",               "proximity",     "#FFB987", "s", "High school"),
    ("Hospital ward",             "proximity",     "#FFC3BB", "o", "Hospital"),
]
_ALIAS = {
    "Hospital": "Hospital ward", "School": "High school", "Conference": "Conference (SFHH)",
    "Email": "Email (EU institution)", "Email (EU)": "Email (EU institution)",
    "CollegeMsg": "CollegeMsg (Direct Msgs)", "Wikibooks (bipartite)": "Wikibooks",
    "Wikibooks (Bots)": "Wikibooks bots", "Alibaba micro": "Microservices",
    "Alibaba microservices": "Microservices",
    "SNN": "SNN (bursting)", "SNN (refractory)": "SNN (bursting)",
    "Brain": "Cortex", "Allen Brain": "Cortex", "Allen Brain Network": "Cortex",
    "Cortex (Allen)": "Cortex", "Wikibooks bots (rebuilt)": "Wikibooks bots",
}
_BY_NAME = {d[0]: d for d in _DATA}

# the chromatic-order list: darkest first, hue order, every record once
DATASET_ORDER = [d[0] for d in _DATA]


def key(name):
    return _ALIAS.get(name, name)


def family(name):
    return _BY_NAME[key(name)][1]


def label(name):
    return _BY_NAME[key(name)][4]


COLORS = {d[0]: d[2] for d in _DATA}
MARKERS = {d[0]: d[3] for d in _DATA}
COLORS.update({a: COLORS[k] for a, k in _ALIAS.items()})
MARKERS.update({a: MARKERS[k] for a, k in _ALIAS.items()})

# families named in the text, with a neutral heading colour (family is no longer a colour code)
FAMILY_LABEL = {"proximity": "proximity", "communication": "communication",
                "algorithmic": "algorithmic", "neural": "neural"}
FAMILY = {f: INK for f in FAMILY_LABEL}          # legacy name: family headings are neutral ink


def style(name, size=None):
    """Keyword arguments for ax.scatter markers of one record: fill, dark hairline edge, size."""
    k = key(name)
    big = {"*": 1.6, "P": 1.15, "X": 1.15, "p": 1.15, "h": 1.15}.get(MARKERS[k], 1.0)
    return dict(color=COLORS[k], marker=MARKERS[k], edgecolors=INK, linewidths=0.35,
                s=(size or 30) * big, zorder=4)


def ramp(n, lo=0.20, hi=0.85, hue=255, chroma=0.10):
    """n colours from dark (index 0) to bright (index n-1), one hue, lightness rising in OKLab."""
    import numpy as np
    out = []
    for i in range(n):
        L = lo + (hi - lo) * (i / max(n - 1, 1))
        out.append(_oklch_hex(L, chroma, hue))
    return out


def _oklch_hex(L, C, h):
    import numpy as np
    a = C * np.cos(np.radians(h)); b = C * np.sin(np.radians(h))
    l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    rgb = np.array([4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
                    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
                    -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s])
    rgb = np.clip(rgb, 0, 1)
    enc = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * np.power(rgb, 1 / 2.4) - 0.055)
    return "#%02X%02X%02X" % tuple(int(round(v * 255)) for v in np.clip(enc, 0, 1))


def bottom_legend(fig, ax, **kw):
    """Single frameless legend centred below the figure (kept for the supplementary scripts)."""
    h, l = ax.get_legend_handles_labels()
    if not h:
        return
    ncol = (len(h) + 1) // 2
    kw.setdefault("fontsize", 6.5)
    kw.setdefault("handletextpad", 0.4)
    kw.setdefault("columnspacing", 1.2)
    fig.legend(h, l, loc="lower center", ncol=ncol, frameon=False,
               bbox_to_anchor=(0.5, -0.05), **kw)
