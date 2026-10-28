"""Positions on the sky: angular separation, and which constellation contains a position (IAU
boundaries, from web/data/sky/sky.json). Used by tools/build_sky.py and tools/extract_slices.py;
web/js/skymap.js does the same in the browser."""

import json
import math
from pathlib import Path

SKY = Path(__file__).resolve().parents[1] / "web" / "data" / "sky" / "sky.json"


def separation(ra1, dec1, ra2, dec2):
    """Angle between two positions (all in degrees)."""
    r = math.radians
    c = math.sin(r(dec1)) * math.sin(r(dec2)) + math.cos(r(dec1)) * math.cos(r(dec2)) * math.cos(r(ra1 - ra2))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def unwrap(ring):
    """A boundary's vertices with longitudes made continuous (each within 180 degrees of the
    previous). A boundary around a celestial pole doesn't close that way: it's closed through the
    pole instead."""
    pts = [tuple(ring[0])]
    for lon, lat in ring[1:]:
        prev = pts[-1][0]
        pts.append((lon + 360 * round((prev - lon) / 360), lat))
    gap = pts[-1][0] - pts[0][0] if pts[-1] != pts[0] else 0
    if abs(abs(gap) - 360) < 1:  # around a pole
        pole = 90 if sum(lat for _, lat in pts) > 0 else -90
        pts += [(pts[-1][0], pole), (pts[0][0], pole)]
    return pts


def inside(ra, dec, ring):
    """Whether (ra, dec) lies in a boundary polygon of (lon, lat) vertices, by the even-odd rule
    on the unwrapped polygon, with the point's RA tried in each turn of the sky."""
    pts = unwrap(ring)
    for x in (ra - 360, ra, ra + 360):
        hits = False
        for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]):
            if (y1 > dec) != (y2 > dec) and x < x1 + (dec - y1) * (x2 - x1) / (y2 - y1):
                hits = not hits
        if hits:
            return True
    return False


def constellation_at(ra_deg, dec_deg, boundaries=None):
    """The name of the constellation containing a position. boundaries: [{"name", "rings"}], by
    default those in web/data/sky/sky.json. Raises if the position isn't in exactly one."""
    if boundaries is None:
        boundaries = json.loads(SKY.read_text())["boundaries"]
    found = [b["name"] for b in boundaries if any(inside(ra_deg, dec_deg, ring) for ring in b["rings"])]
    if len(found) != 1:
        raise ValueError(f"({ra_deg}, {dec_deg}) falls in {found or 'no constellation'}")
    return found[0]
