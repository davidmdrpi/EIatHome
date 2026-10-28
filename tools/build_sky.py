#!/usr/bin/env python3
"""Builds web/data/sky/sky.json, the Live Lab's star map data, from d3-celestial's data files.

d3-celestial (Olaf Frohn, BSD-3-Clause) packages the Hipparcos catalogue (ESA 1997) as GeoJSON,
with constellation stick figures, names and IAU boundaries. This script takes, from version
0.7.35 on npm (checked against its published SHA-512):

- the 1,500 brightest stars, with right ascension, declination, V magnitude and B-V colour;
- every Hipparcos star near the observed targets, so that zooming in shows the real field;
- the constellation lines, and a label position and name for each constellation;
- the proper names of the brightest stars;
- the IAU constellation boundaries, and reference targets with the constellation they're in.

    python tools/build_sky.py            # download the package and (re)write the data
    python tools/build_sky.py --check    # rebuild and compare with the committed file

The license notice that the BSD license asks for is written next to the data.
"""

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
from skycoords import constellation_at, inside, separation  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web" / "data" / "sky"
PACKAGE = "https://registry.npmjs.org/d3-celestial/-/d3-celestial-0.7.35.tgz"
INTEGRITY = "sha512-cURxIl0E+FGWnYj6gTDt80SjuiM9lklcGykj/skVy7glDg5nj/QxTUoPPArU+bpEQ+1fLy5hi920OvJ/TgliRw=="
BRIGHT = 1500
NAMED_BRIGHTER_THAN = 1.6  # proper names shown for stars brighter than this

# Fields whose Hipparcos stars are all kept, for the zoomed views: Voyager 1 and the off-source
# positions of its 2020 cadence (1 degree north, east and south), and around each reference.
VOYAGER_FIELD = {"name": "Voyager 1 and its off-source positions", "ra_deg": 258.12, "dec_deg": 12.35, "radius_deg": 4.0}
REFERENCE_RADIUS = 3.0

# Reference targets: J2000 positions (SIMBAD), distances from the Sun.
REFERENCES = [
    {"id": "proxima", "label": "Proxima Centauri", "ra_hours": 14.495263, "dec_deg": -62.679489,
     "distance_label": "4.24 light-years",
     "note": "The nearest star. BLC1, a 2019 candidate signal seen toward it with the Parkes telescope, turned out to be interference."},
    {"id": "barnard", "label": "Barnard's Star", "ra_hours": 17.963471, "dec_deg": 4.693364,
     "distance_label": "5.96 light-years", "note": "One of the nearest stars, in Ophiuchus like Voyager 1."},
    {"id": "tau-ceti", "label": "Tau Ceti", "ra_hours": 1.734468, "dec_deg": -15.937481,
     "distance_label": "11.9 light-years", "note": "A Sun-like star; one of the two targets of Project Ozma, the first radio SETI search (1960)."},
    {"id": "trappist-1", "label": "TRAPPIST-1", "ra_hours": 23.108158, "dec_deg": -5.041400,
     "distance_label": "40.7 light-years", "note": "Seven Earth-sized planets; searched by Breakthrough Listen with the Green Bank Telescope."},
    {"id": "galactic-centre", "label": "Galactic Centre", "ra_hours": 17.761121, "dec_deg": -29.007825,
     "distance_label": "about 26,700 light-years", "note": "Sagittarius A*; Breakthrough Listen surveyed the region with the Green Bank Telescope."},
]


def download():
    with urllib.request.urlopen(PACKAGE, timeout=120) as response:
        data = response.read()
    algorithm, expected = INTEGRITY.split("-", 1)
    if base64.b64encode(hashlib.new(algorithm, data).digest()).decode() != expected:
        raise SystemExit(f"{PACKAGE} doesn't match its published integrity hash")
    return tarfile.open(fileobj=io.BytesIO(data), mode="r:gz")


def member(archive, name):
    return archive.extractfile(f"package/{name}").read()


def polygons(geometry):
    if geometry["type"] == "Polygon":
        return [geometry["coordinates"][0]]
    return [p[0] for p in geometry["coordinates"]]


def star(feature):
    lon, lat = feature["geometry"]["coordinates"]
    bv = feature["properties"].get("bv")
    return [round(lon % 360, 3), round(lat, 3), round(float(feature["properties"]["mag"]), 2),
            round(float(bv), 2) if bv not in (None, "") else None]


def build(archive):
    load = lambda name: json.loads(member(archive, f"data/{name}"))
    bright = sorted(load("stars.6.json")["features"], key=lambda f: (f["properties"]["mag"], f["id"]))[:BRIGHT]
    bright_ids = {f["id"] for f in bright}
    hipparcos = load("stars.14.json")["features"]
    # One entry per constellation, except Serpens: its two parts share the id "Ser" in every file.
    # A boundary or a set of lines with a shared id goes to the part whose label it contains or is
    # nearest to.
    entries = [{"id": f["id"], "name": f["properties"]["name"], "rank": int(f["properties"].get("rank", 3)),
                "label": [f["geometry"]["coordinates"][0] % 360, f["geometry"]["coordinates"][1]]}
               for f in load("constellations.json")["features"]]

    def part_of(cid, fits):
        candidates = [e for e in entries if e["id"] == cid]
        return candidates[0] if len(candidates) == 1 else max(candidates, key=fits)

    boundaries = []
    for f in load("constellations.bounds.json")["features"]:
        rings = [[[round(lon % 360, 3), round(lat, 3)] for lon, lat in ring] for ring in polygons(f["geometry"])]
        part = part_of(f["id"], lambda e: any(inside(*e["label"], ring) for ring in rings))
        boundaries.append({"id": f["id"], "name": part["name"], "rings": rings})
    boundaries.sort(key=lambda b: (b["id"], b["name"]))

    fields = [VOYAGER_FIELD] + [{"name": r["label"], "ra_deg": round(r["ra_hours"] * 15, 4), "dec_deg": r["dec_deg"],
                                  "radius_deg": REFERENCE_RADIUS} for r in REFERENCES]
    for field in fields:
        near = [f for f in hipparcos if f["id"] not in bright_ids
                and separation(f["geometry"]["coordinates"][0] % 360, f["geometry"]["coordinates"][1], field["ra_deg"], field["dec_deg"]) <= field["radius_deg"]]
        field["stars"] = [star(f) for f in sorted(near, key=lambda f: (f["properties"]["mag"], f["id"]))]

    for e in entries:
        e["lines"] = []
    for f in load("constellations.lines.json")["features"]:
        vertices = [v for line in f["geometry"]["coordinates"] for v in line]
        closeness = lambda e: -sum(separation(lon % 360, lat, *e["label"]) for lon, lat in vertices)
        part_of(f["id"], closeness)["lines"] += [[[round(lon % 360, 2), round(lat, 2)] for lon, lat in line] for line in f["geometry"]["coordinates"]]
    constellations = [{"id": e["id"], "name": e["name"], "rank": e["rank"], "label": [round(e["label"][0], 2), round(e["label"][1], 2)], "lines": e["lines"]}
                      for e in sorted(entries, key=lambda e: (e["id"], e["name"]))]

    starnames = load("starnames.json")
    by_id = {f["id"]: f for f in load("stars.6.json")["features"]}
    named = []
    for hip, entry in starnames.items():
        f = by_id.get(int(hip)) if hip.isdigit() else None
        if f and entry.get("name") and f["properties"]["mag"] < NAMED_BRIGHTER_THAN:
            ra, dec, mag, _ = star(f)
            named.append([ra, dec, mag, entry["name"]])
    named.sort(key=lambda s: s[2])

    references = [{**r, "constellation": constellation_at(r["ra_hours"] * 15, r["dec_deg"], boundaries), "epoch": "J2000"} for r in REFERENCES]
    return {
        "format": "eiathome-sky", "version": 1,
        "source": "Hipparcos catalogue (ESA 1997), constellation lines, names and IAU boundaries, from d3-celestial 0.7.35 "
                  "(Olaf Frohn, BSD-3-Clause: see LICENSE.d3-celestial.txt). Built by tools/build_sky.py.",
        "columns": ["ra_deg", "dec_deg", "v_mag", "b_v"],
        "stars": [star(f) for f in bright],
        "fields": fields,
        "constellations": constellations,
        "names": named,
        "boundaries": boundaries,
        "references": references,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true", help="compare a rebuild with the committed files instead of writing them")
    args = p.parse_args(argv)
    archive = download()
    text = json.dumps(build(archive), separators=(",", ":")) + "\n"
    notice = member(archive, "LICENSE")
    if args.check:
        same = (OUT / "sky.json").read_text() == text and (OUT / "LICENSE.d3-celestial.txt").read_bytes() == notice
        raise SystemExit(0 if same else "The committed sky data doesn't match a rebuild")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sky.json").write_text(text)
    (OUT / "LICENSE.d3-celestial.txt").write_bytes(notice)
    print(f"Wrote {OUT / 'sky.json'} ({len(text):,} bytes)")


if __name__ == "__main__":
    main()
