"""The sessions corpus: independent observing sessions for the session benchmark (BENCHMARK.md).

Every merge benchmark before it trained and tested on one 2020 Voyager session, so none of them
could show that a model generalizes to new observations. This corpus is one on-target scan of a
nearby star from each of many Green Bank observing sessions, from the pipeline directories
bldata.berkeley.edu serves over HTTPS (https://bldata.berkeley.edu/pipeline/<session>/...).

The sessions are drawn before any data is read: the pipeline index lists each session's targets;
the candidates are the sessions with a Hipparcos (HIP...) or Gliese (GJ...) target; a seeded
shuffle orders them; and each is taken in turn if its directory lists a fine-resolution scan
(`.0000.h5`) of one of those targets, until there are `--sessions` of them. The scan taken is the
first such file by name. `draw` writes the pool file `tools/extract_slices.py --pool` cuts:

    python -m tracks.llm.sessions_corpus draw --output tracks/llm/data/sessions/pool.json
    python tools/extract_slices.py --pool tracks/llm/data/sessions/pool.json \\
        --out tracks/llm/data/sessions --width 64 --product fine_resolution

The listings are parsed as text, nothing fetched is executed, and the pool records each file's
URL, so the draw and the cut can both be repeated and checked.
"""

import argparse
import html
import json
from pathlib import Path
import random
import re
import urllib.request

BASE = "https://bldata.berkeley.edu/pipeline/"
SEED = 20261107
SESSIONS = 60
STAR = re.compile(r"^(HIP\d+|GJ\d+[A-Z]?)$")
# blc00_guppi_57774_83685_LHS1140_0003.gpuspec.0000.h5, and spliced_blc..._guppi_... files.
SCAN = re.compile(r"_guppi_(\d+)_(\d+)_(.+)_(\d{4})\.(?:gpuspec|rawspec)\.0000\.h5$")


def fetch(url, timeout=60):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8", "replace")


def index(text):
    """The pipeline index's sessions and their targets: {session: [target, ...]}."""
    rows = re.findall(r'<a href="([^"/]+)/?">[^<]*</a></td><td>\s*(.*?)\s*</td>', text, re.S)
    return {html.unescape(session): targets.split() for session, targets in rows}


def links(text):
    return [html.unescape(h) for h in re.findall(r'href="([^"?]+)"', text) if not h.startswith(("..", "/", "http"))]


def star_scans(session, listing, fetch_text, max_subdirs=4):
    """Fine-resolution scans of a HIP or GJ target in a session's directory (and its first few
    subdirectories): [(url, target)], in name order within each directory."""
    found = []
    folders = [("", listing)] + [(sub, None) for sub in sorted(l for l in links(listing) if l.endswith("/"))[:max_subdirs]]
    for sub, text in folders:
        if text is None:
            text = fetch_text(f"{BASE}{session}/{sub}")
        for name in sorted(l for l in links(text) if not l.endswith("/")):
            match = SCAN.search(name)
            if match and STAR.match(match[3]):
                found.append((f"{BASE}{session}/{sub}{name}", match[3]))
    return found


def draw(sessions=SESSIONS, seed=SEED, fetch_text=fetch, log=print):
    """The pool: one scan per drawn session, in draw order."""
    table = index(fetch_text(BASE))
    candidates = sorted(s for s, targets in table.items() if any(STAR.match(t) for t in targets))
    order = random.Random(seed).sample(candidates, len(candidates))
    targets = []
    for session in order:
        if len(targets) == sessions:
            break
        try:
            scans = star_scans(session, fetch_text(f"{BASE}{session}/"), fetch_text)
        except OSError as err:
            log(f"{session}: couldn't list it ({err}); skipped")
            continue
        if not scans:
            log(f"{session}: no fine-resolution scan of a HIP or GJ target listed; skipped")
            continue
        url, target = scans[0]
        name = re.sub(r"^(HIP|GJ)", r"\1 ", target)
        targets.append({"key": session.lower().replace("_", "-"), "name": name,
                        "about": f"{name}, Green Bank session {session}: the first fine-resolution on-target scan its directory lists.",
                        "session": session, "urls": [url], "hits": 6, "blind": 2})
        log(f"{session}: {url.rsplit('/', 1)[-1]}")
    return {
        "id": "llm-sessions/v1",
        "about": (f"The sessions corpus (tracks/llm/sessions_corpus.py): {len(targets)} Green Bank observing sessions drawn at random "
                  f"(seed {seed}) from the {len(candidates)} in bldata.berkeley.edu's pipeline index with a Hipparcos or Gliese target, "
                  "one on-target fine-resolution scan each. From each scan: up to `hits` on-target hits (family stars) and `blind` "
                  "blind windows (family blind-window), 64 channels by 16 time steps, as the Live Lab's pool cuts them."),
        "seed": seed,
        "draw": {"index": BASE, "candidates": len(candidates), "sessions": sessions, "star_targets": STAR.pattern},
        "threshold": 10,
        "limits": {"max_slices": 8 * sessions, "max_bytes": 8 * sessions * 16 * 64 * 4, "max_targets": 0},
        "targets": targets,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("draw", help="draw the sessions and write the pool file")
    d.add_argument("--output", type=Path, required=True)
    d.add_argument("--sessions", type=int, default=SESSIONS)
    d.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args(argv)
    pool = draw(args.sessions, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(pool, indent=2) + "\n")
    print(f"{len(pool['targets'])} sessions written to {args.output}")


if __name__ == "__main__":
    main()
