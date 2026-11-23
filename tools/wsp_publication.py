"""Run and check the complete frozen sessions-v1 comparison, without changing its protocol.

python -m tools.wsp_publication run --output runs/wsp-v1-attempt-001
python -m tools.wsp_publication check --folder runs/wsp-v1-attempt-001/results
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from types import SimpleNamespace

METHODS = ["random", "coordinate", "specialist", "llm", "llm-anonymized"]
SEEDS = [42, 43, 44]
PUBLICATION = Path("tracks/llm/benchmarks/wsp-v1")


def complete(folder):
    """Check coverage before the existing numerical replay; no model/API imports."""
    report = json.loads((folder / "report.json").read_text())
    if report["sessions_benchmark"] != "signal-lm-sessions-bench/v1":
        raise ValueError("Expected sessions-v1")
    if report["settings"]["methods"] != METHODS or report["settings"]["seeds"] != SEEDS:
        raise ValueError("Expected all five methods and seeds 42, 43, 44 in frozen order")
    if set(report["seeds"]) != set(map(str, SEEDS)):
        raise ValueError("Missing or extra seed results")
    for seed in SEEDS:
        methods = report["seeds"][str(seed)]["methods"]
        if set(methods) != set(METHODS):
            raise ValueError(f"seed {seed}: missing or extra method results")
        for method in METHODS:
            if method != "specialist":
                artifact = json.loads((folder / "edits" / f"seed{seed}" / f"{method}.json").read_text())
                if artifact["seed"] != seed or artifact["method"] != method:
                    raise ValueError("Edit identity mismatch")
            if method.startswith("llm"):
                transcript = json.loads((folder / "transcripts" / f"seed{seed}" / f"{method}.json").read_text())
                if transcript["seed"] != seed or transcript["method"] != method:
                    raise ValueError("Transcript identity mismatch")
                for key in ("stopped", "summary", "turns", "editor_seconds", "served_by", "cost"):
                    if transcript[key] != methods[method][key]:
                        raise ValueError(f"seed {seed} {method}: transcript/report {key} mismatch")
                if not transcript["per_turn"] or not transcript["served_by"]:
                    raise ValueError("Editor has no recorded API response")
                if not transcript["transcript"]:
                    raise ValueError("Missing editor conversation")
    return report


def check(folder):
    complete(folder)
    from tracks.llm import wsp_bench
    wsp_bench.verify("v1", folder=folder)


def run(output):
    # Reserve the directory even on preflight failure. Never overwrite or auto-retry an attempt.
    output.mkdir(parents=True, exist_ok=False)
    status = {"format": "eiathome-wsp-attempt/v1", "started": datetime.now(timezone.utc).isoformat(),
              "methods": METHODS, "seeds": SEEDS, "state": "preflight"}

    def record():
        (output / "attempt.json").write_text(json.dumps(status, indent=2) + "\n")

    record()
    try:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            status["reason"] = "ANTHROPIC_API_KEY unavailable; no arms executed"
            raise RuntimeError(status["reason"])
        from tracks.llm import wsp_bench
        if wsp_bench.protocol_sha256() != wsp_bench.PROTOCOL_SHA256:
            raise RuntimeError("Frozen protocol mismatch")
        status["protocol_sha256"] = wsp_bench.PROTOCOL_SHA256
        status["state"] = "running"
        record()
        wsp_bench.run(SimpleNamespace(benchmark="v1", methods=METHODS, seeds=SEEDS,
                                     output=output / "results", publish=False))
        status["state"] = "verifying"
        record()
        check(output / "results")
        status["state"] = "verified"
    except BaseException as error:
        status["failed_stage"] = status["state"]
        status["state"] = "incomplete"
        # Exception messages can contain request details. Record only the type.
        status["error_type"] = type(error).__name__
        raise
    finally:
        status["finished"] = datetime.now(timezone.utc).isoformat()
        record()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    execute = commands.add_parser("run")
    execute.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("check")
    verify.add_argument("--folder", type=Path, default=PUBLICATION)
    verify.add_argument("--allow-absent", action="store_true",
                        help="CI before publication: absence is pending, never a successful experiment")
    args = parser.parse_args()
    if args.command == "run":
        run(args.output)
    elif args.allow_absent and not args.folder.exists():
        print("PENDING: wsp-v1 publication absent; no experimental result claimed")
    else:
        check(args.folder)


if __name__ == "__main__":
    main()
