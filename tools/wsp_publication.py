"""Run and check the complete frozen sessions-v1 comparison, without changing its protocol.

python -m tools.wsp_publication run --output runs/wsp-v1-attempt-002
python -m tools.wsp_publication check --folder runs/wsp-v1-attempt-002/results   # an attempt's results
python -m tools.wsp_publication check                                            # the publication

`check` without --folder checks the publication under tracks/llm/benchmarks/wsp-v1 and requires a
verified attempt record under tracks/llm/wsp-attempts/ naming its report.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

METHODS = ["random", "coordinate", "specialist", "llm", "llm-anonymized"]
SEEDS = [42, 43, 44]
V1 = "signal-lm-sessions-bench/v1"
ROOT = Path(__file__).resolve().parents[1]
PUBLICATION = ROOT / "tracks/llm/benchmarks/wsp-v1"
ATTEMPTS = ROOT / "tracks/llm/wsp-attempts"


def report_sha256(folder):
    return hashlib.sha256((folder / "report.json").read_bytes()).hexdigest()


def complete(folder, benchmark=V1):
    """Check coverage before the existing numerical replay; no model/API imports."""
    report = json.loads((folder / "report.json").read_text())
    if report["sessions_benchmark"] != benchmark:
        raise ValueError(f"Expected {benchmark}")
    if report["settings"]["methods"] != METHODS or report["settings"]["seeds"] != SEEDS:
        raise ValueError(f"Expected all five methods and seeds {', '.join(map(str, SEEDS))} in frozen order")
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


def attested(folder, attempts=ATTEMPTS):
    """The verified attempt record that names this report: a publication is one attempt's results."""
    digest = report_sha256(folder)
    for path in sorted(attempts.glob("*.json")):
        record = json.loads(path.read_text())
        if record.get("state") == "verified" and record.get("report_sha256") == digest:
            return path
    raise ValueError(f"No verified attempt record in {attempts} names this report (SHA-256 {digest[:12]}...)")


def check_publication(folder=PUBLICATION, attempts=ATTEMPTS):
    check(folder)
    return attested(folder, attempts)


def preflight_problem():
    """Why the frozen run would stop before any arm executes, checked first so the attempt record
    says preflight (wsp_bench.anthropic_send refuses the same things)."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return "ANTHROPIC_API_KEY unavailable; no arms executed"
    if os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return "ANTHROPIC_AUTH_TOKEN is set; the frozen editor authenticates with ANTHROPIC_API_KEY only; no arms executed"
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return "the anthropic package isn't installed; no arms executed"
    from tracks.llm import wsp_bench
    if wsp_bench.protocol_sha256() != wsp_bench.PROTOCOL_SHA256:
        return "this code's protocol isn't the frozen one; no arms executed"
    return None


def run(output):
    # Reserve the directory even on preflight failure. Never overwrite or auto-retry an attempt.
    output.mkdir(parents=True, exist_ok=False)
    status = {"format": "eiathome-wsp-attempt/v1", "started": datetime.now(timezone.utc).isoformat(),
              "methods": METHODS, "seeds": SEEDS, "state": "preflight"}

    def record():
        (output / "attempt.json").write_text(json.dumps(status, indent=2) + "\n")

    record()
    try:
        problem = preflight_problem()
        if problem:
            status["reason"] = problem
            raise RuntimeError(problem)
        from tracks.llm import wsp_bench
        status["protocol_sha256"] = wsp_bench.PROTOCOL_SHA256
        status["state"] = "running"
        record()
        wsp_bench.run(SimpleNamespace(benchmark="v1", methods=METHODS, seeds=SEEDS,
                                     output=output / "results", publish=False))
        status["state"] = "verifying"
        record()
        check(output / "results")
        status["report_sha256"] = report_sha256(output / "results")
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    execute = commands.add_parser("run")
    execute.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("check")
    verify.add_argument("--folder", type=Path, default=None,
                        help="an attempt's results; default: the publication, which also needs its verified attempt record")
    verify.add_argument("--allow-absent", action="store_true",
                        help="CI before publication: absence is pending, never a successful experiment")
    args = parser.parse_args()
    if args.command == "run":
        run(args.output)
    elif args.folder is not None:
        check(args.folder)
    elif args.allow_absent and not PUBLICATION.exists():
        print("PENDING: wsp-v1 publication absent; no experimental result claimed")
    else:
        print(f"verified, and published from {check_publication().relative_to(ROOT)}")


if __name__ == "__main__":
    main()
