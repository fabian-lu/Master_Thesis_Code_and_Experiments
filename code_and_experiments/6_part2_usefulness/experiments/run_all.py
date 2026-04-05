#!/usr/bin/env python3
"""
Run all 5 experiment notebooks sequentially with retry logic.

Usage:
    # Test run (N=2, quick check that everything works)
    python run_all.py --n 2

    # Full experiment run
    python run_all.py --n 60

    # Run inside tmux so it survives terminal disconnect:
    tmux new -s experiments
    python run_all.py --n 60
    # Then Ctrl+B, D to detach. Reattach with: tmux attach -t experiments

    # Or use nohup:
    nohup python run_all.py --n 60 > run_all_output.log 2>&1 &

    # Monitor progress (from another terminal):
    python run_all.py --status
    # Or just: cat run_status.json | python -m json.tool
    # Or tail the log: tail -f run_all.log
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"

NOTEBOOKS = [
    {
        "name": "E1",
        "file": "E1_closeness_task.ipynb",
        "valid_col": "predicted_bucket",
        "valid_values": {"small", "medium", "large", "very_large"},
        "expected_rows": lambda n: 12 * n,
    },
    {
        "name": "E2",
        "file": "E2_anomaly_detection.ipynb",
        "valid_col": "reliability",
        "valid_values": {"reliable", "unreliable"},
        "expected_rows": lambda n: 12 * n,
    },
    {
        "name": "E3",
        "file": "E3_counterfactual_simulatability.ipynb",
        "valid_col": "predicted_direction",
        "valid_values": {"higher", "lower", "similar"},
        "expected_rows": lambda n: 6 * n,
    },
    {
        "name": "E4",
        "file": "E4_mental_model_transfer.ipynb",
        "valid_col": "predicted_bucket",
        "valid_values": {"small", "medium", "large", "very_large"},
        "expected_rows": lambda n: 6 * min(n, 55),
    },
    {
        "name": "E5",
        "file": "E5_placebic_nle_control.ipynb",
        "valid_col": "predicted_bucket",
        "valid_values": {"small", "medium", "large", "very_large"},
        "expected_rows": lambda n: 10 * n,
    },
]

MAX_RETRIES = 5           # max retry attempts per notebook
WAIT_BETWEEN_NB = 30      # seconds between notebooks
WAIT_BETWEEN_RETRY = 60   # seconds between retries of the same notebook
NOTEBOOK_TIMEOUT = 7200   # 2 hours max per notebook execution

LOG_FILE = SCRIPT_DIR / "run_all.log"
STATUS_FILE = SCRIPT_DIR / "run_status.json"


# ── Logging ─────────────────────────────────────────────────────────

def log(msg: str):
    """Print and append to log file."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


# ── Status tracking ─────────────────────────────────────────────────

def load_status() -> dict:
    if STATUS_FILE.exists():
        return json.loads(STATUS_FILE.read_text())
    return {}


def save_status(status: dict):
    STATUS_FILE.write_text(json.dumps(status, indent=2, default=str))


def print_status():
    """Pretty-print current status."""
    status = load_status()
    if not status:
        print("No status file found. Run the experiments first.")
        return

    print(f"\n{'='*60}")
    print(f"  Experiment Runner Status")
    print(f"  N = {status.get('n', '?')}")
    print(f"  Started: {status.get('started', '?')}")
    print(f"  Last update: {status.get('last_update', '?')}")
    print(f"{'='*60}\n")

    for name in ["E1", "E2", "E3", "E4", "E5"]:
        info = status.get("experiments", {}).get(name, {})
        st = info.get("status", "pending")
        expected = info.get("expected_rows", "?")
        valid = info.get("valid_rows", "?")
        error = info.get("error_rows", 0)
        attempts = info.get("attempts", 0)
        duration = info.get("duration_s", None)

        if st == "complete":
            icon = "[DONE]"
        elif st == "running":
            icon = "[....]"
        elif st == "failed":
            icon = "[FAIL]"
        else:
            icon = "[    ]"

        dur_str = f" ({duration:.0f}s)" if duration else ""
        print(f"  {icon} {name}: {valid}/{expected} valid rows, "
              f"{error} errors, {attempts} attempt(s){dur_str}")
        if st == "failed":
            last_err = info.get('last_error') or 'unknown'
            print(f"         Last error: {last_err[:100]}")

    overall = status.get("overall_status", "in_progress")
    print(f"\n  Overall: {overall}")
    if overall == "complete":
        elapsed = status.get("elapsed_s", 0)
        print(f"  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print()


# ── CSV checking ────────────────────────────────────────────────────

def check_csv(nb_config: dict, n: int) -> dict:
    """Check if a notebook's result CSV is complete.

    Returns dict with: complete (bool), valid_rows, error_rows, expected_rows, csv_exists
    """
    import pandas as pd

    name = nb_config["name"]
    pilot_mode = n < (55 if name == "E4" else 60)
    prefix = "pilot" if pilot_mode else "full"
    csv_path = RESULTS_DIR / name / f"{prefix}_results.csv"

    expected = nb_config["expected_rows"](n)
    result = {
        "csv_path": str(csv_path),
        "expected_rows": expected,
        "valid_rows": 0,
        "error_rows": 0,
        "total_rows": 0,
        "csv_exists": csv_path.exists(),
        "complete": False,
    }

    if not csv_path.exists():
        return result

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        result["read_error"] = str(e)
        return result

    result["total_rows"] = len(df)

    valid_col = nb_config["valid_col"]
    valid_values = nb_config["valid_values"]

    if valid_col not in df.columns:
        return result

    # Count valid rows
    is_valid = df[valid_col].isin(valid_values)

    # Also check for ERROR in raw_response
    if "raw_response" in df.columns:
        is_error = df["raw_response"].astype(str).str.startswith("ERROR")
        is_valid = is_valid & ~is_error

    # Check Duration_s not null (indicates successful API call)
    if "Duration_s" in df.columns:
        is_valid = is_valid & df["Duration_s"].notna()

    result["valid_rows"] = int(is_valid.sum())
    result["error_rows"] = int((~is_valid).sum())
    result["complete"] = result["valid_rows"] >= expected

    return result


# ── Notebook execution ──────────────────────────────────────────────

def run_notebook(nb_config: dict, n: int) -> tuple[bool, str]:
    """Execute a notebook via jupyter nbconvert.

    Returns (success: bool, error_msg: str)
    """
    nb_path = SCRIPT_DIR / nb_config["file"]
    if not nb_path.exists():
        return False, f"Notebook not found: {nb_path}"

    env = os.environ.copy()
    env["EXPERIMENT_N"] = str(n)

    cmd = [
        sys.executable, "-m", "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--inplace",
        "--ExecutePreprocessor.timeout", str(NOTEBOOK_TIMEOUT),
        "--ExecutePreprocessor.kernel_name", "python3",
        str(nb_path),
    ]

    log(f"  Running: {' '.join(cmd[:6])}...")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=NOTEBOOK_TIMEOUT + 120,  # extra buffer
            cwd=str(SCRIPT_DIR),
            env=env,
        )

        if result.returncode != 0:
            # Extract last few lines of stderr for diagnosis
            stderr_tail = "\n".join(result.stderr.strip().split("\n")[-10:])
            return False, f"Exit code {result.returncode}:\n{stderr_tail}"

        return True, ""

    except subprocess.TimeoutExpired:
        return False, f"Timed out after {NOTEBOOK_TIMEOUT}s"
    except Exception as e:
        return False, str(e)


# ── Main loop ───────────────────────────────────────────────────────

def run_all(n: int, only: list[str] | None = None):
    """Run all notebooks sequentially with retry logic."""
    import pandas as pd  # ensure available

    start_time = time.time()

    status = {
        "n": n,
        "started": datetime.now().isoformat(),
        "last_update": datetime.now().isoformat(),
        "overall_status": "in_progress",
        "experiments": {},
    }

    notebooks = NOTEBOOKS
    if only:
        notebooks = [nb for nb in NOTEBOOKS if nb["name"] in only]

    # Initialize status for all experiments
    for nb in notebooks:
        expected = nb["expected_rows"](n)
        csv_check = check_csv(nb, n)
        status["experiments"][nb["name"]] = {
            "status": "complete" if csv_check["complete"] else "pending",
            "expected_rows": expected,
            "valid_rows": csv_check["valid_rows"],
            "error_rows": csv_check["error_rows"],
            "attempts": 0,
            "duration_s": None,
            "last_error": None,
        }
    save_status(status)

    log(f"{'='*60}")
    log(f"Starting experiment runner — N={n}")
    log(f"Notebooks: {[nb['name'] for nb in notebooks]}")

    total_expected = sum(nb["expected_rows"](n) for nb in notebooks)
    log(f"Total expected judgments: {total_expected}")
    log(f"{'='*60}")

    all_complete = True

    for nb_idx, nb in enumerate(notebooks):
        name = nb["name"]
        exp_status = status["experiments"][name]

        # Check if already complete
        csv_check = check_csv(nb, n)
        if csv_check["complete"]:
            log(f"\n[{name}] Already complete: {csv_check['valid_rows']}/{csv_check['expected_rows']} valid rows")
            exp_status["status"] = "complete"
            exp_status["valid_rows"] = csv_check["valid_rows"]
            exp_status["error_rows"] = csv_check["error_rows"]
            save_status(status)
            continue

        log(f"\n{'='*60}")
        log(f"[{name}] Starting ({nb_idx+1}/{len(notebooks)})")
        log(f"[{name}] Expected rows: {nb['expected_rows'](n)}")
        if csv_check["csv_exists"]:
            log(f"[{name}] Existing: {csv_check['valid_rows']} valid, {csv_check['error_rows']} errors")
        log(f"{'='*60}")

        nb_start = time.time()
        nb_complete = False

        for attempt in range(1, MAX_RETRIES + 1):
            exp_status["status"] = "running"
            exp_status["attempts"] = attempt
            status["last_update"] = datetime.now().isoformat()
            save_status(status)

            log(f"[{name}] Attempt {attempt}/{MAX_RETRIES}")

            success, error_msg = run_notebook(nb, n)

            if not success:
                log(f"[{name}] Execution failed: {error_msg[:200]}")
                exp_status["last_error"] = error_msg[:500]
            else:
                log(f"[{name}] Execution succeeded")

            # Check CSV regardless (partial results may have been saved)
            csv_check = check_csv(nb, n)
            exp_status["valid_rows"] = csv_check["valid_rows"]
            exp_status["error_rows"] = csv_check["error_rows"]
            status["last_update"] = datetime.now().isoformat()
            save_status(status)

            log(f"[{name}] CSV check: {csv_check['valid_rows']}/{csv_check['expected_rows']} valid, "
                f"{csv_check['error_rows']} errors")

            if csv_check["complete"]:
                nb_complete = True
                break

            if attempt < MAX_RETRIES:
                log(f"[{name}] Waiting {WAIT_BETWEEN_RETRY}s before retry...")
                time.sleep(WAIT_BETWEEN_RETRY)

        nb_duration = time.time() - nb_start
        exp_status["duration_s"] = round(nb_duration, 1)
        exp_status["status"] = "complete" if nb_complete else "failed"
        status["last_update"] = datetime.now().isoformat()
        save_status(status)

        if nb_complete:
            log(f"[{name}] COMPLETE in {nb_duration:.0f}s ({nb_duration/60:.1f} min)")
        else:
            log(f"[{name}] FAILED after {MAX_RETRIES} attempts "
                f"({csv_check['valid_rows']}/{csv_check['expected_rows']} rows)")
            all_complete = False

        # Wait between notebooks (unless this is the last one)
        if nb_idx < len(notebooks) - 1:
            log(f"Waiting {WAIT_BETWEEN_NB}s before next notebook...")
            time.sleep(WAIT_BETWEEN_NB)

    elapsed = time.time() - start_time
    status["elapsed_s"] = round(elapsed, 1)
    status["overall_status"] = "complete" if all_complete else "partial"
    status["last_update"] = datetime.now().isoformat()
    save_status(status)

    log(f"\n{'='*60}")
    log(f"FINISHED — {'ALL COMPLETE' if all_complete else 'SOME FAILED'}")
    log(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    log(f"{'='*60}")

    # Print final summary
    print_status()

    return 0 if all_complete else 1


# ── CLI ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run all experiment notebooks sequentially with retry logic.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all.py --n 2          # Quick test run
  python run_all.py --n 60         # Full experiment
  python run_all.py --n 60 --only E2 E3  # Run only E2 and E3
  python run_all.py --status       # Check progress
  python run_all.py --check --n 60 # Check CSV completeness without running
        """,
    )
    parser.add_argument("--n", type=int, help="Number of instances per experiment")
    parser.add_argument("--status", action="store_true", help="Show current status and exit")
    parser.add_argument("--check", action="store_true", help="Check CSV completeness without running")
    parser.add_argument("--only", nargs="+", choices=["E1", "E2", "E3", "E4", "E5"],
                        help="Run only these experiments")

    args = parser.parse_args()

    if args.status:
        print_status()
        return 0

    if args.check:
        if not args.n:
            print("Error: --check requires --n")
            return 1
        print(f"\nCSV completeness check (N={args.n}):\n")
        all_ok = True
        for nb in NOTEBOOKS:
            result = check_csv(nb, args.n)
            icon = "OK" if result["complete"] else "INCOMPLETE"
            print(f"  [{icon:>10}] {nb['name']}: {result['valid_rows']}/{result['expected_rows']} valid, "
                  f"{result['error_rows']} errors")
            if not result["complete"]:
                all_ok = False
        print(f"\nOverall: {'ALL COMPLETE' if all_ok else 'INCOMPLETE'}")
        return 0 if all_ok else 1

    if not args.n:
        parser.print_help()
        print("\nError: --n is required to run experiments")
        return 1

    if args.n < 1 or args.n > 60:
        print(f"Error: N must be between 1 and 60 (got {args.n})")
        return 1

    return run_all(args.n, only=args.only)


if __name__ == "__main__":
    sys.exit(main())
