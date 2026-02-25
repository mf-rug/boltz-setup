"""Cluster-specific configuration and job submission.

This module stays on the cluster. When the interactive CLI moves to
the laptop, this gets replaced by rsync + ssh calls.
"""

import os
import subprocess
from pathlib import Path

# ---- Cluster-specific defaults (edit for your environment) ----

SCRATCH = f"/scratch/{os.environ['LOGNAME']}"
BOLTZ_JOBS_DIR = f"{SCRATCH}/boltz_jobs"
BOLTZ_CACHE_DIR = f"{SCRATCH}/boltz"
PYTHON_MODULE = "Python/3.11.5-GCCcore-13.2.0"


def check_boltz_installation():
    """Check that boltz is installed and whether a newer version is available.

    Returns (installed_version, latest_version, needs_update).
    installed_version is None if boltz is not importable.
    latest_version is None if the PyPI check fails (offline, timeout, etc.).
    """
    import shutil

    installed = None
    latest = None

    # 1. Check if boltz is installed
    if not shutil.which("boltz"):
        try:
            import boltz  # noqa: F401
        except ImportError:
            return None, None, False

    try:
        import boltz
        installed = boltz.__version__
    except (ImportError, AttributeError):
        return None, None, False

    # 2. Check latest version on PyPI (fast, 3s timeout)
    try:
        import json
        import urllib.request
        req = urllib.request.Request(
            "https://pypi.org/pypi/boltz/json",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
            latest = data["info"]["version"]
    except Exception:
        pass  # offline or slow — skip silently

    needs_update = False
    if installed and latest and installed != latest:
        iv = tuple(int(x) for x in installed.split(".") if x.isdigit())
        lv = tuple(int(x) for x in latest.split(".") if x.isdigit())
        needs_update = lv > iv

    return installed, latest, needs_update


def submit_job(job_dir: str, script_name: str = "job.sh") -> str:
    """Submit a job script via sbatch. Returns the Slurm job ID.

    The job script self-renames to include the Slurm job ID on startup
    (e.g. job.sh -> job_12345.sh).
    """
    script = Path(job_dir) / script_name
    result = subprocess.run(
        ["sbatch", str(script)],
        capture_output=True,
        text=True,
        cwd=str(job_dir),
    )
    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed: {result.stderr.strip()}")
    # "Submitted batch job 12345"
    return result.stdout.strip().split()[-1]


def check_job(job_id: str) -> str:
    """Return the current status of a Slurm job."""
    result = subprocess.run(
        ["squeue", "-j", job_id, "-o", "%T", "--noheader"],
        capture_output=True,
        text=True,
    )
    status = result.stdout.strip()
    if not status:
        # Job left the queue; check accounting
        result = subprocess.run(
            ["sacct", "-j", job_id, "-o", "State", "--noheader", "-n"],
            capture_output=True,
            text=True,
        )
        lines = result.stdout.strip().split("\n")
        status = lines[0].strip() if lines and lines[0].strip() else "UNKNOWN"
    return status
