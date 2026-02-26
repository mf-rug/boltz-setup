"""Cluster-specific configuration for boltz-setup.

All cluster-specific settings (Python module, scratch paths, GPU tiers,
Slurm partitions) are read from ~/.config/boltz-setup/config.yaml.
If that file doesn't exist, built-in defaults are used and a config file
is written so the user can customise it.

Run `boltz-setup-yaml --init` to (re)write the config interactively.
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

CONFIG_PATH = Path.home() / ".config" / "boltz-setup" / "config.yaml"

# ---------------------------------------------------------------------------
# Cluster username detection
# ---------------------------------------------------------------------------

def _cluster_user() -> str:
    """Return the cluster username, derived from (in priority order):
    1. hpc / HPC env var (user@host) — if exported to the environment
    2. rsyncer config (~/.config/rsyncer/config.json → "server" field)
    3. $LOGNAME / $USER fallback
    """
    for var in ("hpc", "HPC"):
        val = os.environ.get(var, "")
        if "@" in val:
            return val.split("@")[0]
    try:
        rsyncer_cfg = Path.home() / ".config" / "rsyncer" / "config.json"
        if rsyncer_cfg.exists():
            cfg = json.loads(rsyncer_cfg.read_text())
            server = cfg.get("server", "")
            if "@" in server:
                return server.split("@")[0]
    except Exception:
        pass
    return os.environ.get("LOGNAME") or os.environ.get("USER", "user")


# ---------------------------------------------------------------------------
# Default configuration (Hábrók / RUG HPC — edit config.yaml to override)
# ---------------------------------------------------------------------------

_DEFAULTS: Dict[str, Any] = {
    # 'module load <python_module>' in every job script
    "python_module": "Python/3.11.5-GCCcore-13.2.0",

    # Scratch root — {user} is replaced at runtime with the cluster username
    "scratch_dir": "/scratch/{user}",

    # Boltz model/data cache (large, keep on scratch)
    "cache_dir": "/scratch/{user}/boltz",

    # Default parent directory for all job subdirectories
    "jobs_dir": "/scratch/{user}/boltz_jobs",

    # GPU recommendation tiers: first entry where actual_tokens <= max_tokens wins.
    # gpu_sbatch: value passed to #SBATCH --gpus-per-node
    # extra_flags: extra boltz CLI flags added automatically
    # warn: if true, emit a "very large job" warning
    "gpu_tiers": [
        {"max_tokens": 700,     "gpu_sbatch": "v100:1", "mem": "16GB",
         "extra_flags": ["--no_kernels"]},
        {"max_tokens": 1500,    "gpu_sbatch": "l40s:1", "mem": "32GB",
         "extra_flags": []},
        {"max_tokens": 2500,    "gpu_sbatch": "a100:1", "mem": "32GB",
         "extra_flags": []},
        {"max_tokens": 9_999_999, "gpu_sbatch": "a100:1", "mem": "64GB",
         "extra_flags": [], "warn": True},
    ],

    # Slurm partitions: listed in priority order (shortest first).
    # max_hours: wall-time limit for this partition
    # gpus: gpu_sbatch values available on this partition
    "partitions": [
        {"name": "gpushort",  "max_hours": 4,  "gpus": ["v100:1", "a100:1", "l40s:1", "rtxpro6000:1"]},
        {"name": "gpumedium", "max_hours": 24, "gpus": ["v100:1", "a100:1", "rtxpro6000:1"]},
        {"name": "gpulong",   "max_hours": 72, "gpus": ["v100:1", "a100:1", "rtxpro6000:1"]},
    ],

    # String that marks the start of the cluster epilog block in Slurm logs.
    # Set to "" to disable epilog parsing.
    "epilog_marker": "Hábrók Cluster",
}

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_config() -> Dict[str, Any]:
    """Load ~/.config/boltz-setup/config.yaml, writing defaults if absent."""
    if CONFIG_PATH.exists():
        try:
            import yaml
            raw = yaml.safe_load(CONFIG_PATH.read_text()) or {}
            # Deep-merge: top-level keys from file override defaults
            merged = dict(_DEFAULTS)
            merged.update(raw)
            return merged
        except Exception:
            pass
    # First run: write defaults so the user can find and edit them
    _write_default_config()
    return dict(_DEFAULTS)


def _write_default_config() -> None:
    """Write the default config to disk so users can customise it."""
    try:
        import yaml
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        header = (
            "# boltz-setup cluster configuration\n"
            "# Edit to match your HPC environment.\n"
            "# Re-run `boltz-setup-yaml --init` to regenerate from scratch.\n\n"
        )
        CONFIG_PATH.write_text(header + yaml.dump(_DEFAULTS, default_flow_style=False, allow_unicode=True))
    except Exception:
        pass  # yaml not available locally; silently skip


# ---------------------------------------------------------------------------
# Exported constants (resolved at import time)
# ---------------------------------------------------------------------------

_cfg = _load_config()
_user = _cluster_user()

def _expand(s: str) -> str:
    return s.replace("{user}", _user)

PYTHON_MODULE: str     = _cfg["python_module"]
SCRATCH: str           = _expand(_cfg["scratch_dir"])
BOLTZ_CACHE_DIR: str   = _expand(_cfg["cache_dir"])
BOLTZ_JOBS_DIR: str    = _expand(_cfg["jobs_dir"])
GPU_TIERS: List[Dict]  = _cfg["gpu_tiers"]
PARTITIONS: List[Dict] = _cfg["partitions"]
EPILOG_MARKER: str     = _cfg.get("epilog_marker", "")


# ---------------------------------------------------------------------------
# Cluster utilities (used when running on the cluster itself)
# ---------------------------------------------------------------------------

def check_boltz_installation():
    """Check that boltz is installed and whether a newer version is available.

    Returns (installed_version, latest_version, needs_update).
    installed_version is None if boltz is not importable.
    latest_version is None if the PyPI check fails (offline, timeout, etc.).
    """
    import shutil

    installed = None
    latest = None

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

    try:
        import urllib.request
        req = urllib.request.Request(
            "https://pypi.org/pypi/boltz/json",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
            latest = data["info"]["version"]
    except Exception:
        pass

    needs_update = False
    if installed and latest and installed != latest:
        iv = tuple(int(x) for x in installed.split(".") if x.isdigit())
        lv = tuple(int(x) for x in latest.split(".") if x.isdigit())
        needs_update = lv > iv

    return installed, latest, needs_update


def submit_job(job_dir: str, script_name: str = "job.sh") -> str:
    """Submit a job script via sbatch. Returns the Slurm job ID."""
    script = Path(job_dir) / script_name
    result = subprocess.run(
        ["sbatch", str(script)],
        capture_output=True, text=True, cwd=str(job_dir),
    )
    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed: {result.stderr.strip()}")
    return result.stdout.strip().split()[-1]


def check_job(job_id: str) -> str:
    """Return the current status of a Slurm job."""
    result = subprocess.run(
        ["squeue", "-j", job_id, "-o", "%T", "--noheader"],
        capture_output=True, text=True,
    )
    status = result.stdout.strip()
    if not status:
        result = subprocess.run(
            ["sacct", "-j", job_id, "-o", "State", "--noheader", "-n"],
            capture_output=True, text=True,
        )
        lines = result.stdout.strip().split("\n")
        status = lines[0].strip() if lines and lines[0].strip() else "UNKNOWN"
    return status
