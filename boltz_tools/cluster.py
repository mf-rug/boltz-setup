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
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

CONFIG_PATH = Path.home() / ".config" / "boltz-setup" / "config.yaml"

# ---------------------------------------------------------------------------
# Cluster username detection
# ---------------------------------------------------------------------------

def _resolve_ssh_user(target: str) -> Optional[str]:
    """Resolve the remote username for an SSH target (alias or user@host).

    Uses ``ssh -G`` to query the effective config without connecting.
    Returns the username or None if resolution fails.
    """
    if "@" in target:
        return target.split("@")[0]
    try:
        result = subprocess.run(
            ["ssh", "-G", target],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if line.lower().startswith("user "):
                    return line.split(None, 1)[1]
    except Exception:
        pass
    return None


def _cluster_user() -> str:
    """Return the cluster username, derived from (in priority order):
    1. hpc / HPC env var (user@host or SSH alias)
    2. hpc-submit config (~/.config/hpc-submit/config.yaml → remote_host)
    3. rsyncer config (~/.config/rsyncer/config.json → server)
    4. $LOGNAME / $USER fallback
    """
    # 1. Environment variable
    for var in ("hpc", "HPC"):
        val = os.environ.get(var, "").strip()
        if val:
            user = _resolve_ssh_user(val)
            if user:
                return user

    # 2. hpc-submit config
    try:
        hpc_cfg = Path.home() / ".config" / "hpc-submit" / "config.yaml"
        if hpc_cfg.exists():
            import yaml
            cfg = yaml.safe_load(hpc_cfg.read_text())
            host = cfg.get("remote_host", "")
            if host:
                user = _resolve_ssh_user(host)
                if user:
                    return user
    except Exception:
        pass

    # 3. rsyncer config
    try:
        rsyncer_cfg = Path.home() / ".config" / "rsyncer" / "config.json"
        if rsyncer_cfg.exists():
            cfg = json.loads(rsyncer_cfg.read_text())
            server = cfg.get("server", "")
            if server:
                user = _resolve_ssh_user(server)
                if user:
                    return user
    except Exception:
        pass

    # 4. Fallback — warn, because this is almost certainly wrong on a cluster
    fallback = os.environ.get("LOGNAME") or os.environ.get("USER", "user")
    print(
        f"[boltz-setup] WARNING: could not detect cluster username — "
        f"falling back to local user '{fallback}'.\n"
        f"  Set $hpc (e.g. export hpc=user@cluster) or run hpc-submit --init first.\n"
        f"  Or edit ~/.config/boltz-setup/config.yaml to set paths manually.",
        file=sys.stderr,
    )
    return fallback


def _get_ssh_target() -> Optional[str]:
    """Return the SSH target for the cluster (alias or user@host).

    Checks (in priority order):
    1. $hpc / $HPC env var
    2. hpc-submit config → remote_host
    3. rsyncer config → server
    """
    for var in ("hpc", "HPC"):
        val = os.environ.get(var, "").strip()
        if val:
            return val
    try:
        hpc_cfg = Path.home() / ".config" / "hpc-submit" / "config.yaml"
        if hpc_cfg.exists():
            import yaml
            cfg = yaml.safe_load(hpc_cfg.read_text())
            host = cfg.get("remote_host", "")
            if host:
                return host
    except Exception:
        pass
    try:
        rsyncer_cfg = Path.home() / ".config" / "rsyncer" / "config.json"
        if rsyncer_cfg.exists():
            cfg = json.loads(rsyncer_cfg.read_text())
            server = cfg.get("server", "")
            if server:
                return server
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Default configuration (Hábrók / RUG HPC — edit config.yaml to override)
# ---------------------------------------------------------------------------

_DEFAULTS: Dict[str, Any] = {
    # Full path to the boltz binary on the cluster (auto-detected by --init)
    # "boltz" works if it is on PATH after module load; set to absolute path if not.
    "boltz_bin": "boltz",

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


def _write_config(cfg: Dict[str, Any]) -> None:
    """Write a config dict to CONFIG_PATH."""
    import yaml
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# boltz-setup cluster configuration\n"
        "# Edit to match your HPC environment.\n"
        "# Re-run `boltz-setup-yaml --init` to detect boltz and cache on the cluster.\n\n"
    )
    CONFIG_PATH.write_text(header + yaml.dump(cfg, default_flow_style=False, allow_unicode=True))


def _write_default_config() -> None:
    """Write the default config to disk and hint about --init."""
    try:
        _write_config(dict(_DEFAULTS))
        print(
            f"[boltz-setup] Config created at {CONFIG_PATH}\n"
            "  Run `boltz-setup-yaml --init` to auto-detect boltz and the model cache on the cluster.",
            file=sys.stderr,
        )
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
BOLTZ_BIN: str         = _cfg.get("boltz_bin", "boltz")
SCRATCH: str           = _expand(_cfg["scratch_dir"])
BOLTZ_CACHE_DIR: str   = _expand(_cfg["cache_dir"])
BOLTZ_JOBS_DIR: str    = _expand(_cfg["jobs_dir"])
GPU_TIERS: List[Dict]  = _cfg["gpu_tiers"]
PARTITIONS: List[Dict] = _cfg["partitions"]
EPILOG_MARKER: str     = _cfg.get("epilog_marker", "")


# ---------------------------------------------------------------------------
# Remote detection (SSH-based, called by --init)
# ---------------------------------------------------------------------------

# Env var names commonly set by HPC sysadmins to point to user storage
_STORAGE_VARS = ("SCRATCH", "WORK", "DATA", "PROJECT", "LUSTRE", "TMPDIR")


def remote_detect(ssh_target: str) -> Dict[str, Any]:
    """SSH to the cluster and detect the boltz binary and model cache.

    Runs a login shell (bash -l) so that module-system env vars and
    sysadmin-set storage vars ($SCRATCH, $WORK, …) are visible.
    Uses BatchMode=yes so it never hangs waiting for a password —
    it relies on an active SSH ControlMaster or key-based auth.

    Returns a dict with:
        boltz_bin:  str | None  — absolute path to the boltz binary
        cache_dir:  str | None  — path to the model cache (contains boltz2_conf.ckpt)
        storage:    dict        — storage env vars found on the remote
        error:      str | None  — error message if SSH failed
    """
    # One-shot script: print env, find boltz, search for cache under storage vars
    script = textwrap.dedent("""\
        env
        _b=$(which boltz 2>/dev/null || find "$HOME/.local/bin" -name boltz -maxdepth 1 2>/dev/null | head -1)
        echo "BOLTZ_SETUP_BIN=${_b}"
        if [ -n "$_b" ]; then
            for _dir in "$SCRATCH" "$WORK" "$DATA" "$PROJECT" "$LUSTRE" "$HOME"; do
                [ -z "$_dir" ] && continue
                for _sfx in "/boltz" "/.boltz" ""; do
                    _p="${_dir}${_sfx}"
                    [ -f "${_p}/boltz2_conf.ckpt" ] || continue
                    echo "BOLTZ_SETUP_CACHE=${_p}"
                    break 2
                done
            done
        fi
    """)
    try:
        result = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
             ssh_target, "bash", "-l"],
            input=script, capture_output=True, text=True, timeout=30,
        )
    except Exception as e:
        return {"boltz_bin": None, "cache_dir": None, "storage": {}, "error": str(e)}

    if result.returncode != 0:
        msg = result.stderr.strip() or f"exit code {result.returncode}"
        return {"boltz_bin": None, "cache_dir": None, "storage": {}, "error": msg}

    boltz_bin = None
    cache_dir = None
    storage: Dict[str, str] = {}

    for line in result.stdout.splitlines():
        key, sep, val = line.partition("=")
        if not sep:
            continue
        if key == "BOLTZ_SETUP_BIN":
            boltz_bin = val.strip() or None
        elif key == "BOLTZ_SETUP_CACHE":
            cache_dir = val.strip() or None
        elif key in _STORAGE_VARS:
            storage[key] = val

    return {"boltz_bin": boltz_bin, "cache_dir": cache_dir, "storage": storage, "error": None}


def run_init() -> None:
    """Interactive --init: SSH to the cluster to detect boltz and its model cache,
    then write (or update) ~/.config/boltz-setup/config.yaml.
    """
    print("boltz-setup: cluster detection")
    print("=" * 40)

    # Resolve SSH target
    ssh_target = _get_ssh_target()
    if not ssh_target:
        ssh_target = input("SSH target (e.g. 'hpc' or 'user@cluster.example.com'): ").strip()
        if not ssh_target:
            print("Aborted.", file=sys.stderr)
            return

    print(f"Connecting to {ssh_target} ...")
    detected = remote_detect(ssh_target)

    if detected["error"]:
        print(f"\nSSH failed: {detected['error']}", file=sys.stderr)
        print(
            "Make sure an SSH ControlMaster session is open (`ssh hpc`) "
            "or key-based auth works without interaction.",
            file=sys.stderr,
        )
        return

    # --- Boltz binary ---
    boltz_bin = detected["boltz_bin"]
    if boltz_bin:
        print(f"  boltz binary : {boltz_bin}")
    else:
        print("  boltz binary : NOT FOUND", file=sys.stderr)
        print("  → Install on the cluster with: pip install --user boltz", file=sys.stderr)
        boltz_bin = "boltz"  # keep default; job will fail with clear error at runtime

    # --- Cache directory ---
    cache_dir = detected["cache_dir"]
    if cache_dir:
        print(f"  model cache  : {cache_dir}")
    else:
        if boltz_bin != "boltz":
            print("  model cache  : not found (models not yet downloaded)")
        # Build a suggestion from the first available storage var
        storage = detected["storage"]
        suggestion = None
        for var in _STORAGE_VARS:
            if var in storage:
                suggestion = f"{storage[var]}/boltz"
                break
        if not suggestion:
            suggestion = _expand(_DEFAULTS["cache_dir"])  # fallback to {user} default

        cache_dir_input = input(f"  Cache path on cluster [{suggestion}]: ").strip()
        cache_dir = cache_dir_input or suggestion

    # Load existing config (preserve gpu_tiers, partitions, etc.) and update
    cfg = _load_config()
    cfg["boltz_bin"] = boltz_bin
    cfg["cache_dir"] = cache_dir  # absolute path — no {user} substitution needed

    try:
        _write_config(cfg)
        print(f"\nConfig updated: {CONFIG_PATH}")
    except Exception as e:
        print(f"Failed to write config: {e}", file=sys.stderr)


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
