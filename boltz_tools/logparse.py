"""Slurm output log processor for Boltz jobs.

Parses slurm-*.out files, extracts structured information, loads confidence
and affinity scores, and produces a clean, human-readable log file.
"""

import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .cluster import BOLTZ_JOBS_DIR, EPILOG_MARKER
from .tui import (
    USE_COLOR,
    bold,
    dim,
    error,
    highlight,
    info,
    print_error,
    print_info,
    print_success,
    print_value,
    success,
    value,
    warning,
)


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class ParsedLog:
    """Structured data extracted from a Slurm job output."""
    job_id: str = ""
    job_name: str = ""
    status: str = "UNKNOWN"
    exit_code: Optional[int] = None
    partition: Optional[str] = None
    node: Optional[str] = None
    gpu_allocated: Optional[str] = None       # e.g. "a100"
    gpu_device: Optional[str] = None          # e.g. "NVIDIA A100-PCIE-40GB"
    time_limit: Optional[str] = None
    submit_time: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    boltz_command: Optional[str] = None
    num_failed_examples: Optional[int] = None
    failed_inputs: List[str] = field(default_factory=list)
    has_traceback: bool = False
    trap_message: Optional[str] = None
    wall_seconds: Optional[int] = None
    epilog: Dict[str, str] = field(default_factory=dict)
    successfully_processed: List[str] = field(default_factory=list)
    skipped_existing: int = 0
    confidence_scores: List[Dict] = field(default_factory=list)
    affinity_scores: List[Dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_scontrol_block(lines: List[str]) -> dict:
    """Extract key=value pairs from scontrol show job output."""
    text = " ".join(lines)
    pairs = re.findall(r'(\w+)=(\S+)', text)
    return {k: v for k, v in pairs}


def _parse_epilog_block(lines: List[str]) -> dict:
    """Extract key:value pairs from the Hábrók epilog section."""
    result = {}
    for line in lines:
        m = re.match(r'^(.+?)\s{2,}:\s+(.+)$', line)
        if m:
            result[m.group(1).strip()] = m.group(2).strip()
    return result


def parse_slurm_log(text: str) -> ParsedLog:
    """Parse the raw slurm output text into a ParsedLog."""
    log = ParsedLog()
    lines = text.splitlines()

    # --- Phase 1: Identify sections ---
    scontrol_lines = []
    epilog_lines = []
    in_scontrol = False
    in_epilog = False

    for line in lines:
        # scontrol block starts with JobId=
        if line.startswith("JobId="):
            in_scontrol = True
        if in_scontrol:
            if line.strip() == "":
                in_scontrol = False
            else:
                scontrol_lines.append(line)

        # Epilog block between ###... markers
        if line.startswith("###") and (not EPILOG_MARKER or EPILOG_MARKER not in line) and in_epilog:
            in_epilog = False
            continue
        if EPILOG_MARKER and EPILOG_MARKER in line:
            in_epilog = True
            continue
        if in_epilog:
            epilog_lines.append(line)

    # --- Phase 2: Parse scontrol ---
    if scontrol_lines:
        sc = _parse_scontrol_block(scontrol_lines)
        log.job_id = sc.get("JobId", "")
        log.job_name = sc.get("JobName", "")
        log.partition = sc.get("Partition")
        log.node = sc.get("NodeList") or sc.get("BatchHost")
        log.time_limit = sc.get("TimeLimit")
        log.submit_time = sc.get("SubmitTime")
        log.start_time = sc.get("StartTime")

        # GPU type from AllocTRES: gres/gpu:a100=1
        alloc_tres = sc.get("AllocTRES", "")
        gpu_m = re.search(r'gres/gpu:(\w+)=', alloc_tres)
        if gpu_m:
            log.gpu_allocated = gpu_m.group(1)

    # --- Phase 3: Parse epilog ---
    if epilog_lines:
        ep = _parse_epilog_block(epilog_lines)
        log.epilog = ep

        if "State" in ep:
            log.status = ep["State"].strip()
        if "End" in ep:
            log.end_time = ep["End"]
        if "Submit" in ep:
            log.submit_time = ep.get("Submit", log.submit_time)
        if "Start" in ep:
            log.start_time = ep.get("Start", log.start_time)

    # --- Phase 4: Scan individual lines for specific patterns ---
    for line in lines:
        # boltz predict command — must be a timestamped trace line
        m = re.match(r'^\d{2}:\d{2}:\d{2}(boltz predict .*)$', line)
        if m:
            log.boltz_command = m.group(1).strip()

        # Number of failed examples
        m = re.search(r'Number of failed examples:\s*(\d+)', line)
        if m:
            log.num_failed_examples = int(m.group(1))

        # Boltz "Failed to process" lines (boltz catches exceptions and skips)
        m = re.search(r'Failed to process (\S+)\.\s*Skipping', line)
        if m:
            log.failed_inputs.append(m.group(1))

        # Python tracebacks (uncaught exceptions, may follow progress bar)
        if "Traceback (most recent call last):" in line:
            log.has_traceback = True

        # Successfully processed inputs (MSA generation indicates processing)
        m = re.search(r'Generating MSA for (input/\S+\.yaml)', line)
        if m:
            log.successfully_processed.append(m.group(1))

        # Skipped existing inputs
        m = re.search(r'Found (\d+) existing processed inputs, skipping', line)
        if m:
            log.skipped_existing = int(m.group(1))
        m = re.search(r'Found some existing predictions \((\d+)\), skipping', line)
        if m:
            log.skipped_existing = int(m.group(1))

        # GPU device from CUDA message
        m = re.search(r"CUDA device \('(.+?)'\)", line)
        if m:
            log.gpu_device = m.group(1)

        # Trap: Job completed successfully
        m = re.search(r'Job completed successfully in (\d+)s', line)
        if m:
            log.trap_message = "completed"
            log.wall_seconds = int(m.group(1))

        # Trap: Job FAILED
        m = re.search(r'Job FAILED \(exit code (\d+)\) after (\d+)s', line)
        if m:
            log.trap_message = "failed"
            log.exit_code = int(m.group(1))
            log.wall_seconds = int(m.group(2))

    # Remove failed inputs from successfully_processed
    if log.failed_inputs:
        failed_set = set(log.failed_inputs)
        log.successfully_processed = [
            p for p in log.successfully_processed if p not in failed_set
        ]

    # --- Phase 5: Status determination (priority order) ---
    # 1. Epilog State (already set above)
    # 2. Trap message
    if log.status == "UNKNOWN":
        if log.trap_message == "completed":
            log.status = "COMPLETED"
        elif log.trap_message == "failed":
            log.status = "FAILED"
        elif log.num_failed_examples is not None and log.num_failed_examples == 0:
            log.status = "COMPLETED"

    # 3. Override: boltz can fail internally without propagating exit code.
    #    Detect via "Failed to process" lines or tracebacks.
    if log.status == "COMPLETED" and log.failed_inputs:
        log.status = "BOLTZ_ERROR"
    if log.status == "COMPLETED" and log.has_traceback:
        log.status = "BOLTZ_ERROR"

    return log


# ---------------------------------------------------------------------------
# Confidence / affinity score loading
# ---------------------------------------------------------------------------

def _find_prediction_dirs(job_dir: Path) -> List[Path]:
    """Find all prediction subdirectories under output/boltz_results_*/predictions/.

    Returns a sorted list of directories (one per input YAML / variant).
    """
    output_dir = job_dir / "output"
    if not output_dir.is_dir():
        return []

    dirs: List[Path] = []
    for results_dir in sorted(output_dir.iterdir()):
        if results_dir.is_dir() and results_dir.name.startswith("boltz_results_"):
            pred_dir = results_dir / "predictions"
            if pred_dir.is_dir():
                for sub in sorted(pred_dir.iterdir()):
                    if sub.is_dir():
                        dirs.append(sub)
    return dirs


def load_confidence_scores(pred_dir: Path) -> List[Dict]:
    """Load all confidence_*.json files, sorted by confidence_score desc.

    Each entry includes a ``variant`` key (the prediction subdirectory name)
    so scores from different variants can be distinguished.
    """
    scores = []
    variant_name = pred_dir.name
    for f in sorted(pred_dir.glob("confidence_*_model_*.json")):
        try:
            data = json.loads(f.read_text())
            # Extract model number from filename
            m = re.search(r'model_(\d+)', f.name)
            model_num = int(m.group(1)) if m else 0
            scores.append({
                "variant": variant_name,
                "model": model_num,
                "confidence_score": data.get("confidence_score", 0),
                "ptm": data.get("ptm", 0),
                "iptm": data.get("iptm", 0),
                "complex_plddt": data.get("complex_plddt", 0),
            })
        except (json.JSONDecodeError, OSError):
            continue
    scores.sort(key=lambda x: x["confidence_score"], reverse=True)
    return scores


def load_affinity_scores(pred_dir: Path) -> List[Dict]:
    """Load all affinity_*.json files.

    Each entry includes a ``variant`` key for multi-variant distinction.
    """
    scores = []
    variant_name = pred_dir.name
    for f in sorted(pred_dir.glob("affinity_*.json")):
        try:
            data = json.loads(f.read_text())
            scores.append({
                "variant": variant_name,
                "file": f.name,
                "affinity_probability_binary": data.get("affinity_probability_binary"),
                "affinity_pred_value": data.get("affinity_pred_value"),
            })
        except (json.JSONDecodeError, OSError):
            continue
    return scores


# ---------------------------------------------------------------------------
# Promote key output files
# ---------------------------------------------------------------------------

def _rename_cif(name: str, job_name: str) -> str:
    """Rename e.g. 'test_model_0.cif' to 'mdl1_test.cif' (1-indexed)."""
    m = re.match(r'(.+)_model_(\d+)\.cif$', name)
    if m:
        num = int(m.group(2)) + 1
        return f"mdl{num}_{m.group(1)}.cif"
    return name


def _fix_ligand_ids(text: str) -> str:
    """Fix Boltz's 4-char ligand IDs (LIG1) to wwPDB-compliant 5-char (LIG01).

    Boltz generates LIG1, LIG2 etc. which are 4 characters — violating the
    wwPDB rule that comp_id must be 3 or 5 characters. Zero-pad to 5 chars.
    See https://github.com/jwohlwend/boltz/issues/497
    """
    def _pad(m):
        return f"LIG{int(m.group(1)):02d}"
    return re.sub(r'LIG(\d+)', _pad, text)


def _promote_outputs(job_dir: Path, pred_dir: Path, multi_variant: bool = False) -> List[Path]:
    """Move .cif and affinity JSON files to output/, leave symlinks behind.

    CIF files are renamed: test_model_0.cif -> mdl1_test.cif (1-indexed).
    When *multi_variant* is True, the variant subdirectory name is included
    in the promoted filename for uniqueness (e.g. mdl1_job_001.cif).
    Ligand IDs are fixed: LIG1 -> LIG01 (wwPDB compliance).
    Returns list of promoted file paths in output/.
    """
    output_dir = job_dir / "output"
    promoted = []
    variant_name = pred_dir.name if multi_variant else None

    # CIF structures: promote new files, fix ligand IDs in all
    for src in sorted(pred_dir.glob("*.cif")):
        if not src.is_symlink():
            # New file: fix, rename, move, symlink
            content = src.read_text()
            fixed = _fix_ligand_ids(content)
            if fixed != content:
                src.write_text(fixed)
            if variant_name:
                new_name = _rename_cif(src.name, variant_name)
            else:
                new_name = _rename_cif(src.name, job_dir.name)
            dest = output_dir / new_name
            if dest.exists():
                continue
            src.rename(dest)
            src.symlink_to(dest)
            promoted.append(dest)

    # Fix ligand IDs in already-promoted CIF files
    for cif in sorted(output_dir.glob("*.cif")):
        if cif.is_symlink():
            continue
        content = cif.read_text()
        fixed = _fix_ligand_ids(content)
        if fixed != content:
            cif.write_text(fixed)

    # Affinity JSONs — prefix with variant name for multi-variant uniqueness
    for src in sorted(pred_dir.glob("affinity_*.json")):
        if src.is_symlink():
            continue
        if variant_name:
            dest_name = f"{variant_name}_{src.name}"
        else:
            dest_name = src.name
        dest = output_dir / dest_name
        if dest.exists():
            continue
        src.rename(dest)
        src.symlink_to(dest)
        promoted.append(dest)

    return promoted


# ---------------------------------------------------------------------------
# Clean log formatting
# ---------------------------------------------------------------------------

_RULER = "=" * 80


def _fmt_kv(key: str, val, width: int = 15) -> str:
    """Format a key-value line for the clean log."""
    return f"{key:<{width}}: {val}"


def _wrap_command(cmd: str, indent: int = 17, max_width: int = 78) -> str:
    """Wrap a long boltz command across multiple lines at flag boundaries."""
    if not cmd or len(cmd) + indent <= max_width:
        return cmd

    parts = cmd.split()
    lines = []
    current_line = ""

    for part in parts:
        test = f"{current_line} {part}".strip() if current_line else part
        if len(test) + indent > max_width and current_line:
            lines.append(current_line)
            current_line = part
        else:
            current_line = test

    if current_line:
        lines.append(current_line)

    if not lines:
        return cmd

    prefix = " " * indent
    return ("\n" + prefix).join(lines)


def format_clean_log(log: ParsedLog) -> str:
    """Format the ParsedLog into a clean, readable report string."""
    sections = []

    # --- Header ---
    sections.append(_RULER)
    sections.append("BOLTZ JOB REPORT")
    sections.append(_RULER)
    sections.append("")
    sections.append(_fmt_kv("Job Name", log.job_name))
    sections.append(_fmt_kv("Job ID", log.job_id))
    sections.append(_fmt_kv("Status", log.status))
    if log.partition:
        sections.append(_fmt_kv("Partition", log.partition))
    if log.node:
        sections.append(_fmt_kv("Node", log.node))
    gpu_str = ""
    if log.gpu_allocated and log.gpu_device:
        gpu_str = f"{log.gpu_allocated} ({log.gpu_device})"
    elif log.gpu_allocated:
        gpu_str = log.gpu_allocated
    elif log.gpu_device:
        gpu_str = log.gpu_device
    if gpu_str:
        sections.append(_fmt_kv("GPU", gpu_str))

    # --- Timing ---
    sections.append("")
    sections.append(_RULER)
    sections.append("TIMING")
    sections.append(_RULER)
    sections.append("")
    if log.submit_time:
        sections.append(_fmt_kv("Submitted", log.submit_time))
    if log.start_time:
        sections.append(_fmt_kv("Started", log.start_time))
    if log.end_time:
        sections.append(_fmt_kv("Ended", log.end_time))

    # Walltime from epilog
    used_wall = log.epilog.get("Used walltime")
    reserved_wall = log.epilog.get("Reserved walltime") or log.time_limit
    if used_wall and reserved_wall:
        sections.append(_fmt_kv("Walltime", f"{used_wall}  (of {reserved_wall} reserved)"))
    elif used_wall:
        sections.append(_fmt_kv("Walltime", used_wall))
    elif log.wall_seconds is not None:
        m, s = divmod(log.wall_seconds, 60)
        h, m = divmod(m, 60)
        sections.append(_fmt_kv("Walltime", f"{h:02d}:{m:02d}:{s:02d}"))

    # --- Resource usage ---
    ep = log.epilog
    has_resources = any(k in ep for k in [
        "Maximum memory used", "Used CPU time",
        "Max GPU utilization", "Max GPU memory used",
    ])
    if has_resources:
        sections.append("")
        sections.append(_RULER)
        sections.append("RESOURCE USAGE")
        sections.append(_RULER)
        sections.append("")

        mem_used = ep.get("Maximum memory used")
        mem_reserved = ep.get("Total memory reserved")
        if mem_used and mem_reserved:
            sections.append(_fmt_kv("Memory Used", f"{mem_used}  (of {mem_reserved} reserved)"))
        elif mem_used:
            sections.append(_fmt_kv("Memory Used", mem_used))

        cpu_time = ep.get("Used CPU time")
        if cpu_time:
            sections.append(_fmt_kv("CPU Time", cpu_time))

        gpu_util = ep.get("Max GPU utilization")
        if gpu_util:
            sections.append(_fmt_kv("GPU Utilization", gpu_util))

        gpu_mem = ep.get("Max GPU memory used")
        if gpu_mem:
            sections.append(_fmt_kv("GPU Memory", gpu_mem))

    # --- Prediction ---
    sections.append("")
    sections.append(_RULER)
    sections.append("PREDICTION")
    sections.append(_RULER)
    sections.append("")
    if log.boltz_command:
        sections.append(_fmt_kv("Command", _wrap_command(log.boltz_command)))
    if log.num_failed_examples is not None:
        sections.append(_fmt_kv("Failed Examples", log.num_failed_examples))
    if log.failed_inputs:
        sections.append(_fmt_kv("Failed Inputs", f"{len(log.failed_inputs)} input(s) failed:"))
        for fi in log.failed_inputs:
            sections.append(f"                 - {fi}")
    if log.has_traceback:
        sections.append(_fmt_kv("Traceback", "Python traceback detected in output"))

    # --- Confidence scores ---
    if log.confidence_scores:
        # Detect multi-variant: check if there are multiple distinct variant names
        variants = []
        seen = set()
        for e in log.confidence_scores:
            v = e.get("variant", "")
            if v not in seen:
                variants.append(v)
                seen.add(v)
        multi_variant = len(variants) > 1

        sections.append("")
        sections.append(_RULER)
        sections.append("CONFIDENCE SCORES (ranked by confidence)")
        sections.append(_RULER)

        if multi_variant:
            # Summary line
            all_sorted = sorted(
                log.confidence_scores,
                key=lambda x: x["confidence_score"], reverse=True,
            )
            best = all_sorted[0]
            sections.append("")
            sections.append(
                f"  {len(variants)} variants, best overall: "
                f"{best['variant']} model {best['model']} "
                f"(confidence={best['confidence_score']:.4f})"
            )

            # Group by variant
            for vname in variants:
                v_scores = [e for e in log.confidence_scores if e.get("variant") == vname]
                v_scores.sort(key=lambda x: x["confidence_score"], reverse=True)
                sections.append("")
                sections.append(f"  --- {vname} ---")
                sections.append(
                    f"  {'Model':>5}  {'Confidence':>10}  {'pTM':>6}  {'ipTM':>6}  {'pLDDT':>6}"
                )
                sections.append(
                    f"  {'-----':>5}  {'----------':>10}  {'------':>6}  {'------':>6}  {'------':>6}"
                )
                for entry in v_scores:
                    sections.append(
                        f"  {entry['model']:>5}  "
                        f"{entry['confidence_score']:>10.4f}  "
                        f"{entry['ptm']:>6.3f}  "
                        f"{entry['iptm']:>6.3f}  "
                        f"{entry['complex_plddt']:>6.3f}"
                    )
        else:
            sections.append("")
            # Table header
            sections.append(
                f"  {'Model':>5}  {'Confidence':>10}  {'pTM':>6}  {'ipTM':>6}  {'pLDDT':>6}"
            )
            sections.append(
                f"  {'-----':>5}  {'----------':>10}  {'------':>6}  {'------':>6}  {'------':>6}"
            )

            for entry in log.confidence_scores:
                sections.append(
                    f"  {entry['model']:>5}  "
                    f"{entry['confidence_score']:>10.4f}  "
                    f"{entry['ptm']:>6.3f}  "
                    f"{entry['iptm']:>6.3f}  "
                    f"{entry['complex_plddt']:>6.3f}"
                )

            best = log.confidence_scores[0]
            sections.append("")
            sections.append(
                f"  Best: model {best['model']} "
                f"(confidence={best['confidence_score']:.4f})"
            )

    # --- Affinity scores ---
    if log.affinity_scores:
        # Detect multi-variant for affinity too
        aff_variants = []
        aff_seen = set()
        for e in log.affinity_scores:
            v = e.get("variant", "")
            if v not in aff_seen:
                aff_variants.append(v)
                aff_seen.add(v)
        multi_aff = len(aff_variants) > 1

        sections.append("")
        sections.append(_RULER)
        sections.append("AFFINITY SCORES")
        sections.append(_RULER)
        sections.append("")

        if multi_aff:
            for vname in aff_variants:
                v_affs = [e for e in log.affinity_scores if e.get("variant") == vname]
                sections.append(f"  --- {vname} ---")
                for entry in v_affs:
                    prob = entry.get("affinity_probability_binary")
                    pred = entry.get("affinity_pred_value")
                    parts = []
                    if prob is not None:
                        parts.append(f"P(binder)={prob:.4f}")
                    if pred is not None:
                        parts.append(f"log10(IC50)={pred:.4f}")
                    sections.append(f"  {entry['file']}: {', '.join(parts)}")
        else:
            for entry in log.affinity_scores:
                prob = entry.get("affinity_probability_binary")
                pred = entry.get("affinity_pred_value")
                parts = []
                if prob is not None:
                    parts.append(f"P(binder)={prob:.4f}")
                if pred is not None:
                    parts.append(f"log10(IC50)={pred:.4f}")
                sections.append(f"  {entry['file']}: {', '.join(parts)}")

    # Final ruler
    sections.append("")
    sections.append(_RULER)

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Output filtering helpers
# ---------------------------------------------------------------------------

def _parse_job_times(log: ParsedLog) -> Tuple[Optional[datetime], Optional[datetime]]:
    """Parse job start/end times from the parsed log.

    Uses epilog end_time (actual end) rather than scontrol EndTime (which is
    the time limit). Returns (None, None) on failure.
    """
    try:
        start_dt = datetime.fromisoformat(log.start_time) if log.start_time else None
    except (ValueError, TypeError):
        start_dt = None
    try:
        end_dt = datetime.fromisoformat(log.end_time) if log.end_time else None
    except (ValueError, TypeError):
        end_dt = None
    return start_dt, end_dt


def _prediction_dir_in_window(
    pred_dir: Path,
    start_dt: Optional[datetime],
    end_dt: Optional[datetime],
) -> bool:
    """Check if any confidence file in pred_dir has mtime within the job window.

    Window is [start - 60s, end + 60s] to account for clock skew.
    Returns True if times are None (safe fallback — preserves current behavior).
    """
    if start_dt is None or end_dt is None:
        return True  # can't filter, keep everything

    margin = timedelta(seconds=60)
    window_start = start_dt - margin
    window_end = end_dt + margin

    conf_files = list(pred_dir.glob("confidence_*_model_*.json"))
    if not conf_files:
        return True  # no confidence files, can't filter

    for f in conf_files:
        mtime = datetime.fromtimestamp(f.stat().st_mtime)
        if window_start <= mtime <= window_end:
            return True
    return False


def _prediction_dir_matches_inputs(pred_dir: Path, log: ParsedLog) -> Optional[bool]:
    """Check if a prediction directory matches the inputs processed by this job.

    Returns:
        True  — directory matches a successfully processed input
        False — directory matches a failed input, or successfully_processed is
                non-empty but this dir doesn't match any
        None  — can't determine (fall back to timestamp filtering)
    """
    dir_name = pred_dir.name

    # Strip the "input/" prefix from failed_inputs for comparison
    failed_stems = set()
    for fi in log.failed_inputs:
        # "input/foo.yaml" -> "foo"
        stem = Path(fi).stem
        failed_stems.add(stem)

    if dir_name in failed_stems:
        return False

    if log.successfully_processed:
        processed_stems = set()
        for sp in log.successfully_processed:
            stem = Path(sp).stem
            processed_stems.add(stem)
        if dir_name in processed_stems:
            return True
        return False

    return None  # can't determine, fall back to timestamp


# ---------------------------------------------------------------------------
# Main processing pipeline
# ---------------------------------------------------------------------------

def _find_slurm_file(job_dir: Path) -> Optional[Path]:
    """Find the most recent slurm-*.out file in job_dir."""
    candidates = sorted(job_dir.glob("slurm-*.out"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def process_log(job_dir: Path, slurm_file: Optional[Path] = None):
    """Full pipeline: parse slurm log, load scores, write clean log.

    Returns (output_path, parsed_log).
    """
    job_dir = Path(job_dir)

    if slurm_file is None:
        slurm_file = _find_slurm_file(job_dir)
    if slurm_file is None:
        raise FileNotFoundError(f"No slurm-*.out found in {job_dir}")

    text = slurm_file.read_text()
    log = parse_slurm_log(text)

    # Load confidence/affinity scores and promote key files.
    # Filter prediction dirs to only include those produced by THIS job.
    all_pred_dirs = _find_prediction_dirs(job_dir)
    start_dt, end_dt = _parse_job_times(log)

    pred_dirs = []
    for pd in all_pred_dirs:
        input_match = _prediction_dir_matches_inputs(pd, log)
        if input_match is False:
            continue
        if input_match is True:
            pred_dirs.append(pd)
            continue
        # Unknown — use timestamp filter
        if _prediction_dir_in_window(pd, start_dt, end_dt):
            pred_dirs.append(pd)

    multi_variant = len(pred_dirs) > 1
    for pred_dir in pred_dirs:
        log.confidence_scores.extend(load_confidence_scores(pred_dir))
        log.affinity_scores.extend(load_affinity_scores(pred_dir))
        _promote_outputs(job_dir, pred_dir, multi_variant=multi_variant)
    # Sort all confidence scores by score descending
    log.confidence_scores.sort(key=lambda x: x["confidence_score"], reverse=True)

    # Format and write
    clean = format_clean_log(log)
    name = log.job_name or job_dir.name
    status = log.status or "UNKNOWN"
    job_id = log.job_id or "noid"
    output_path = job_dir / f"{name}_{status}_{job_id}.log"
    output_path.write_text(clean + "\n")

    return output_path, log


# ---------------------------------------------------------------------------
# Terminal summary (colored)
# ---------------------------------------------------------------------------

def _print_terminal_summary(log: ParsedLog, output_path: Path):
    """Print a short colored summary to the terminal."""
    print()

    # Status line
    status_str = log.status
    if status_str == "COMPLETED":
        status_styled = success(status_str)
    elif status_str in ("FAILED", "TIMEOUT", "BOLTZ_ERROR"):
        status_styled = error(status_str)
    else:
        status_styled = warning(status_str)

    print(f"  {bold(log.job_name or '?')} [{log.job_id}] — {status_styled}")

    # Timing
    used_wall = log.epilog.get("Used walltime")
    if used_wall:
        print(f"  Walltime: {value(used_wall)}")
    elif log.wall_seconds is not None:
        m, s = divmod(log.wall_seconds, 60)
        h, m = divmod(m, 60)
        print(f"  Walltime: {value(f'{h:02d}:{m:02d}:{s:02d}')}")

    # GPU
    gpu_util = log.epilog.get("Max GPU utilization")
    if gpu_util:
        print(f"  GPU util: {value(gpu_util)}")

    # Failed examples
    if log.num_failed_examples is not None:
        if log.num_failed_examples == 0:
            print(f"  Failed examples: {success('0')}")
        else:
            print(f"  Failed examples: {error(str(log.num_failed_examples))}")

    # Boltz-level failures
    if log.failed_inputs:
        print(f"  Failed inputs: {error(str(len(log.failed_inputs)))}")
        for fi in log.failed_inputs:
            print(f"    - {fi}")
    if log.has_traceback and not log.failed_inputs:
        print(f"  {error('Python traceback detected in output')}")

    # Best confidence
    if log.confidence_scores:
        # Detect multi-variant
        variants = {e.get("variant", "") for e in log.confidence_scores}
        all_sorted = sorted(
            log.confidence_scores,
            key=lambda x: x["confidence_score"], reverse=True,
        )
        best = all_sorted[0]

        if len(variants) > 1:
            print(f"  Variants: {value(str(len(variants)))}")
            conf_str = f"{best['confidence_score']:.4f}"
            print(
                f"  Best overall: {value(best.get('variant', '?'))} "
                f"model {value(str(best['model']))} "
                f"(confidence={value(conf_str)})"
            )
        else:
            conf_str = f"{best['confidence_score']:.4f}"
            print(
                f"  Best model: {value(str(best['model']))} "
                f"(confidence={value(conf_str)})"
            )

    # Affinity
    if log.affinity_scores:
        aff_variants = {e.get("variant", "") for e in log.affinity_scores}
        if len(aff_variants) > 1:
            # Show best affinity across all variants
            best_aff = max(
                (e for e in log.affinity_scores if e.get("affinity_probability_binary") is not None),
                key=lambda x: x["affinity_probability_binary"],
                default=None,
            )
            if best_aff:
                prob = best_aff["affinity_probability_binary"]
                pred = best_aff.get("affinity_pred_value")
                parts = [f"P(binder)={prob:.4f}"]
                if pred is not None:
                    parts.append(f"log10(IC50)={pred:.4f}")
                print(
                    f"  Best affinity: {value(best_aff.get('variant', '?'))} — "
                    f"{value(', '.join(parts))}"
                )
        else:
            for entry in log.affinity_scores:
                prob = entry.get("affinity_probability_binary")
                pred = entry.get("affinity_pred_value")
                parts = []
                if prob is not None:
                    parts.append(f"P(binder)={prob:.4f}")
                if pred is not None:
                    parts.append(f"log10(IC50)={pred:.4f}")
                print(f"  Affinity: {value(', '.join(parts))}")

    print()
    print_success(f"Clean log written to {output_path}")
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def log_main():
    """CLI entry point for `python -m boltz_tools log`."""
    args = sys.argv[2:]  # skip 'boltz_tools' and 'log'

    # Parse arguments
    job_path = None
    slurm_file = None
    i = 0
    while i < len(args):
        if args[i] == "--file" and i + 1 < len(args):
            slurm_file = Path(args[i + 1])
            i += 2
        elif args[i].startswith("-"):
            print_error(f"Unknown option: {args[i]}")
            sys.exit(1)
        else:
            job_path = args[i]
            i += 1

    if job_path is None:
        print(f"Usage: boltz-setup log {highlight('<job_dir>')} [--file <slurm_file>]")
        print()
        print(f"  {dim('job_dir can be an absolute path or a job name')} ")
        print(f"  {dim(f'(resolved against {BOLTZ_JOBS_DIR})')}")
        sys.exit(1)

    # Resolve job directory
    job_dir = Path(job_path)
    if not job_dir.is_absolute():
        job_dir = Path(BOLTZ_JOBS_DIR) / job_path
    if not job_dir.is_dir():
        print_error(f"Directory not found: {job_dir}")
        sys.exit(1)

    # Resolve slurm file
    if slurm_file and not slurm_file.is_absolute():
        slurm_file = job_dir / slurm_file

    try:
        output_path, log = process_log(job_dir, slurm_file)
    except FileNotFoundError as e:
        print_error(str(e))
        sys.exit(1)

    _print_terminal_summary(log, output_path)
