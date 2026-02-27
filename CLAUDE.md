# boltz-setup — Developer Notes for Claude

## Cluster Configuration

All cluster-specific settings are stored in `~/.config/boltz-setup/config.yaml`.
Edit that file to match your HPC environment. On first run, the file is created
with annotated defaults that you can customise.

Key settings:
- `python_module` — `module load <name>` used in every job script
- `scratch_dir` — scratch root (`{user}` is replaced with the cluster username)
- `cache_dir` — Boltz model/data cache (keep on scratch; large download ~10 GB)
- `jobs_dir` — parent directory for all job subdirectories
- `gpu_tiers` — GPU recommendation tiers (max_tokens → GPU type + memory)
- `partitions` — Slurm partition definitions (name, max_hours, available GPUs)
- `epilog_marker` — string that marks the cluster's job-end epilog in Slurm logs

The cluster username is detected from (in priority order):
1. `$hpc` / `$HPC` env var (user@host or SSH alias — resolved via `ssh -G`)
2. `~/.config/hpc-submit/config.yaml` → `remote_host` field
3. `~/.config/rsyncer/config.json` → `server` field
4. `$LOGNAME` / `$USER`

SSH aliases (e.g. `hpc`) are resolved to a username via `ssh -G <alias>` which
queries the SSH config without connecting.

## What Boltz Does

Boltz-2 predicts structures of biomolecular complexes — proteins, DNA, RNA, and
small-molecule ligands. It can also predict binding affinities. Input is a YAML
file; output is mmCIF + confidence JSON (+ affinity JSON if requested).

## What This Repo Does

`boltz_tools/` provides:
- **`cli_yaml.py`** — `boltz-setup-yaml` CLI: generates YAML input(s) + `job.sh`
  locally from flags. No cluster access needed.
- **`generate.py`** — core logic: YAML building, GPU/time recommendation,
  job script templating. Pure computation, no I/O side effects.
- **`logparse.py`** — parses `slurm-*.out` → clean `.log` summary with confidence
  and affinity scores. Called automatically by the job's cleanup trap.
- **`cluster.py`** — reads `~/.config/boltz-setup/config.yaml`; exports
  `PYTHON_MODULE`, `SCRATCH`, `BOLTZ_CACHE_DIR`, `BOLTZ_JOBS_DIR`, `GPU_TIERS`,
  `PARTITIONS`, `EPILOG_MARKER`.
- **`cli.py`** — `boltz-setup` interactive wizard (legacy; prefer `boltz-setup-yaml`).

## Boltz Input Format

### Minimal example (protein only)

```yaml
sequences:
  - protein:
      id: [A]
      sequence: MVHLTPEEKSAVTALWG...
```

### Supported entity types

- **protein**: `sequence` (one-letter AA codes), optional `msa` (.a3m path) or `--use_msa_server`
- **dna**: `sequence`
- **rna**: `sequence`
- **ligand**: `smiles` or `ccd` (mutually exclusive)

Each entity has an `id` field: single chain or list for identical copies (`[A,B]` = homodimer).

### Optional sections

- **constraints**: `pocket` (binding site residues + max_distance)
- **properties**: `affinity` with a `binder` chain ID

## Running Boltz (on the cluster)

```bash
boltz predict ./input/ --out_dir ./output/ \
  --use_msa_server --recycling_steps 10 --diffusion_samples 10 \
  --cache $BOLTZ_CACHE_DIR
```

## Output Structure

```
<out_dir>/
  boltz_results_<input>/
    predictions/
      <yaml_name>/
        <yaml_name>_model_0.cif          # predicted structure
        confidence_<yaml_name>_model_0.json  # ptm, iptm, plddt, confidence_score
        affinity_<yaml_name>.json            # if affinity requested
```

## Slurm Job Script

`build_job_script()` in `generate.py` produces a portable script that:
- Uses `$SLURM_SUBMIT_DIR` (not a hardcoded path) as the job directory
- Loads `python_module` from the cluster config
- Runs the cleanup trap at exit: parses the Slurm log via `boltz_tools log`
- The `boltz_tools/` package is shipped alongside `job.sh` by `cli_yaml.py`
  (copied into `<out-dir>/boltz_tools/`) so the trap works without any
  cluster-side installation.

## GPU / Partition Recommendation

`recommend_gpu()` reads `GPU_TIERS` from the cluster config.
`recommend_time()` reads `PARTITIONS` from the cluster config.
Both are fully configurable — no hardcoded partition names in the Python code.
