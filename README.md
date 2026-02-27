# boltz-setup

Workflow automation toolkit for running [Boltz-2](https://github.com/jwohlwend/boltz) structure prediction jobs on a Slurm cluster. Handles input generation, GPU/time recommendation, job submission, and result parsing.

Boltz-2 predicts structures of biomolecular complexes — proteins, DNA, RNA, and small-molecule ligands — and can also predict binding affinities. It approaches AlphaFold3-level accuracy.

## Tools

| Command | Use case |
|---|---|
| `boltz-setup <name>` | Interactive wizard — guided setup for one-off jobs |
| `boltz-setup-yaml` | Non-interactive CLI — all options as flags, for scripts and pipelines |
| `boltz-setup log <dir>` | Parse a finished job's Slurm log into a clean summary |

---

## Installation

```bash
git clone git@github.com:mf-rug/boltz-setup.git
echo 'alias boltz-setup="/path/to/boltz-setup"' >> ~/.bashrc
echo 'alias boltz-setup-yaml="/path/to/boltz-setup-yaml"' >> ~/.bashrc
source ~/.bashrc
```

Requires Python 3.10+ and PyYAML:
```bash
pip install pyyaml
```

## Cluster configuration

On first run, `boltz-setup-yaml` writes `~/.config/boltz-setup/config.yaml`
with annotated defaults. Edit it to match your HPC environment, or run
`boltz-setup-yaml --init` to auto-detect the boltz binary and model cache
via SSH:

```bash
boltz-setup-yaml --init
```

This SSHs to your cluster (using `$hpc`/`$HPC` env var or your SSH config),
runs a login shell to find the `boltz` binary and model cache under common
storage locations (`$SCRATCH`, `$WORK`, `$DATA`, etc.), and writes the
detected paths to config.

Key config fields:

```yaml
# Python module loaded in every job script
python_module: Python/3.11.5-GCCcore-13.2.0

# boltz binary name or path on the cluster (auto-detected by --init)
boltz_bin: boltz

# Scratch paths ({user} is replaced with your cluster username)
scratch_dir: /scratch/{user}
cache_dir: /scratch/{user}/boltz      # auto-detected by --init
jobs_dir: /scratch/{user}/boltz_jobs

# GPU recommendation tiers (first entry where tokens <= max_tokens wins)
gpu_tiers:
  - {max_tokens: 700,  gpu_sbatch: "v100:1", mem: 16GB, extra_flags: [--no_kernels]}
  - {max_tokens: 1500, gpu_sbatch: "l40s:1", mem: 32GB, extra_flags: []}
  ...

# Slurm partitions (shortest first)
partitions:
  - {name: gpushort,  max_hours: 4,  gpus: [v100:1, a100:1, l40s:1]}
  - {name: gpumedium, max_hours: 24, gpus: [v100:1, a100:1]}
  ...
```

The cluster username is auto-detected from the `$hpc`/`$HPC` environment variable
(`user@host` format), from `~/.config/rsyncer/config.json`, or from `$LOGNAME`.

---

## `boltz-setup` — interactive wizard

```
boltz-setup <job_name>
boltz-setup --resume [JOB_NAME]
boltz-setup log <job_dir> [--file FILE]
```

Launches a step-by-step terminal wizard:

1. **Entities** — add proteins (sequence or FASTA), DNA, RNA, ligands (SMILES or CCD code), with optional variant values for screening
2. **Constraints** — optional pocket constraints (binding site residues + max distance)
3. **Properties** — optional affinity prediction
4. **Templates** — optional structural templates for guided prediction
5. **Boltz parameters** — recycling steps, diffusion samples, potentials, model version
6. **Slurm settings** — GPU, memory, time, partition (auto-recommended; press Enter to accept)
7. **Review** — shows generated YAML + job script; jump back to any step to edit

On confirmation, writes files to `$BOLTZ_JOBS_DIR/<job_name>/` (default: `/scratch/$LOGNAME/boltz_jobs/`) and optionally submits via `sbatch`.

### Resume

```bash
boltz-setup --resume myjob    # add new inputs to an existing job
boltz-setup --resume          # pick from a list of existing jobs
```

Adds new YAML inputs without removing old ones. Boltz automatically skips already-processed inputs on re-run. The wizard offers to pre-populate from an existing YAML so you only need to change what's different. New scripts are written as `job_1.sh`, `job_2.sh`, etc.

### Log parsing

```bash
boltz-setup log myjob
boltz-setup log /scratch/$LOGNAME/boltz_jobs/myjob --file slurm-12345.out
```

Parses a finished job's `slurm-*.out` and writes a clean summary `<name>_<STATUS>_<jobid>.log` containing:
- Job metadata: partition, node, GPU type
- Timing: submit/start/end, walltime
- Resource usage: memory, CPU efficiency, GPU utilisation
- Confidence scores from all models, ranked
- Affinity scores if present

Runs automatically at job end (via the cleanup trap in `job.sh`), but can also be run manually.

---

## `boltz-setup-yaml` — non-interactive CLI

All entity, constraint, property, and Slurm settings as flags. No prompts. Writes `input/*.yaml` + `job.sh` to `--out-dir`, or prints YAML to stdout with `--stdout`.

```
boltz-setup-yaml [entity flags] [constraint flags] [property flags]
                 [boltz params] [slurm settings] [output flags]
```

### Entity flags

All flags are **repeatable** — each invocation creates one entity. Chain IDs are auto-assigned A, B, C… in type order: `protein → dna → rna → smiles → ccd`.

| Flag | Description |
|---|---|
| `--protein SEQ` | Protein sequence |
| `--dna SEQ` | DNA sequence |
| `--rna SEQ` | RNA sequence |
| `--smiles SMI` | Ligand SMILES string |
| `--ccd CODE` | CCD ligand code (e.g. `FAD`, `ATP`, `UPG`) |

**Copy count** — append `[n]` for n identical chains (homodimer, etc.):
```bash
--protein "MVHLTPEEK[2]"   # chains A,B with same sequence
--smiles "[Mg][2]"         # two copies of the ligand
```

**Variants for screening** — use `|` to provide multiple values for one entity position.
Each unique combination becomes its own YAML:
```bash
--smiles "c1ccccc1|CC(=O)O|c1ccc(O)cc1"   # → 3 YAMLs
```

**File input** — prefix with `@` to load from a file. FASTA is auto-detected;
otherwise one value per line. Each sequence becomes a variant → one YAML per sequence:
```bash
--protein @seqs.fasta           # one YAML per sequence in the FASTA
--protein "@seqs.fasta[2]"      # same, but 2 copies of each
```

### Pocket constraint flags

Repeat all four flags once per constraint; they are matched positionally.

| Flag | Default | Description |
|---|---|---|
| `--pocket-binder CHAIN` | — | Binder chain ID |
| `--pocket-contacts CHAIN:RES,...` | — | e.g. `A:96,A:100` |
| `--pocket-max-dist FLOAT` | `6.0` | Max contact distance (Å) |
| `--no-pocket-force` | — | Disable force for this constraint (default: force=true) |

### Property flags

| Flag | Description |
|---|---|
| `--affinity CHAIN` | Enable affinity prediction for the given binder chain |

### Boltz parameter flags

Written into `job.sh`, not the YAML.

| Flag | Default | Description |
|---|---|---|
| `--recycling-steps N` | `10` | Structure recycling iterations |
| `--diffusion-samples N` | `10` | Number of structure samples |
| `--sampling-steps N` | `200` | Diffusion sampling steps |
| `--no-msa-server` | — | Disable MSA generation via MMseqs2 server (on by default) |
| `--model boltz1\|boltz2` | `boltz2` | Model version |
| `--output-format mmcif\|pdb` | `mmcif` | Structure output format |
| `--use-potentials` | off | Apply physics-based steering potentials |
| `--seed N` | — | Random seed for reproducibility |
| `--affinity-mw-correction` | off | Apply MW correction to affinity head |

### Slurm settings

Auto-recommended from sequence length and boltz parameters if omitted. Override any or all:

| Flag | Default | Description |
|---|---|---|
| `--partition NAME` | auto | e.g. `gpushort`, `gpumedium`, `gpulong` |
| `--time HH:MM:SS` | auto | Wall-time limit |
| `--gpu SPEC` | auto | e.g. `a100:1`, `v100:1` |
| `--mem SIZE` | auto | e.g. `32GB` |
| `--cpus N` | `7` | CPUs per task |

### Output flags

| Flag | Default | Description |
|---|---|---|
| `--out-dir DIR` | — | Directory for `input/*.yaml` + `job.sh`. Required unless `--stdout` |
| `--name NAME` | out-dir basename | Prefix for YAML filenames and Slurm `--job-name` |
| `--stdout` | off | Print YAML to stdout only (single-YAML; no job script) |
| `--init` | — | SSH to cluster, auto-detect boltz binary and model cache, update config |

### Examples

```bash
# Single protein + CCD ligand → YAML to stdout
boltz-setup-yaml --protein MVHLTPEEKSAV --ccd UPG --stdout

# Homodimer + ligand
boltz-setup-yaml --protein "MVHLTPEEK[2]" --smiles "c1ccccc1" --stdout

# Protein + ligand with affinity + pocket constraint
boltz-setup-yaml --protein MVHLTPEEK --smiles "c1ccccc1" \
  --affinity B --pocket-binder B --pocket-contacts A:96,A:100 \
  --name myjob --out-dir ./myjob/

# Screen 3 SMILES variants against a protein → 3 YAMLs + job.sh
boltz-setup-yaml --protein MVHLTPEEK --smiles "c1ccccc1|CC(=O)O|c1ccc(O)cc1" \
  --name screen --out-dir ./screen/

# Screen all sequences in a FASTA against a fixed ligand
boltz-setup-yaml --protein @seqs.fasta --smiles "c1ccccc1" \
  --affinity B --pocket-binder B --pocket-contacts A:96,A:100 \
  --name screen --out-dir ./screen/

# Heterodimer + CCD ligand (chains A, B, C)
boltz-setup-yaml --protein MVHLT --protein MAIMILIANFR --ccd FAD \
  --affinity C --out-dir ./holo/

# Complex: transferase + homodimer partner + DNA + 3 screened ligands + CCD cofactor
boltz-setup-yaml \
  --protein MENNIDLNVYFCFVNRP...LENSRS \
  --protein "MAIMILIANFR[2]" \
  --dna ATCGATCGATCG \
  --smiles "c1ccccc1|CC(=O)O|c1ccc(O)cc1" \
  --ccd UPG \
  --affinity E \
  --pocket-binder E --pocket-contacts A:96,A:100 \
  --diffusion-samples 10 --use-potentials \
  --name screen --out-dir ./screen/

# Override Slurm settings
boltz-setup-yaml --protein MVHLTPEEK --smiles "c1ccccc1" \
  --diffusion-samples 25 --partition gpumedium --time 04:00:00 \
  --name bigrun --out-dir ./bigrun/
```

---

## Output structure

Each job directory contains:

```
<job_dir>/
  input/
    <name>.yaml                  # boltz input (entities, constraints, properties)
    <name>_pAv1_lB.yaml          # variant YAMLs for screening jobs
    <name>_pAv2_lB.yaml
  boltz_tools/                   # copy of the boltz_tools package (for log parsing on cluster)
  job.sh                         # Slurm submission script (self-renames to job_<id>.sh on start)
  slurm-<jobid>.out              # raw Slurm output
  <name>_COMPLETED_<jobid>.log   # clean summary (auto-generated at job end)
  output/
    boltz_results_input/
      predictions/<name>/
        <name>_model_0.cif              # predicted structure (pLDDT in B-factor)
        confidence_<name>_model_0.json  # ptm, iptm, plddt, confidence_score
        affinity_<name>.json            # affinity output (if requested)
        pae_<name>_model_0.npz          # PAE matrix
```

### Affinity output

| Field | Description |
|---|---|
| `affinity_probability_binary` | 0–1 probability of binding (use for hit discovery) |
| `affinity_pred_value` | Predicted log₁₀(IC₅₀) in µM (use for lead optimisation) |

---

## Auto GPU/time recommendation

Both tools estimate the total token count (residues + ligand heavy atoms across all chains) and recommend:

| Tokens | GPU | Memory | Notes |
|---|---|---|---|
| < 700 | V100 | 16 GB | `--no_kernels` added automatically |
| 700–1500 | L40s | 32 GB | |
| 1500–2500 | A100 | 32 GB | |
| > 2500 | A100 | 64 GB | Warning: consider splitting |

Partition is chosen to fit the estimated wall time (`gpushort` ≤4h, `gpumedium` ≤1day, `gpulong` ≤3days). GPU type is upgraded automatically if the recommended GPU isn't available on the required partition.

---

## Typical workflow

```bash
# 1. Generate job directory (locally)
boltz-setup-yaml --protein @targets.fasta --smiles "$(cat ligand.smi)" \
  --affinity B --name screen --out-dir ./screen/

# 2. Submit to cluster (uploads files + runs sbatch via SSH)
hpc-submit screen/job.sh
# → prints: Job ID: 12345678

# 3. Check status
hpc-submit --status 12345678
# → RUNNING  or  COMPLETED 00:04:32 node01

# 4. Download results
rsyncer screen --yes
# → syncs ./screen/ from cluster, including the clean summary log

# 5. View results
cat screen/screen_COMPLETED_*.log
```

Or use the interactive wizard for one-off jobs:
```bash
boltz-setup myjob
```
