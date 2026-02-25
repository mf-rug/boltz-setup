# Boltz Structure Prediction - Workflow Automation

## Cluster Environment

- **User**: p273962
- **Storage**:
  - `/home/p273962/` - very limited quota, avoid for data
  - `/scratch/p273962/` - large quota, fast I/O, use for jobs and data
  - `/projects/p273962/` - intermediate quota
- **Scheduler**: Slurm
- **Login node**: no heavy CPU work; submit jobs via `sbatch`
- **Module system**: `module load Python/3.11.5-GCCcore-13.2.0` required before using boltz
- **Running Python from this repo**: Always set `PYTHONPATH=/scratch/p273962/boltz_claude` so that `boltz_tools` is importable. Example: `module load Python/3.11.5-GCCcore-13.2.0 && PYTHONPATH=/scratch/p273962/boltz_claude python -c "from boltz_tools.generate import ..."`. Without this you get `ModuleNotFoundError: No module named 'boltz_tools'`.
- **Boltz binary**: installed at `/home3/p273962/.local/bin/boltz`
- **Boltz model cache**: `/scratch/p273962/boltz/`

### GPU Partitions

| Partition  | Time Limit | GPUs Available                    |
|------------|------------|-----------------------------------|
| gpushort   | 4h         | A100 (x4), RTX Pro 6000 (x8), V100 (x2), L40s (x2) |
| gpumedium  | 1 day      | A100 (x4), RTX Pro 6000 (x8), V100 (x2)             |
| gpulong    | 3 days     | A100 (x4), RTX Pro 6000 (x8), V100 (x2)             |

Most boltz jobs fit within `gpushort` (4h). Use `gpumedium` for large complexes or many diffusion samples.

## What Boltz Does

Boltz-2 is a deep learning model for predicting biomolecular complex structures and binding affinities. It handles proteins, DNA, RNA, and small-molecule ligands. It succeeds Boltz-1 and approaches AlphaFold3-level accuracy. Both models are available via the `--model` flag (`boltz1` or `boltz2`, default `boltz2`).

## Input Format

Input is a YAML file (or a directory of YAML files). Each file describes one prediction job.

### Minimal example (protein only)

```yaml
sequences:
  - protein:
      id: [A]
      sequence: MVHLTPEEKSAVTALWGKVNV...
```

### Supported entity types

- **protein**: `sequence` (amino acid one-letter codes), optional `msa` (.a3m path) or use `--use_msa_server`
- **dna**: `sequence` (nucleotide codes)
- **rna**: `sequence` (nucleotide codes)
- **ligand**: `smiles` (SMILES string) or `ccd` (CCD code), mutually exclusive

Each entity gets an `id` field: a single chain ID or a list for identical copies (e.g., `id: [A, B]` for a homodimer).

### Optional sections

- **modifications**: modified residues by position and CCD code
- **constraints**: `bond` (covalent), `pocket` (binding site residues + max_distance), `contact` (residue pairs). All support `force: true` for enforcement potentials.
- **templates**: CIF/PDB paths for structural guidance, optional `chain_id`, `force`, `threshold`
- **properties**: currently supports `affinity` with a `binder` chain ID (ligand <=128 atoms, recommended <=56)

### Example with ligands, constraints, and affinity

```yaml
sequences:
  - protein:
      id: [A]
      sequence: MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH
  - protein:
      id: [B]
      sequence: MAIMILIANFRST
  - ligand:
      id: [C,D]
      smiles: '[Mg]'
  - ligand:
      id: [E]
      smiles: 'c12cc(C)c(C)cc1N=C3C(=O)NC(=O)N=C3N2C[C@H](O)[C@H](O)[C@H](O)COP(=O)(O)OP(=O)(O)OC[C@@H]4[C@@H](O)[C@@H](O)[C@@H](O4)n5cnc6c5ncnc6N'
constraints:
  - pocket:
      binder: E
      contacts: [[ A, 96 ]]
      force: true
      max_distance: 3.5
properties:
    - affinity:
        binder: E
```

## Running Boltz

### CLI command

```bash
boltz predict <input_path> [options]
```

Where `<input_path>` is a YAML file or directory of YAML files.

### Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--out_dir PATH` | (required) | Output directory |
| `--cache PATH` | `~/.boltz` | Model/data cache (use `/scratch/p273962/boltz/`) |
| `--use_msa_server` | off | Auto-generate MSAs via MMseqs2 server |
| `--recycling_steps N` | 3 | Structure recycling iterations (AF3 uses 10) |
| `--diffusion_samples N` | 1 | Number of structure samples (AF3 uses 25) |
| `--sampling_steps N` | 200 | Diffusion sampling steps |
| `--step_scale F` | 1.638 (boltz1) / 1.5 (boltz2) | Temperature; lower = more diverse |
| `--output_format pdb\|mmcif` | mmcif | Structure output format |
| `--use_potentials` | off | Apply physics-based steering potentials |
| `--override` | off | Re-run even if cached results exist |
| `--model boltz1\|boltz2` | boltz2 | Model version |
| `--seed N` | none | Random seed for reproducibility |
| `--devices N` | 1 | Number of GPUs |
| `--write_full_pae` | on | Dump PAE matrix |
| `--write_full_pde` | off | Dump PDE matrix |
| `--affinity_mw_correction` | off | MW correction for affinity head |
| `--diffusion_samples_affinity N` | 5 | Diffusion samples for affinity |

### Typical job command

```bash
boltz predict ./input/ --out_dir ./output/ --use_msa_server --recycling_steps 10 --diffusion_samples 10 --cache /scratch/p273962/boltz/
```

## Output Structure

```
<out_dir>/
  boltz_results_<input>/
    predictions/
      <yaml_name>/
        <yaml_name>_model_0.cif          # predicted structure (pLDDT in B-factor)
        confidence_<yaml_name>_model_0.json  # ptm, iptm, plddt, pde, confidence_score
        affinity_<yaml_name>.json            # if affinity requested
        pae_<yaml_name>_model_0.npz          # PAE matrix (if --write_full_pae)
```

### Affinity output fields

- `affinity_probability_binary` (0-1): probability the ligand is a binder (use for hit discovery)
- `affinity_pred_value`: predicted log10(IC50) in uM (use for optimization)

## Slurm Job Script Template

```bash
#!/bin/bash
#SBATCH --job-name=boltz_<name>
#SBATCH --time=00:30:00
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=16GB

scontrol show job $SLURM_JOB_ID
set -x
set -e
PS4='$(date +%H:%M:%S)'

echo start

module purge
module load Python/3.11.5-GCCcore-13.2.0

cd /scratch/p273962/boltz_claude/

boltz predict ./input/ --out_dir ./output/ --use_msa_server --recycling_steps 10 --diffusion_samples 10 --cache /scratch/p273962/boltz/

echo done
```

Notes:
- `set -e` ensures the job fails fast on errors
- `set -x` with `PS4` timestamps gives timestamped command tracing in the log
- `scontrol show job` prints job details (node, GPU type) for debugging
- `module purge` + `module load` ensures a clean environment
- The `--cache` flag avoids downloading models to the small home directory
- For jobs requesting affinity, consider adding `--affinity_mw_correction`
- GPU type is not explicitly constrained; Slurm picks whatever is free. Add `#SBATCH --gres=gpu:a100:1` to request a specific type if needed.

## Automation Goals

The goal is to build tooling that automates common boltz workflows:
1. **Input generation**: given protein sequences, ligand SMILES, and optional constraints, generate the YAML input file(s)
2. **Job script generation**: produce a Slurm submission script with sensible defaults, adjustable parameters
3. **Submission and monitoring**: submit via `sbatch`, track job status via `squeue`/`sacct`
4. **Result inspection**: parse output confidence JSON and affinity JSON, summarize results
