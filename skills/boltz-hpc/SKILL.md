---
name: boltz-hpc
description: Predict protein / nucleic acid / ligand structures or binding affinities using Boltz-2 on the remote Slurm HPC cluster. Use when the user wants to run a structure prediction or affinity screen.
---

# boltz-hpc — predict protein structures on the HPC cluster

**Before starting any job**: confirm with the user what they want to predict and which flags to use. Structure prediction jobs consume real cluster resources and take minutes to hours — never submit without explicit user confirmation.

**On first use in a session**: run `~/.claude/scripts/check-tool-updates.sh`. It
fast-forwards these tools when they are behind, and is silent and near-instant
when they are not (self-rate-limited to once every 6 h, so running it twice
costs nothing). A `SessionStart` hook normally does this already — this line is
the backstop for machines where that hook isn't configured.

Two local CLI tools handle the full pipeline, and both are **multi-cluster** —
pass `--cluster NAME` to target any configured cluster (without it, the registry's
`default_cluster` is used):

| Step | Tool | What it does | Repo |
|------|------|--------------|------|
| 1. Generate | `boltz-setup-yaml` | Writes Boltz YAML input(s) + a **cluster-correct** `job.sh` locally | [mf-rug/boltz-setup](https://github.com/mf-rug/boltz-setup) |
| 2. Submit / fetch | `hpcjob` | `hpcjob submit` uploads + `sbatch`; `hpcjob pull` finds results by name + rsyncs them back; plus `status` / `cancel` / `recent` | [mf-rug/hpcjob](https://github.com/mf-rug/hpcjob) |

`hpcjob` supersedes the older `hpc-submit` (submit) and `rsyncer` (download) tools;
both names still work as thin shims (`hpc-submit` → `hpcjob submit`, `rsyncer` →
`hpcjob pull`), so existing commands keep running.

> **Using this skill yourself**: drop this file at `~/.claude/skills/boltz-hpc/SKILL.md`, install [boltz-setup](https://github.com/mf-rug/boltz-setup) and [hpcjob](https://github.com/mf-rug/hpcjob), then run `boltz-setup-yaml --init` and `hpcjob init`. Built and tested against Hábrók (RUG) and Snellius (SURF). All cluster-specific settings — paths, GPU tiers, partitions, modules — live in config (`~/.config/boltz-setup/config.yaml` for job-script generation, `~/.config/hpcjob/clusters.yaml` for transport), so it adapts to any SLURM cluster: add a cluster block to each and select it with `--cluster`.

---

## Tool 1 — `boltz-setup-yaml`

Generates `<out-dir>/input/*.yaml`, `<out-dir>/job.sh`, and `<out-dir>/boltz_tools/`
(a copy of the log-parsing package used by the job script on the cluster).
Auto-recommends GPU type, memory, partition, and wall time from sequence length.

### First-time setup

```bash
boltz-setup-yaml --init
```

SSHes to the cluster (via the selected cluster's `ssh_target`), detects the Boltz
model cache directory, and writes config to `~/.config/boltz-setup/config.yaml`.
Run once per cluster (`--init --cluster NAME`) after installation or when the
cluster environment changes.

`--init` also creates the **per-user venv** on the cluster (path from `venv.path`,
e.g. `/scratch/$USER/venvs/boltz/`), populated with `torch==2.10.0+cu128` and
`boltz[cuda]` — see "Universal venv" below. The venv is created on the login node
and the pip installs run via a short Slurm job (the login node kills large pip
installs).

**Adding a cluster**: `--init --cluster NAME` *updates* an existing block, it does
not create one — add the block to `config.yaml` first (a legacy single-cluster
config has none, so every job silently goes to the one cluster it knows). Check
whether that cluster already has a venv and populated cache before running
`--init`, which otherwise rebuilds them.

**Multi-cluster**: `boltz-setup-yaml --cluster NAME` bakes that cluster's
partition, GPU tiers, `module load`, venv, and cache paths into the generated
`job.sh`. Without `--cluster`, `default_cluster` is used. `hpcjob clusters` (or
the `clusters:` map in the config) lists what's available.

### Entity flags (all repeatable)

```bash
--protein SEQ          # amino acid sequence
--dna SEQ              # DNA sequence
--rna SEQ              # RNA sequence
--smiles "SMI"         # ligand SMILES
--ccd CODE             # CCD ligand code (e.g. FAD, ATP, HEM)
```

Chain IDs are assigned **A, B, C… in type order**: protein → dna → rna → smiles → ccd.
So `--protein A --protein B --smiles C` gives chains A (prot1), B (prot2), C (lig).

**Copy count** (homodimers etc.): append `[n]`
```bash
--protein "MVHLT[2]"   # chains A, B — same sequence
```

**Screening variants**: use `|` — produces one YAML per combination
```bash
--smiles "c1ccccc1|CC(=O)O|c1ccc(O)cc1"   # → 3 YAMLs, one job
```

**File input**: prefix with `@` (FASTA auto-detected; else one value per line)
```bash
--protein @seqs.fasta          # one YAML per sequence
--protein "@seqs.fasta[2]"     # homodimers of each sequence
```

### Key optional flags

```bash
--affinity CHAIN              # enable affinity prediction for binder chain
--pocket-binder CHAIN         # pocket constraint: binder chain
--pocket-contacts CHAIN:RES,… # e.g. A:96,A:100
--pocket-max-dist FLOAT        # default 6.0 Å
--no-pocket-force             # disable force for matching pocket constraint
--no-msa-server               # OPT OUT of MSA generation via MMseqs2 server. MSA is ON BY DEFAULT — the tool always adds `--use_msa_server` to the generated boltz command unless you pass `--no-msa-server`. Only use --no-msa-server if every YAML ships its own precomputed MSA (.a3m); otherwise Boltz fails with "Missing MSA's in input and --use_msa_server flag not set". Note: there is NO `--use-msa-server` flag on boltz-setup-yaml because it's the default; don't try to pass one.
--diffusion-samples N         # default 10; AF3 uses 25
--recycling-steps N           # default 10
--model boltz1|boltz2         # default boltz2
--use-potentials              # physics steering (improves ligand placement)
--affinity-mw-correction      # MW correction for affinity head
--output-format mmcif|pdb     # default mmcif
--partition NAME              # override auto-recommendation
--time HH:MM:SS               # override wall-time
--gpu SPEC                    # e.g. a100:1, v100:1
--cluster NAME                # target cluster from config (default: default_cluster)
--name NAME                   # job name prefix (default: out-dir basename)
--out-dir DIR                 # required (unless --stdout)
--stdout                      # print single YAML to stdout, no job script
```

### Examples

```bash
# Protein + ligand, affinity, pocket constraint
boltz-setup-yaml \
  --protein MVHLTPEEKSAVTALWG \
  --smiles "c1ccccc1" \
  --affinity B \
  --pocket-binder B --pocket-contacts A:96,A:100 \
  \
  --name myjob --out-dir ./myjob/

# Screen 3 SMILES against a protein
boltz-setup-yaml \
  --protein MVHLTPEEKSAVTALWG \
  --smiles "c1ccccc1|CC(=O)O|c1ccc(O)cc1" \
  \
  --name screen --out-dir ./screen/

# All sequences in a FASTA screened against one ligand
boltz-setup-yaml \
  --protein @targets.fasta \
  --smiles "c1ccccc1" \
  --affinity B \
  --name screen --out-dir ./screen/

# Heterodimer + CCD cofactor
boltz-setup-yaml \
  --protein MVHLT --protein MAIMI \
  --ccd FAD --affinity C \
  --out-dir ./holo/
```

---

## Tool 2 — `hpcjob`

One tool for **submit + fetch + monitor**, multi-cluster. Every cluster-touching
subcommand takes `--cluster NAME` (default: the registry's `default_cluster`).

**Config registry**: `~/.config/hpcjob/clusters.yaml` — one block per cluster with
`host`, `jobs_dir`, `search_paths`, `rsync_flags`, and optional `ssh_stderr_filter`
/ `notes_file`. Set up with `hpcjob init` (interactive) or `hpcjob init --migrate`
(import existing hpc-submit + rsyncer configs). List with `hpcjob clusters`; test a
cluster with `hpcjob check --cluster NAME`.

### Preflight — run this before submitting
```bash
hpcjob preflight --all                     # reachable? which GPUs are free?
hpcjob preflight -c snellius --gpu a100:1  # + a note if that GPU is saturated
hpcjob preflight -c snellius --quota       # + disk quota and SBU budget (opt-in)
```
One ssh round-trip: reachability, free GPUs per partition, queue pressure and
fairshare standing. **A failed ssh is not evidence about your credentials** — a
cluster in maintenance refuses connections in a way that reads as an auth error
(Habrók returns `Permission denied (keyboard-interactive)` with every node
down), so `preflight` fetches the configured status page whenever a cluster is
unreachable and tells you which it is. Check that before touching ssh config.

`--quota` is opt-in: site quota output carries personal data (names, emails,
group members), so it stays out of routine checks.

### Submit
```bash
hpcjob submit myjob/job.sh                    # default_cluster
hpcjob submit myjob/job.sh --cluster snellius
```
Reads the job name from `#SBATCH --job-name`, creates `<jobs_dir>/<name>/` on the
cluster, rsyncs the whole job dir (input/, boltz_tools/, job.sh — excluding
`output/`), runs `sbatch`, prints the job ID. `--overwrite` replaces an existing
remote dir; `--jobname X` overrides the dir name; `--files` adds files from outside
the job dir.

### Status / cancel
```bash
hpcjob status JOB_ID [--cluster NAME]         # squeue, then sacct
hpcjob cancel JOB_ID [--cluster NAME]
```

### Pull (download results)
```bash
hpcjob pull myjob [--cluster NAME] --yes      # find dir by name, rsync to ./myjob/
hpcjob pull /abs/remote/path --yes            # absolute path skips the search
hpcjob pull myjob --filter                    # choose file types to sync (skip big .npz)
hpcjob pull myjob --dest DIR                  # sync into DIR
hpcjob pull myjob --path PREFIX               # disambiguate multi-matches by parent prefix
```
`--yes` skips confirmations (auto-selects the first match; non-TTY stdin also falls
through so it never blocks). Per-cluster `rsync_flags` apply automatically — e.g.
Snellius uses `-L` to follow Boltz's `/tmp` symlinks.

### Recent
```bash
hpcjob recent [N] [--cluster NAME]            # recently active job dirs (via sacct)
```

**Directory naming**: remote dir = `#SBATCH --job-name` = the `--name` passed to
`boltz-setup-yaml` = the name you give `hpcjob pull`.

**Legacy shims** (still work, but use the default cluster only): `hpc-submit
myjob/job.sh` → `hpcjob submit`; `hpc-submit --status/--cancel/--check`;
`rsyncer myjob --yes` → `hpcjob pull`; `rsyncer --recent`.

---

## Full workflow example

Keep the cluster **consistent** across generate → submit → pull: `boltz-setup-yaml
--cluster X` bakes X's partition/module/venv/cache into `job.sh`, so submitting it
to a different cluster will fail. (Omit `--cluster` everywhere to use the default.)

```bash
# 1. Generate YAML + cluster-correct job.sh (also copies boltz_tools/ into myjob/)
boltz-setup-yaml \
  --protein MVHLTPEEKSAVTALWG \
  --smiles "c1ccccc1" \
  --affinity B \
  --pocket-binder B --pocket-contacts A:96,A:100 \
  --cluster snellius \
  --name myjob --out-dir ./myjob/

# 2. Submit to the same cluster
hpcjob submit myjob/job.sh --cluster snellius
# → prints: Job ID: 12345678

# 3. Check status
hpcjob status 12345678 --cluster snellius
# → RUNNING 00:02:14  or  COMPLETED 0:0 00:08:40

# 4. Download results when done
hpcjob pull myjob --cluster snellius --yes
# Results land in ./myjob/output/boltz_results_input/predictions/myjob/
#   myjob_model_0.cif              — predicted structure
#   confidence_myjob_model_0.json  — ptm, iptm, plddt, confidence_score
#   affinity_myjob.json            — affinity_probability_binary, affinity_pred_value
#   myjob_COMPLETED_<id>.log       — clean summary (auto-generated by job.sh cleanup)
```

---

## Interpreting & setting up co-folds — non-obvious gotchas

- **High confidence ≠ correct pose.** `ptm`/`iptm` can be uniformly high (~0.98) while the ligand sits in
  a non-productive pose — high confidence means a *consistent* placement, not correct chemistry. Check the
  actual geometry (distances/coordination) independently.
- **A local near-attack distance ≠ pose identity.** In a ternary co-fold (enzyme + donor + acceptor) a
  within-pose distance like "acceptor Nu → donor electrophile" can be ~constant across samples even when
  the whole ligand assembly sits elsewhere — the reacting atoms co-fold together and track each other. Judge
  conformational consistency by superposing on the protein and comparing ligand-atom positions, not by the
  reactive distance.
- **A reactive near-attack distance is NOT enough to call a modified substrate "accepted."** Also check the
  recognition/anchoring of the *non-reactive* determinants (H-bond/salt-bridge partners, including SIGN of
  charge). A modification distal to the catalytic atom can abolish binding without perturbing reaction
  geometry — Boltz won't show an overt clash, so measure it (lost salt bridge / buried like-charge).
- **Homo-oligomer with N active sites → supply N copies of EVERY ligand,** not one. Given a single
  cofactor + substrate, a dimer distributes them across the two sites → neither site is a complete complex.
- **Ligand whose CCD code is also a standard residue name (e.g. `MET`) → select by HETATM/chain, never by
  resname alone** (resname alone also captures the protein's own residues).
- **Validate a pose against a real bound-ligand crystal — but check that ligand isn't a buffer additive**
  (`_exptl_crystal_grow.pdbx_details`). Overlap with a crystallisation additive confirms only the *region*,
  not a productive pose.
- **A structure-template PDB must include SEQRES** — Boltz builds each chain's sequence from SEQRES and aligns it to the observed residues; an ATOM-only template yields an empty sequence → `parse_polymer` IndexError.
- **Resubmits reuse stale predictions** (remote `output/` survives `hpcjob --overwrite`) → Boltz silently skips; `job.sh` now defaults to `--override` to force a fresh run (MSA reused).

## Minimizing predicted structures on the cluster (YASARA array)

A common post-prediction step is energy-minimizing many predicted structures
(e.g. in neutralized water). YASARA does the minimization; the cluster just runs
it at scale. The actual EM is the **`minimize_protein` recipe in the yasara
skill** (`docs/recipes/minimize_protein.py`) — copy it next to the runner.

A generic SLURM-array template lives in `templates/minimize_array/`
(`array_job.sh` + `minimize_slice.py` + README). Adapt the `EDIT-ME` values
(YASARA pym path, partition, `module load`, slice size).

The shape that matters:
- **One long-lived headless YASARA per array task, over a contiguous SLICE of a
  manifest** — *not* one process per structure. YASARA starts once and the
  ligand-parameterization cache stays warm across the slice. (First sight of a
  ligand runs AM1BCC ~minutes, then it's cached to `<yasara>/fof/amber14.cache`,
  a disk file shared across processes — paid ~once per ligand, not per structure.)
- **Headless:** `yasara.info.mode='txt'` before the first command; pym dir on
  `sys.path`; `unset DISPLAY` in the job script.
- **Resumable** (skip if the output `.pdb` exists) and **failure-isolated**
  (per-structure `try/except`) — essential across hundreds of inputs.
- **Boltz ligand names:** Boltz mmCIFs name CCD-less ligands `LIG01`.. or custom
  5-char codes; `LoadCIF` rejects resnames >3 chars, so rewrite them to unique
  ≤3-char codes in a tmp copy first (chain-based selection is unaffected). The
  template's `fix_long_ligand_names` does this; drop it for PDB inputs.

```bash
ls /path/to/inputs/*.cif > manifest.txt
N=$(grep -c . manifest.txt); SLICE=70; LAST=$(( (N + SLICE - 1)/SLICE - 1 ))
sbatch --array=0-$LAST array_job.sh        # then rsync outputs/ back
```

---

## Cluster configuration (multi-cluster)

Two config files, both machine-local (they hold your paths/accounts — never the
tool source), each a `default_cluster` + `clusters: {name: {...}}` registry:

**`~/.config/boltz-setup/config.yaml`** — job-script generation. Each cluster block:
- `python_module` — the `module load` line (e.g. `2024 Python/3.12…` on year-stacked module systems — put the whole thing here, it's one `module load`)
- `scratch_dir` / `cache_dir` / `jobs_dir` — paths (`{user}` substituted, resolved per cluster via its `ssh_target`; SSH is skipped entirely when no path needs `{user}`, e.g. literal `/projects/<id>` paths)
- `venv.path` / `venv.pip_install` — per-user venv (see below)
- `gpu_tiers` — GPU recommendation tiers (max_tokens, gpu_sbatch, mem, extra_flags)
- `partitions` — Slurm partitions (name, max_hours, available GPUs)
- `epilog_marker` — cluster epilog start string in logs (`""` disables)
- `ssh_target` — ssh alias/user@host, used for `{user}` resolution and `--init`

**`~/.config/hpcjob/clusters.yaml`** — transport: `host`, `jobs_dir`,
`search_paths`, `rsync_flags`, optional `ssh_stderr_filter` (drop benign SSH
banners, e.g. Snellius' post-quantum warning) and `notes_file`. A `notes_file`
(e.g. `snellius.md` next to the registry) holds narrative operational gotchas that
don't fit structured config — quota etiquette, module quirks, a fresh-run
checklist — **read it before submitting to that cluster.**

A legacy flat (single-cluster) config in either file is auto-wrapped as one
`default` cluster, so pre-multicluster setups keep working unchanged.

GPU recommendation is per-cluster (`gpu_tiers`). Example — the **Hábrók** tier set
by token count (residues + ligand heavy atoms):
- < 700 tokens → V100, 16 GB (+ `--no_kernels`)
- 700–1500 → L40s, 32 GB
- 1500–2500 → A100, 32 GB
- \> 2500 → RTX Pro 6000, 64 GB (96 GB VRAM; warns: consider splitting)

Other clusters define their own — e.g. **Snellius** uses A100 (≤2500 tokens) then
H100 (larger jobs) on partitions `gpu_a100` / `gpu_h100`.

---

## Universal venv (`torch==2.10.0+cu128`)

Hábrók's CUDA driver supports up to CUDA 12.9. The default PyPI torch wheel is
now built against CUDA 13.x, which fails on every Hábrók node with
"The NVIDIA driver on your system is too old". The fix is a pinned cu128 wheel.

We verified `torch==2.10.0+cu128` on all four GPU tiers (`torch._C._cuda_getArchFlags`
includes `sm_70 75 80 86 90 100 120`; runtime tensor ops succeed on V100, A100,
L40s, and RTX Pro 6000). Newer wheels (2.11+) drop `sm_70` and break V100.

The toolchain handles this automatically:

1. **Venv creation** — `boltz-setup-yaml --init` creates one venv at
   `/scratch/$USER/venvs/boltz/` (path is configurable via `venv.path` in
   config.yaml). The venv is created on the login node; the pip installs run
   in a single short Slurm job because the login node kills heavy installs.
   The install sequence (from `venv.pip_install` in the config):
   - `pip install 'torch==2.10.0' --index-url https://download.pytorch.org/whl/cu128`
   - `pip install 'boltz[cuda]' -U`
   - `pip install --force-reinstall 'torch==2.10.0' --index-url …/cu128`
     (defensive — keeps the pin if boltz ever re-pulls torch in the future)

2. **Job script generation** — every `job.sh` sources the venv after
   `module load` (both taken from the **selected cluster's** config block), then
   invokes boltz via `python -c "from boltz.main import cli; cli()" predict …` so
   the venv's Python (and pinned torch) is used regardless of any boltz binary on
   PATH. The same template serves every cluster; only the values differ (this is
   why `--cluster snellius` yields a Snellius-correct script with no hand-editing).

If a job dies with "no kernel image is available" or "driver too old",
re-run `boltz-setup-yaml --init` after wiping the venv:
`ssh hpc "rm -rf /scratch/$USER/venvs/boltz"`.

Legacy: older toolchain versions kept a Blackwell-only venv at
`/scratch/$USER/venvs/boltz-blackwell` and used the cluster's default boltz
binary for other GPUs. That split is gone. `--init` now prints a hint about
the legacy directory so you can remove it to reclaim disk.
