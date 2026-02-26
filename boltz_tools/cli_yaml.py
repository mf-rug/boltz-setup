"""Non-interactive YAML generation CLI for boltz-setup-yaml.

Takes all entity/constraint/property definitions as flags and writes
YAML file(s) + a Slurm job script, with no prompts. Designed for
scripting and batch pipelines.
"""

import argparse
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .generate import (
    AffinityProperty,
    BoltzParams,
    DnaEntity,
    LigandEntity,
    PocketConstraint,
    ProteinEntity,
    RnaEntity,
    SlurmParams,
    VariantSet,
    build_job_script,
    build_yaml,
    build_yaml_variants,
    estimate_tokens,
    parse_fasta,
    recommend_gpu,
    recommend_time,
    validate_ccd,
    validate_dna_sequence,
    validate_protein_sequence,
    validate_rna_sequence,
    validate_smiles,
)
from .cluster import BOLTZ_CACHE_DIR, PYTHON_MODULE


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_copy_count(raw: str) -> Tuple[str, int]:
    """Parse optional [n] suffix. Returns (value_without_suffix, n). Default n=1."""
    m = re.match(r'^(.*)\[(\d+)\]$', raw.strip())
    if m:
        return m.group(1).strip(), int(m.group(2))
    return raw.strip(), 1


def _load_file_values(
    path_str: str,
    validator_fn,
) -> Tuple[List[str], List[Optional[str]]]:
    """Load validated values from a file.

    FASTA format is auto-detected (any line starting with '>').
    Otherwise one value per non-empty line.

    Returns (values, names) where names come from FASTA headers or are None.
    Raises ValueError on missing file or validation errors.
    """
    path = Path(path_str)
    if not path.exists():
        raise ValueError(f"File not found: {path_str}")
    text = path.read_text()

    fasta_entries = parse_fasta(text)
    if fasta_entries:
        raw_pairs: List[Tuple[str, Optional[str]]] = [
            (seq, name) for name, seq in fasta_entries
        ]
    else:
        raw_pairs = [
            (line.strip(), None)
            for line in text.splitlines()
            if line.strip()
        ]

    if not raw_pairs:
        raise ValueError(f"No values found in {path_str}")

    values: List[str] = []
    names: List[Optional[str]] = []
    for seq, name in raw_pairs:
        cleaned, err = validator_fn(seq)
        if err and not err.startswith("Warning:"):
            raise ValueError(f"In {path_str}: {err}")
        values.append(cleaned)
        names.append(name)
    return values, names


def _parse_entity_values(
    raw: str,
    validator_fn,
) -> Tuple[List[str], List[Optional[str]], int]:
    """Parse one entity flag value into (values, names, copy_count).

    Syntax supported:
      SEQ            → single value, copy_count=1
      SEQ[2]         → single value, copy_count=2
      SEQ1|SEQ2      → two variant values, copy_count=1
      @path          → load from file, copy_count=1
      @path[2]       → load from file, copy_count=2
      @path|SEQ      → file values + inline, copy_count=1

    All | parts must agree on copy_count or an error is raised.
    """
    parts = raw.split('|')
    all_values: List[str] = []
    all_names: List[Optional[str]] = []
    copy_counts: List[int] = []

    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part.startswith('@'):
            path_raw = part[1:]
            path_str, cc = _parse_copy_count(path_raw)
            vals, nms = _load_file_values(path_str, validator_fn)
            all_values.extend(vals)
            all_names.extend(nms)
            copy_counts.append(cc)
        else:
            val, cc = _parse_copy_count(part)
            if not val:
                continue
            cleaned, err = validator_fn(val)
            if err and not err.startswith("Warning:"):
                raise ValueError(err)
            all_values.append(cleaned)
            all_names.append(None)
            copy_counts.append(cc)

    if not all_values:
        raise ValueError(f"No values parsed from: {raw!r}")

    unique_ccs = set(copy_counts)
    if len(unique_ccs) > 1:
        raise ValueError(
            f"Inconsistent copy counts in '{raw}': {unique_ccs}. "
            "All | parts must have the same [n] suffix."
        )
    copy_count = copy_counts[0] if copy_counts else 1
    return all_values, all_names, copy_count


def _next_chain_ids(used_count: int, n: int) -> List[str]:
    """Assign n chain IDs starting from the given offset.

    A=0, B=1, …, Z=25, AA=26, AB=27, …
    """
    ids = []
    for i in range(n):
        idx = used_count + i
        if idx < 26:
            ids.append(chr(ord('A') + idx))
        else:
            first = (idx // 26) - 1
            second = idx % 26
            ids.append(chr(ord('A') + first) + chr(ord('A') + second))
    return ids


def _parse_pocket_contacts(raw: str) -> List[Tuple[str, int]]:
    """Parse 'A:96,A:100' → [('A', 96), ('A', 100)]."""
    contacts = []
    for pair in raw.split(','):
        pair = pair.strip()
        if not pair:
            continue
        if ':' not in pair:
            raise ValueError(
                f"Invalid contact format '{pair}' — expected CHAIN:RESIDUE (e.g. A:96)"
            )
        chain, res_str = pair.split(':', 1)
        try:
            res = int(res_str.strip())
        except ValueError:
            raise ValueError(
                f"Residue number must be an integer, got '{res_str.strip()}'"
            )
        contacts.append((chain.strip(), res))
    return contacts


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def yaml_main() -> None:
    """Entry point for boltz-setup-yaml."""
    parser = argparse.ArgumentParser(
        prog="boltz-setup-yaml",
        description=(
            "Non-interactive YAML generation for Boltz predictions.\n"
            "Takes all entity/constraint/property definitions as flags.\n"
            "Suitable for scripting and batch pipelines.\n\n"
            "Chain IDs are assigned A, B, C… in type order: protein → dna → rna → smiles → ccd.\n"
            "Within each type, flags are processed left-to-right as they appear on the command line."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Single protein + ligand → YAML to stdout
  boltz-setup-yaml --protein MVHLTPEEK --smiles "c1ccccc1" --stdout

  # Homodimer (2 copies) + ligand
  boltz-setup-yaml --protein "MVHLTPEEK[2]" --smiles "c1ccccc1" --stdout

  # Protein + two ligand variants (screening) → 2 YAMLs + job.sh
  boltz-setup-yaml --protein MVHLTPEEK --smiles "c1ccccc1|CC(=O)O" \\
    --name myjob --out-dir ./myjob/

  # Protein from FASTA + ligand + affinity + pocket constraint
  boltz-setup-yaml --protein @seqs.fasta --smiles "c1ccccc1" \\
    --affinity B --pocket-binder B --pocket-contacts A:96,A:100 \\
    --use-msa-server --name screen --out-dir ./screen/

  # Protein from FASTA with 2 copies each
  boltz-setup-yaml --protein "@seqs.fasta[2]" --smiles "c1ccccc1" --out-dir ./s2/

  # Heterodimer + CCD ligand with affinity
  boltz-setup-yaml --protein MVHLT --protein MAIMI --ccd FAD \\
    --affinity C --out-dir ./holo/

  # Override Slurm settings
  boltz-setup-yaml --protein "LONGSEQ" --smiles "c1ccccc1" \\
    --diffusion-samples 25 --recycling-steps 10 \\
    --partition gpumedium --time 04:00:00 \\
    --name bigrun --out-dir ./bigrun/
""",
    )

    # --- Entity flags ---
    ent = parser.add_argument_group(
        "entity flags",
        "All repeatable — each flag invocation creates one entity. "
        "Use | to provide variant values for screening (→ one YAML per combination). "
        "Prefix with @ to load from a file (FASTA auto-detected; else one value per line). "
        "Append [n] for n identical chain copies, e.g. --protein SEQ[2] for a homodimer.",
    )
    ent.add_argument(
        "--protein", action="append", metavar="SEQ[n]|SEQ2[n]",
        help="Protein sequence(s).",
    )
    ent.add_argument(
        "--dna", action="append", metavar="SEQ[n]|SEQ2[n]",
        help="DNA sequence(s).",
    )
    ent.add_argument(
        "--rna", action="append", metavar="SEQ[n]|SEQ2[n]",
        help="RNA sequence(s).",
    )
    ent.add_argument(
        "--smiles", action="append", metavar="SMI[n]|SMI2[n]",
        help="Ligand SMILES string(s).",
    )
    ent.add_argument(
        "--ccd", action="append", metavar="CODE[n]|CODE2[n]",
        help="CCD ligand code(s), e.g. FAD, ATP.",
    )

    # --- Constraint flags ---
    con = parser.add_argument_group(
        "pocket constraint flags",
        "Repeat all four flags once per constraint; they are matched positionally.",
    )
    con.add_argument(
        "--pocket-binder", action="append", metavar="CHAIN",
        help="Binder chain ID for a pocket constraint.",
    )
    con.add_argument(
        "--pocket-contacts", action="append", metavar="CHAIN:RES,...",
        help="Comma-separated CHAIN:RESIDUE contacts, e.g. A:96,A:100.",
    )
    con.add_argument(
        "--pocket-max-dist", action="append", type=float, metavar="FLOAT",
        help="Max contact distance in Å (default: 6.0).",
    )
    con.add_argument(
        "--no-pocket-force", action="append_const", const=True,
        dest="no_pocket_force",
        help="Disable force for the corresponding constraint (default: force=True).",
    )

    # --- Property flags ---
    prop = parser.add_argument_group("property flags")
    prop.add_argument(
        "--affinity", action="append", metavar="CHAIN",
        help="Enable affinity prediction for the given binder chain.",
    )

    # --- Boltz parameter flags ---
    bp_grp = parser.add_argument_group("boltz parameters (written to job script, not YAML)")
    bp_grp.add_argument("--recycling-steps", type=int, default=10, metavar="N",
                        help="Recycling iterations (default: 10).")
    bp_grp.add_argument("--diffusion-samples", type=int, default=10, metavar="N",
                        help="Number of diffusion samples (default: 10).")
    bp_grp.add_argument("--sampling-steps", type=int, default=200, metavar="N",
                        help="Diffusion sampling steps (default: 200).")
    bp_grp.add_argument("--use-msa-server", action="store_true", default=False,
                        help="Generate MSAs via MMseqs2 server.")
    bp_grp.add_argument("--model", default="boltz2", choices=["boltz1", "boltz2"],
                        help="Model version (default: boltz2).")
    bp_grp.add_argument("--output-format", default="mmcif", choices=["mmcif", "pdb"],
                        help="Structure output format (default: mmcif).")
    bp_grp.add_argument("--use-potentials", action="store_true", default=False,
                        help="Apply physics-based steering potentials.")
    bp_grp.add_argument("--seed", type=int, default=None, metavar="N",
                        help="Random seed for reproducibility.")
    bp_grp.add_argument("--affinity-mw-correction", action="store_true", default=False,
                        help="Apply MW correction to affinity head.")

    # --- Slurm flags ---
    sl_grp = parser.add_argument_group(
        "slurm settings",
        "Auto-recommended from sequence length and boltz params if omitted.",
    )
    sl_grp.add_argument("--partition", metavar="NAME",
                        help="Slurm partition, e.g. gpushort, gpumedium, gpulong.")
    sl_grp.add_argument("--time", metavar="HH:MM:SS",
                        help="Wall-time limit.")
    sl_grp.add_argument("--gpu", metavar="SPEC",
                        help="GPU spec, e.g. a100:1 (default: auto-recommended).")
    sl_grp.add_argument("--mem", metavar="SIZE",
                        help="Memory, e.g. 32GB (default: auto-recommended).")
    sl_grp.add_argument("--cpus", type=int, default=7, metavar="N",
                        help="CPUs per task (default: 7).")

    # --- Output flags ---
    out_grp = parser.add_argument_group("output flags")
    out_grp.add_argument(
        "--out-dir", metavar="DIR",
        help="Directory for YAML(s) and job.sh. Required unless --stdout.",
    )
    out_grp.add_argument(
        "--name", default="job", metavar="NAME",
        help="Job name prefix for YAML filenames and Slurm --job-name (default: job).",
    )
    out_grp.add_argument(
        "--stdout", action="store_true",
        help="Print YAML to stdout only (single-YAML; no job script written).",
    )

    args = parser.parse_args()

    # Validate output mode
    if not args.stdout and not args.out_dir:
        parser.error("Either --out-dir or --stdout is required.")

    # -----------------------------------------------------------------------
    # 1. Parse entities
    # Chain ID order: protein → dna → rna → smiles → ccd
    # -----------------------------------------------------------------------
    entities: List = []
    # entity_index → (values, names) for entities with >1 variant value
    variant_set_data: Dict[int, Tuple[List[str], List[Optional[str]]]] = {}
    used_count = 0

    entity_specs = [
        # (flag_list, EntityClass_or_None, validator_fn, ligand_key)
        (args.protein, ProteinEntity,  validate_protein_sequence, None),
        (args.dna,     DnaEntity,      validate_dna_sequence,     None),
        (args.rna,     RnaEntity,      validate_rna_sequence,     None),
        (args.smiles,  LigandEntity,   validate_smiles,           "smiles"),
        (args.ccd,     LigandEntity,   validate_ccd,              "ccd"),
    ]

    for raw_list, EntityClass, validator_fn, ligand_key in entity_specs:
        if not raw_list:
            continue
        for raw in raw_list:
            try:
                values, names, copy_count = _parse_entity_values(raw, validator_fn)
            except ValueError as exc:
                parser.error(str(exc))

            ids = _next_chain_ids(used_count, copy_count)
            used_count += copy_count
            ent_idx = len(entities)

            if ligand_key == "smiles":
                entity = LigandEntity(ids=ids, smiles=values[0])
            elif ligand_key == "ccd":
                entity = LigandEntity(ids=ids, ccd=values[0])
            else:
                entity = EntityClass(ids=ids, sequence=values[0])

            entities.append(entity)

            if len(values) > 1:
                variant_set_data[ent_idx] = (values, names)

    if not entities:
        parser.error(
            "At least one entity is required "
            "(--protein, --dna, --rna, --smiles, or --ccd)."
        )

    # -----------------------------------------------------------------------
    # 2. Build VariantSet
    # -----------------------------------------------------------------------
    if variant_set_data:
        vs: Optional[VariantSet] = VariantSet(
            variants={i: vals for i, (vals, _names) in variant_set_data.items()},
            variant_names={
                i: nms
                for i, (_vals, nms) in variant_set_data.items()
                if any(n is not None for n in nms)
            },
        )
    else:
        vs = None

    # -----------------------------------------------------------------------
    # 3. Build pocket constraints
    # -----------------------------------------------------------------------
    constraints: List = []
    if args.pocket_binder:
        n_con = len(args.pocket_binder)
        contacts_list = args.pocket_contacts or []
        max_dist_list = args.pocket_max_dist or []
        no_force_list = args.no_pocket_force or []

        for i in range(n_con):
            binder = args.pocket_binder[i]

            if i < len(contacts_list):
                try:
                    contacts = _parse_pocket_contacts(contacts_list[i])
                except ValueError as exc:
                    parser.error(f"--pocket-contacts #{i + 1}: {exc}")
            else:
                contacts = []

            max_dist = max_dist_list[i] if i < len(max_dist_list) else 6.0
            force = not (i < len(no_force_list) and no_force_list[i])

            constraints.append(PocketConstraint(
                binder=binder,
                contacts=contacts,
                force=force,
                max_distance=max_dist,
            ))

    # -----------------------------------------------------------------------
    # 4. Build properties
    # -----------------------------------------------------------------------
    properties: List = []
    if args.affinity:
        for chain in args.affinity:
            properties.append(AffinityProperty(binder=chain))

    # -----------------------------------------------------------------------
    # 5. Build BoltzParams
    # -----------------------------------------------------------------------
    bp = BoltzParams(
        recycling_steps=args.recycling_steps,
        diffusion_samples=args.diffusion_samples,
        sampling_steps=args.sampling_steps,
        use_msa_server=args.use_msa_server,
        model=args.model,
        output_format=args.output_format,
        use_potentials=args.use_potentials,
        seed=args.seed,
        affinity_mw_correction=args.affinity_mw_correction,
    )

    # -----------------------------------------------------------------------
    # 6. Generate YAML(s)
    # -----------------------------------------------------------------------
    c_arg = constraints if constraints else None
    p_arg = properties if properties else None

    if vs is not None:
        yaml_files = build_yaml_variants(entities, c_arg, p_arg, vs, args.name)
    else:
        yaml_content = build_yaml(entities, c_arg, p_arg)
        yaml_files = [(f"{args.name}.yaml", yaml_content)]

    n_yamls = len(yaml_files)

    # -----------------------------------------------------------------------
    # 7. --stdout path: print and exit
    # -----------------------------------------------------------------------
    if args.stdout:
        if n_yamls > 1:
            parser.error(
                f"--stdout only supports single-YAML output but {n_yamls} YAMLs would "
                "be generated (variants detected). Use --out-dir for screening runs."
            )
        print(yaml_files[0][1], end="")
        return

    # -----------------------------------------------------------------------
    # 8. Write YAMLs to <out_dir>/input/
    # -----------------------------------------------------------------------
    out_dir = Path(args.out_dir).resolve()
    input_dir = out_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    for filename, content in yaml_files:
        (input_dir / filename).write_text(content)

    # -----------------------------------------------------------------------
    # 9. GPU + time auto-recommendation
    # -----------------------------------------------------------------------
    total_tokens = estimate_tokens(entities)
    gpu_rec = recommend_gpu(total_tokens)
    time_rec = recommend_time(total_tokens, bp, n_variants=n_yamls, gpu_rec=gpu_rec)

    # Apply extra boltz flags from GPU recommendation (e.g. --no_kernels for V100)
    if gpu_rec.get("boltz_extra_flags"):
        for flag in gpu_rec["boltz_extra_flags"]:
            if flag == "--no_kernels":
                bp.no_kernels = True

    # User overrides take priority over recommendations
    partition = args.partition or time_rec["partition"]
    time_str = args.time or time_rec["time"]
    gpu_sbatch = args.gpu or gpu_rec["gpu_sbatch"]
    mem = args.mem or gpu_rec["mem"]

    slurm_params = SlurmParams(
        job_name=args.name,
        time=time_str,
        partition=partition,
        gpus_per_node=gpu_sbatch,
        nodes=1,
        cpus_per_task=args.cpus,
        mem=mem,
    )

    # -----------------------------------------------------------------------
    # 10. Write job.sh
    # -----------------------------------------------------------------------
    job_script = build_job_script(
        job_dir=str(out_dir),
        cache_dir=BOLTZ_CACHE_DIR,
        boltz_params=bp,
        slurm_params=slurm_params,
        python_module=PYTHON_MODULE,
    )
    job_sh = out_dir / "job.sh"
    job_sh.write_text(job_script)
    job_sh.chmod(0o755)

    # -----------------------------------------------------------------------
    # 11. Copy boltz_tools package into out-dir so the cluster can import it
    # for log parsing (PYTHONPATH="$_job_dir" in job.sh cleanup trap).
    # -----------------------------------------------------------------------
    _tools_src = Path(__file__).resolve().parent
    _tools_dst = out_dir / "boltz_tools"
    if _tools_dst.exists():
        shutil.rmtree(_tools_dst)
    shutil.copytree(_tools_src, _tools_dst)

    # -----------------------------------------------------------------------
    # 12. Summary
    # -----------------------------------------------------------------------
    all_warnings = gpu_rec.get("warnings", []) + time_rec.get("warnings", [])
    for w in all_warnings:
        print(f"Warning: {w}", file=sys.stderr)

    print(f"Wrote {n_yamls} YAML(s) to {input_dir}/")
    print(f"Wrote job.sh to {out_dir}/")
    print(f"Copied boltz_tools/ to {out_dir}/")
    print(
        f"  Slurm: partition={partition}  time={time_str}  "
        f"gpu={gpu_sbatch}  mem={mem}"
    )
    print(f"\nTo submit:")
    print(f"  hpc-submit {out_dir}/job.sh")
