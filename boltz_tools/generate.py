"""Boltz input and job script generation.

Pure Python, no cluster dependencies. This module is portable
and will eventually run on the local laptop as well.
"""

import copy
from dataclasses import dataclass, field
from itertools import product as iterproduct
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ProteinEntity:
    ids: List[str]
    sequence: str


@dataclass
class DnaEntity:
    ids: List[str]
    sequence: str


@dataclass
class RnaEntity:
    ids: List[str]
    sequence: str


@dataclass
class LigandEntity:
    ids: List[str]
    smiles: Optional[str] = None
    ccd: Optional[str] = None


@dataclass
class PocketConstraint:
    binder: str
    contacts: List[Tuple[str, int]]
    force: bool = True
    max_distance: float = 6.0


@dataclass
class AffinityProperty:
    binder: str


@dataclass
class TemplateEntry:
    """A structural template for guided prediction."""
    file_path: str              # absolute path to CIF/PDB
    file_format: str            # "cif" or "pdb"
    chain_ids: Optional[List[str]] = None     # input chains (from entities)
    template_ids: Optional[List[str]] = None  # chains in the template file
    force: bool = False
    threshold: Optional[float] = None         # required when force=True


@dataclass
class BoltzParams:
    recycling_steps: int = 10
    diffusion_samples: int = 10
    sampling_steps: int = 200
    use_msa_server: bool = True
    model: str = "boltz2"
    output_format: str = "mmcif"
    use_potentials: bool = False
    override: bool = False
    seed: Optional[int] = None
    affinity_mw_correction: bool = False
    no_kernels: bool = False


@dataclass
class SlurmParams:
    job_name: str = "boltz"
    time: str = "00:30:00"
    partition: str = "gpu"
    gpus_per_node: str = "1"
    nodes: int = 1
    cpus_per_task: int = 7
    mem: str = "16GB"


@dataclass
class VariantSet:
    """Describes entities with multiple values for screening."""
    variants: Dict[int, List[str]]  # entity_index -> all variant values
    variant_names: Dict[int, List[str]] = field(default_factory=dict)  # entity_index -> human-readable names


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

# Standard 20 amino acids
PROTEIN_STANDARD = set("ACDEFGHIKLMNPQRSTVWY")
# Non-standard but technically valid (warn, don't reject)
PROTEIN_WARN = {"U", "O", "B", "J", "X", "Z"}
# Full allowed alphabet
PROTEIN_ALPHABET = PROTEIN_STANDARD | PROTEIN_WARN

DNA_ALPHABET = set("ACGTRYSWKMBDHVN")  # IUPAC ambiguity codes
RNA_ALPHABET = set("ACGURYSWKMBDHVN")

_WARN_NAMES = {
    "U": "selenocysteine",
    "O": "pyrrolysine",
    "B": "Asn/Asp ambiguity",
    "Z": "Gln/Glu ambiguity",
    "J": "Leu/Ile ambiguity",
    "X": "unknown residue",
}


def validate_protein_sequence(seq: str) -> Tuple[str, Optional[str]]:
    """Validate a protein sequence.

    Returns (cleaned_sequence, error_or_None).
    Also returns warnings as error string prefixed with "Warning:" —
    caller should display but not reject.
    """
    cleaned = "".join(seq.upper().split())
    if not cleaned:
        return "", "Empty sequence."
    bad = set(cleaned) - PROTEIN_ALPHABET
    if bad:
        return cleaned, (
            f"Invalid residue(s): {','.join(sorted(bad))}. "
            f"Standard amino acids: {' '.join(sorted(PROTEIN_STANDARD))}"
        )
    # Warn about non-standard residues
    unusual = set(cleaned) & PROTEIN_WARN
    if unusual:
        details = ", ".join(
            f"{r} ({_WARN_NAMES.get(r, '?')})" for r in sorted(unusual)
        )
        return cleaned, f"Warning: non-standard residue(s): {details}. Boltz may not support these."
    return cleaned, None


def validate_dna_sequence(seq: str) -> Tuple[str, Optional[str]]:
    """Validate a DNA sequence. Returns (cleaned, error_or_None)."""
    cleaned = "".join(seq.upper().split())
    if not cleaned:
        return "", "Empty sequence."
    bad = set(cleaned) - DNA_ALPHABET
    if bad:
        return cleaned, (
            f"Invalid nucleotide(s): {','.join(sorted(bad))}. "
            f"Allowed: {' '.join(sorted(DNA_ALPHABET))}"
        )
    return cleaned, None


def validate_rna_sequence(seq: str) -> Tuple[str, Optional[str]]:
    """Validate an RNA sequence. Returns (cleaned, error_or_None)."""
    cleaned = "".join(seq.upper().split())
    if not cleaned:
        return "", "Empty sequence."
    bad = set(cleaned) - RNA_ALPHABET
    if bad:
        return cleaned, (
            f"Invalid nucleotide(s): {','.join(sorted(bad))}. "
            f"Allowed: {' '.join(sorted(RNA_ALPHABET))}"
        )
    return cleaned, None


def validate_ccd(code: str) -> Tuple[str, Optional[str]]:
    """Validate a CCD (Chemical Component Dictionary) code.

    CCD codes are 1-3 character alphanumeric identifiers.
    Returns (cleaned_code, error_or_None).
    """
    cleaned = code.strip().upper()
    if not cleaned:
        return "", "Empty CCD code."
    if not cleaned.isalnum():
        return cleaned, "CCD code must be alphanumeric."
    if len(cleaned) > 3:
        return cleaned, f"CCD code too long ({len(cleaned)} chars, max 3)."
    return cleaned, None


def validate_smiles(smiles: str) -> Tuple[str, Optional[str]]:
    """Validate a SMILES string using RDKit if available, basic checks otherwise.

    Returns (smiles, error_or_None).
    """
    smiles = smiles.strip()
    if not smiles:
        return "", "Empty SMILES."
    try:
        from rdkit import Chem, RDLogger
        # Suppress RDKit parse-error messages on stderr
        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles, "Invalid SMILES (RDKit could not parse it)."
        return smiles, None
    except ImportError:
        # RDKit not available — do basic sanity check
        if len(smiles) > 5000:
            return smiles, f"SMILES suspiciously long ({len(smiles)} chars)."
        return smiles, None


def parse_fasta(text: str) -> List[Tuple[str, str]]:
    """Parse FASTA-formatted text into (name, sequence) pairs.

    Headers start with '>'; the first whitespace-delimited word is the name.
    Sequence lines are concatenated until the next '>' or EOF.
    Returns empty list if no '>' headers are found (not FASTA format).
    """
    entries: List[Tuple[str, str]] = []
    current_name: Optional[str] = None
    current_seq: List[str] = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            # Save previous entry
            if current_name is not None:
                entries.append((current_name, "".join(current_seq)))
            # Parse header: first word after '>'
            header = line[1:].strip()
            current_name = header.split()[0] if header else f"seq{len(entries) + 1}"
            current_seq = []
        else:
            if current_name is not None:
                current_seq.append(line)

    # Save last entry
    if current_name is not None:
        entries.append((current_name, "".join(current_seq)))

    return entries


def parse_variant_values(
    text: str,
    entity,
) -> Tuple[List[str], List[str]]:
    """Validate each line in *text* against the entity type.

    Returns (valid_values, errors).  Warnings (non-standard residues) are
    kept as valid; hard errors are rejected with a message in *errors*.
    """
    valid: List[str] = []
    errors: List[str] = []

    for lineno, raw in enumerate(text.splitlines(), 1):
        raw = raw.strip()
        if not raw:
            continue

        if isinstance(entity, ProteinEntity):
            cleaned, err = validate_protein_sequence(raw)
            if err and not err.startswith("Warning:"):
                errors.append(f"Line {lineno}: {err}")
            else:
                valid.append(cleaned)
        elif isinstance(entity, (DnaEntity,)):
            cleaned, err = validate_dna_sequence(raw)
            if err:
                errors.append(f"Line {lineno}: {err}")
            else:
                valid.append(cleaned)
        elif isinstance(entity, (RnaEntity,)):
            cleaned, err = validate_rna_sequence(raw)
            if err:
                errors.append(f"Line {lineno}: {err}")
            else:
                valid.append(cleaned)
        elif isinstance(entity, LigandEntity):
            if entity.ccd is not None:
                cleaned, err = validate_ccd(raw)
                if err:
                    errors.append(f"Line {lineno}: {err}")
                else:
                    valid.append(cleaned)
            else:
                cleaned, err = validate_smiles(raw)
                if err:
                    errors.append(f"Line {lineno}: {err}")
                else:
                    valid.append(cleaned)
        else:
            errors.append(f"Line {lineno}: unsupported entity type")

    return valid, errors


# ---------------------------------------------------------------------------
# Template chain extraction
# ---------------------------------------------------------------------------

def extract_template_chains(path: str) -> List[str]:
    """Extract chain IDs from a CIF/PDB structure file using gemmi.

    Returns list of chain names from the first model.
    Returns empty list on failure.
    """
    try:
        import gemmi
        st = gemmi.read_structure(path)
        chains = []
        if st:
            for chain in st[0]:  # first model
                if chain.name not in chains:
                    chains.append(chain.name)
        return chains
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Token estimation and GPU recommendation
# ---------------------------------------------------------------------------

def _estimate_ligand_atoms(smiles: str) -> int:
    """Estimate heavy-atom count from a SMILES string.

    Crude heuristic: count uppercase letters (each is an atom symbol start),
    then subtract explicit hydrogens ('H' not followed by a lowercase letter).
    """
    count = 0
    for ch in smiles:
        if ch.isupper():
            count += 1
    # Subtract explicit H atoms (uppercase H not followed by lowercase)
    i = 0
    h_count = 0
    while i < len(smiles):
        if smiles[i] == 'H' and (i + 1 >= len(smiles) or not smiles[i + 1].islower()):
            h_count += 1
        i += 1
    count -= h_count
    return max(1, count)


def estimate_tokens(entities: list) -> int:
    """Estimate total token count from a list of entities.

    Tokens drive GPU memory consumption (roughly quadratic scaling).
    - Protein/DNA/RNA: len(sequence) * number_of_chains
    - Ligand: estimated heavy atoms * number_of_chains
    """
    total = 0
    for e in entities:
        n_chains = len(e.ids)
        if isinstance(e, (ProteinEntity, DnaEntity, RnaEntity)):
            total += len(e.sequence) * n_chains
        elif isinstance(e, LigandEntity):
            if e.smiles is not None:
                total += _estimate_ligand_atoms(e.smiles) * n_chains
            else:
                # CCD ligand — assume ~30 heavy atoms as a conservative default
                total += 30 * n_chains
    return total


def recommend_gpu(total_tokens: int) -> Dict:
    """Recommend GPU type, memory, and extra boltz flags based on token count.

    Tiers are read from ~/.config/boltz-setup/config.yaml (gpu_tiers).

    Returns a dict with keys:
        gpu_sbatch: str          — e.g. "v100:1" (passed to --gpus-per-node)
        mem: str                 — system memory, e.g. "16GB"
        boltz_extra_flags: list[str] — e.g. ["--no_kernels"]
        warnings: list[str]     — user-facing messages
    """
    from .cluster import GPU_TIERS
    warnings: List[str] = []

    tier = GPU_TIERS[-1]  # fallback: largest tier
    for t in GPU_TIERS:
        if total_tokens <= t["max_tokens"]:
            tier = t
            break

    if tier.get("warn"):
        warnings.append(
            f"Very large job ({total_tokens} tokens). "
            "Consider splitting into smaller complexes if it OOMs."
        )

    return {
        "gpu_sbatch": tier["gpu_sbatch"],
        "mem": tier["mem"],
        "boltz_extra_flags": tier.get("extra_flags", []),
        "warnings": warnings,
    }


def recommend_time(total_tokens: int, boltz_params: BoltzParams, n_variants: int = 1,
                   gpu_rec: Optional[Dict] = None) -> Dict:
    """Recommend wall-time and partition based on token count and boltz params.

    Returns a dict with keys:
        estimated_minutes: int — raw estimate before safety margin
        time: str              — HH:MM:SS with 1.5x safety margin
        partition: str         — gpushort / gpumedium / gpulong
        warnings: list[str]
    """
    # Base minutes per diffusion sample at 3 recycling steps, 200 sampling steps
    if total_tokens < 300:
        base_per_sample = 1.0
    elif total_tokens < 700:
        base_per_sample = 3.0
    elif total_tokens < 1500:
        base_per_sample = 8.0
    elif total_tokens < 2500:
        base_per_sample = 20.0
    else:
        base_per_sample = 45.0

    # Scale by recycling and sampling steps relative to baselines
    recycle_factor = boltz_params.recycling_steps / 3.0
    sampling_factor = boltz_params.sampling_steps / 200.0

    compute_min = (
        base_per_sample
        * recycle_factor
        * sampling_factor
        * boltz_params.diffusion_samples
    )

    # Fixed overhead: model loading (~5 min) + MSA server (~10 min if enabled)
    overhead = 5.0
    if boltz_params.use_msa_server:
        overhead += 10.0

    # Scale compute time by number of variants (overhead is shared)
    raw_minutes = compute_min * n_variants + overhead

    # 1.5x safety margin, minimum 15 minutes
    safe_minutes = max(15, int(raw_minutes * 1.5))

    # Round up to nice increments
    if safe_minutes <= 30:
        safe_minutes = 30
    elif safe_minutes <= 60:
        safe_minutes = 60
    elif safe_minutes <= 120:
        # Round up to nearest 30
        safe_minutes = ((safe_minutes + 29) // 30) * 30
    else:
        # Round up to nearest hour
        safe_minutes = ((safe_minutes + 59) // 60) * 60

    # Select partition — pick the shortest one whose max_hours covers safe_minutes
    from .cluster import GPU_TIERS as _GPU_TIERS, PARTITIONS as _PARTITIONS
    warnings: List[str] = []
    partition = None
    for p in _PARTITIONS:
        if safe_minutes <= p["max_hours"] * 60:
            partition = p["name"]
            break
    if partition is None:
        last_p = _PARTITIONS[-1]
        partition = last_p["name"]
        max_minutes = last_p["max_hours"] * 60
        if safe_minutes > max_minutes:
            safe_minutes = max_minutes
            warnings.append(
                f"Estimated time (~{int(raw_minutes)} min) exceeds "
                f"{partition} limit ({last_p['max_hours']}h). "
                f"Setting {last_p['max_hours']}h — job may not complete. "
                "Consider reducing diffusion_samples or splitting the job."
            )

    # Check GPU-partition compatibility; upgrade GPU if needed
    if gpu_rec is not None:
        gpu_sbatch = gpu_rec["gpu_sbatch"]
        selected_p = next((p for p in _PARTITIONS if p["name"] == partition), None)
        if selected_p and gpu_sbatch not in selected_p.get("gpus", []):
            # Find the cheapest GPU tier available on this partition that fits the job
            fallback_tier = None
            for t in _GPU_TIERS:
                if t["gpu_sbatch"] in selected_p.get("gpus", []) and total_tokens <= t["max_tokens"]:
                    fallback_tier = t
                    break
            if fallback_tier and fallback_tier["gpu_sbatch"] != gpu_sbatch:
                warnings.append(
                    f"{gpu_sbatch} not available on {partition} — "
                    f"upgrading to {fallback_tier['gpu_sbatch']}."
                )
                gpu_rec["gpu_sbatch"] = fallback_tier["gpu_sbatch"]
                gpu_rec["mem"] = fallback_tier["mem"]
                gpu_rec["boltz_extra_flags"] = fallback_tier.get("extra_flags", [])

    hours = safe_minutes // 60
    mins = safe_minutes % 60
    if hours >= 24:
        days = hours // 24
        hours = hours % 24
        time_str = f"{days}-{hours:02d}:{mins:02d}:00"
    else:
        time_str = f"{hours:02d}:{mins:02d}:00"

    return {
        "estimated_minutes": int(raw_minutes),
        "time": time_str,
        "partition": partition,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# YAML generation
# ---------------------------------------------------------------------------

def _format_id_list(ids: List[str]) -> str:
    """Format chain IDs as inline YAML list: [A] or [A,B]."""
    return "[" + ",".join(ids) + "]"


def build_yaml(
    entities: list,
    constraints: Optional[list] = None,
    properties: Optional[list] = None,
    templates: Optional[list] = None,
) -> str:
    """Build a Boltz-compatible YAML input string."""
    lines = ["sequences:"]

    for e in entities:
        if isinstance(e, ProteinEntity):
            lines.append("  - protein:")
            lines.append(f"      id: {_format_id_list(e.ids)}")
            lines.append(f"      sequence: {e.sequence}")
        elif isinstance(e, DnaEntity):
            lines.append("  - dna:")
            lines.append(f"      id: {_format_id_list(e.ids)}")
            lines.append(f"      sequence: {e.sequence}")
        elif isinstance(e, RnaEntity):
            lines.append("  - rna:")
            lines.append(f"      id: {_format_id_list(e.ids)}")
            lines.append(f"      sequence: {e.sequence}")
        elif isinstance(e, LigandEntity):
            lines.append("  - ligand:")
            lines.append(f"      id: {_format_id_list(e.ids)}")
            if e.smiles is not None:
                lines.append(f"      smiles: '{e.smiles}'")
            elif e.ccd is not None:
                lines.append(f"      ccd: {e.ccd}")

    if constraints:
        lines.append("constraints:")
        for c in constraints:
            if isinstance(c, PocketConstraint):
                lines.append("  - pocket:")
                lines.append(f"      binder: {c.binder}")
                contacts_parts = ", ".join(
                    f"[ {chain}, {res} ]" for chain, res in c.contacts
                )
                lines.append(f"      contacts: [{contacts_parts}]")
                lines.append(f"      force: {'true' if c.force else 'false'}")
                lines.append(f"      max_distance: {c.max_distance}")

    if properties:
        lines.append("properties:")
        for p in properties:
            if isinstance(p, AffinityProperty):
                lines.append("  - affinity:")
                lines.append(f"      binder: {p.binder}")

    if templates:
        lines.append("templates:")
        for t in templates:
            lines.append(f"  - {t.file_format}: {t.file_path}")
            if t.chain_ids:
                lines.append(f"    chain_id: {_format_id_list(t.chain_ids)}")
            if t.template_ids:
                lines.append(f"    template_id: {_format_id_list(t.template_ids)}")
            if t.force:
                lines.append(f"    force: true")
                lines.append(f"    threshold: {t.threshold}")

    return "\n".join(lines) + "\n"


def _entity_filename_tag(entity) -> str:
    """Short filename tag for a non-varied entity.

    Protein chain A        -> pA
    Protein chains B,C     -> pBC
    Ligand (SMILES) chain D -> lD
    Ligand (CCD FAD) chain E -> FAD
    DNA chain F            -> dF
    RNA chain G            -> rG
    """
    chains = "".join(entity.ids)
    if isinstance(entity, ProteinEntity):
        return f"p{chains}"
    elif isinstance(entity, DnaEntity):
        return f"d{chains}"
    elif isinstance(entity, RnaEntity):
        return f"r{chains}"
    elif isinstance(entity, LigandEntity):
        if entity.ccd is not None:
            return entity.ccd
        return f"l{chains}"
    return f"x{chains}"


def _sanitize_name(name: str, max_len: int = 50) -> str:
    """Sanitize a name for use in filenames.

    Replaces non-alphanumeric/hyphen/underscore characters with '_',
    strips leading/trailing underscores, and truncates.
    """
    import re
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    sanitized = sanitized.strip('_')
    return sanitized[:max_len] if sanitized else "unnamed"


def _variant_filename_tag(entity, value: str, variant_num: int,
                          name: str = None) -> str:
    """Short filename tag for a varied entity at a specific variant.

    If *name* is provided (e.g. from a FASTA header), it is sanitized and
    used as the tag instead of the generic pAv1 pattern.

    CCD variants use the code directly (FAD, MG, ATP).
    Others use chain + v + index (pAv1, lBv2, dCv3).
    """
    if name is not None:
        return _sanitize_name(name)
    chains = "".join(entity.ids)
    if isinstance(entity, LigandEntity) and entity.ccd is not None:
        return value  # CCD code is already short and descriptive
    if isinstance(entity, ProteinEntity):
        return f"p{chains}v{variant_num}"
    elif isinstance(entity, DnaEntity):
        return f"d{chains}v{variant_num}"
    elif isinstance(entity, RnaEntity):
        return f"r{chains}v{variant_num}"
    elif isinstance(entity, LigandEntity):
        return f"l{chains}v{variant_num}"
    return f"x{chains}v{variant_num}"


def build_yaml_variants(
    entities: list,
    constraints: Optional[list],
    properties: Optional[list],
    variant_set: "VariantSet",
    job_name: str,
    templates: Optional[list] = None,
) -> List[Tuple[str, str]]:
    """Build one YAML per combination of variant values (cartesian product).

    Returns [(filename, yaml_content), ...].
    Filenames encode all entities, e.g. ``job_pA_lBv1_FAD.yaml``.
    """
    indices = sorted(variant_set.variants.keys())
    value_lists = [variant_set.variants[i] for i in indices]
    index_ranges = [range(len(vl)) for vl in value_lists]
    idx_to_pos = {idx: pos for pos, idx in enumerate(indices)}
    varied_set = set(indices)

    results: List[Tuple[str, str]] = []
    for combo_indices in iterproduct(*index_ranges):
        ents = list(entities)
        tags: List[str] = []

        for ent_idx, entity in enumerate(entities):
            if ent_idx in varied_set:
                pos = idx_to_pos[ent_idx]
                vi = combo_indices[pos]
                val = value_lists[pos][vi]
                # Swap value in a shallow copy
                varied = copy.copy(entity)
                if isinstance(varied, (ProteinEntity, DnaEntity, RnaEntity)):
                    varied.sequence = val
                elif isinstance(varied, LigandEntity):
                    if varied.ccd is not None:
                        varied.ccd = val
                    else:
                        varied.smiles = val
                ents[ent_idx] = varied
                # Use human-readable name if available
                vname = None
                if variant_set.variant_names.get(ent_idx):
                    names_list = variant_set.variant_names[ent_idx]
                    if vi < len(names_list):
                        vname = names_list[vi]
                tags.append(_variant_filename_tag(entity, val, vi + 1, name=vname))
            else:
                tags.append(_entity_filename_tag(entity))

        desc = "_".join(tags)
        yaml_content = build_yaml(ents, constraints, properties, templates)
        filename = f"{job_name}_{desc}.yaml"
        results.append((filename, yaml_content))

    return results


# ---------------------------------------------------------------------------
# Job script generation
# ---------------------------------------------------------------------------

def _boltz_flags(bp: BoltzParams) -> List[str]:
    """Build the boltz predict flags as a list of strings."""
    parts = []
    if bp.use_msa_server:
        parts.append("--use_msa_server")
    parts.append(f"--recycling_steps {bp.recycling_steps}")
    parts.append(f"--diffusion_samples {bp.diffusion_samples}")
    parts.append(f"--sampling_steps {bp.sampling_steps}")
    if bp.model != "boltz2":
        parts.append(f"--model {bp.model}")
    if bp.output_format != "mmcif":
        parts.append(f"--output_format {bp.output_format}")
    if bp.use_potentials:
        parts.append("--use_potentials")
    if bp.override:
        parts.append("--override")
    if bp.seed is not None:
        parts.append(f"--seed {bp.seed}")
    if bp.affinity_mw_correction:
        parts.append("--affinity_mw_correction")
    if bp.no_kernels:
        parts.append("--no_kernels")
    return parts


def _boltz_tools_root() -> str:
    """Return the parent directory of the boltz_tools package (for PYTHONPATH)."""
    return str(Path(__file__).resolve().parent.parent)


def build_job_script(
    job_dir: str,
    cache_dir: str,
    boltz_params: BoltzParams,
    slurm_params: SlurmParams,
    python_module: Optional[str] = None,
) -> str:
    """Build a Slurm job script for boltz predict."""
    if python_module is None:
        from .cluster import PYTHON_MODULE
        python_module = PYTHON_MODULE
    flag_list = _boltz_flags(boltz_params)

    # Build multi-line boltz command for readability
    boltz_parts = [
        "boltz predict ./input/",
        "    --out_dir ./output/",
        f"    --cache {cache_dir}",
    ]
    for f in flag_list:
        boltz_parts.append(f"    {f}")
    boltz_cmd = " \\\n".join(boltz_parts)

    return f"""#!/bin/bash
#SBATCH --job-name={slurm_params.job_name}
#SBATCH --time={slurm_params.time}
#SBATCH --gpus-per-node={slurm_params.gpus_per_node}
#SBATCH --partition={slurm_params.partition}
#SBATCH --nodes={slurm_params.nodes}
#SBATCH --cpus-per-task={slurm_params.cpus_per_task}
#SBATCH --mem={slurm_params.mem}
#SBATCH --output=slurm-%j.out

# Resolve job directory: use Slurm's submit dir (most reliable),
# falling back to script location for non-Slurm execution.
_job_dir="${{SLURM_SUBMIT_DIR:-$(cd "$(dirname "$(readlink -f "$0")")" && pwd)}}"

scontrol show job $SLURM_JOB_ID

# Rename script to include job ID (job.sh -> job_12345.sh)
_self="$(readlink -f "$0")"
_stem="${{_self%.sh}}"
mv "$_self" "${{_stem}}_${{SLURM_JOB_ID}}.sh" 2>/dev/null || true

set -x
set -e
PS4='$(date +%H:%M:%S) '

# Exit trap: report timing and status, then generate clean log
cleanup() {{
    local rc=$?
    set +x
    echo "========================================="
    if [ $rc -eq 0 ]; then
        echo "Job completed successfully in ${{SECONDS}}s"
    else
        echo "Job FAILED (exit code $rc) after ${{SECONDS}}s"
    fi
    echo "========================================="
    # Generate clean log (best-effort, never fail the job)
    # boltz_tools/ is shipped alongside job.sh by boltz-setup-yaml
    PYTHONPATH="$_job_dir" python -m boltz_tools log "$_job_dir" || true
}}
trap cleanup EXIT

echo "start $(date)"

module purge
module load {python_module}

# GPU diagnostics
nvidia-smi

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$_job_dir"

{boltz_cmd}

echo "done $(date)"
"""


# ---------------------------------------------------------------------------
# Directory setup
# ---------------------------------------------------------------------------

def parse_boltz_yaml(yaml_content: str) -> Tuple[list, list, list, list, set]:
    """Parse a boltz YAML string back into (entities, constraints, properties, templates, used_ids).

    Uses PyYAML (available via boltz's dependencies).
    """
    try:
        import yaml
    except ImportError:
        raise RuntimeError(
            "PyYAML is required for YAML parsing. "
            "Install it with: pip install pyyaml"
        )

    data = yaml.safe_load(yaml_content)
    if not isinstance(data, dict):
        return [], [], [], [], set()

    entities: List = []
    used_ids: set = set()

    # Parse sequences/entities
    for seq_entry in data.get("sequences", []):
        if not isinstance(seq_entry, dict):
            continue

        if "protein" in seq_entry:
            p = seq_entry["protein"]
            ids = p.get("id", [])
            if isinstance(ids, str):
                ids = [ids]
            sequence = p.get("sequence", "")
            entities.append(ProteinEntity(ids=list(ids), sequence=sequence))
            used_ids.update(ids)

        elif "dna" in seq_entry:
            d = seq_entry["dna"]
            ids = d.get("id", [])
            if isinstance(ids, str):
                ids = [ids]
            sequence = d.get("sequence", "")
            entities.append(DnaEntity(ids=list(ids), sequence=sequence))
            used_ids.update(ids)

        elif "rna" in seq_entry:
            r = seq_entry["rna"]
            ids = r.get("id", [])
            if isinstance(ids, str):
                ids = [ids]
            sequence = r.get("sequence", "")
            entities.append(RnaEntity(ids=list(ids), sequence=sequence))
            used_ids.update(ids)

        elif "ligand" in seq_entry:
            lig = seq_entry["ligand"]
            ids = lig.get("id", [])
            if isinstance(ids, str):
                ids = [ids]
            smiles = lig.get("smiles")
            ccd = lig.get("ccd")
            # Convert smiles to string (may be parsed as other types)
            if smiles is not None:
                smiles = str(smiles)
            if ccd is not None:
                ccd = str(ccd)
            entities.append(LigandEntity(ids=list(ids), smiles=smiles, ccd=ccd))
            used_ids.update(ids)

    # Parse constraints
    constraints: List = []
    for c_entry in data.get("constraints", []):
        if not isinstance(c_entry, dict):
            continue
        if "pocket" in c_entry:
            pc = c_entry["pocket"]
            binder = str(pc.get("binder", ""))
            raw_contacts = pc.get("contacts", [])
            contacts = []
            for contact in raw_contacts:
                if isinstance(contact, (list, tuple)) and len(contact) == 2:
                    contacts.append((str(contact[0]), int(contact[1])))
            force = bool(pc.get("force", True))
            max_distance = float(pc.get("max_distance", 6.0))
            constraints.append(PocketConstraint(
                binder=binder, contacts=contacts,
                force=force, max_distance=max_distance,
            ))

    # Parse properties
    properties: List = []
    for p_entry in data.get("properties", []):
        if not isinstance(p_entry, dict):
            continue
        if "affinity" in p_entry:
            aff = p_entry["affinity"]
            binder = str(aff.get("binder", ""))
            properties.append(AffinityProperty(binder=binder))

    # Parse templates
    templates: List = []
    for t_entry in data.get("templates", []):
        if not isinstance(t_entry, dict):
            continue
        file_path = None
        file_format = None
        for fmt in ("cif", "pdb"):
            if fmt in t_entry:
                file_path = str(t_entry[fmt])
                file_format = fmt
                break
        if file_path is None:
            continue
        # Skip templates whose files no longer exist
        if not Path(file_path).is_file():
            print(f"  Warning: skipping template (file not found): {file_path}")
            continue
        chain_ids = t_entry.get("chain_id")
        if isinstance(chain_ids, str):
            chain_ids = [chain_ids]
        elif isinstance(chain_ids, list):
            chain_ids = [str(c) for c in chain_ids]
        template_ids = t_entry.get("template_id")
        if isinstance(template_ids, str):
            template_ids = [template_ids]
        elif isinstance(template_ids, list):
            template_ids = [str(c) for c in template_ids]
        force = bool(t_entry.get("force", False))
        threshold = t_entry.get("threshold")
        if threshold is not None:
            threshold = float(threshold)
        templates.append(TemplateEntry(
            file_path=file_path,
            file_format=file_format,
            chain_ids=chain_ids,
            template_ids=template_ids,
            force=force,
            threshold=threshold,
        ))

    return entities, constraints, properties, templates, used_ids


def setup_job(
    base_dir: str,
    job_name: str,
    yaml_content: str,
    script_content: str,
) -> Path:
    """Create job directory structure and write input files.

    Returns the job directory path.
    """
    job_dir = Path(base_dir) / job_name
    input_dir = job_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    # Remove stale YAML files from previous runs
    for old in input_dir.glob("*.yaml"):
        old.unlink()

    yaml_path = input_dir / f"{job_name}.yaml"
    yaml_path.write_text(yaml_content)

    script_path = job_dir / "job.sh"
    script_path.write_text(script_content)

    return job_dir


def setup_job_variants(
    base_dir: str,
    job_name: str,
    yaml_files: List[Tuple[str, str]],
    script_content: str,
) -> Path:
    """Create job directory and write multiple YAML input files.

    yaml_files: list of (filename, yaml_content) tuples.
    Returns the job directory path.
    """
    job_dir = Path(base_dir) / job_name
    input_dir = job_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    # Remove stale YAML files from previous runs
    for old in input_dir.glob("*.yaml"):
        old.unlink()

    for filename, content in yaml_files:
        (input_dir / filename).write_text(content)

    script_path = job_dir / "job.sh"
    script_path.write_text(script_content)

    return job_dir


def setup_job_resume(
    job_dir: str,
    yaml_files: List[Tuple[str, str]],
    script_content: str,
) -> Tuple[Path, str]:
    """Add YAML files to an existing job directory without removing old ones.

    Writes the job script as job_N.sh where N is the next available number.
    Returns (job_dir_path, script_filename).
    """
    job_path = Path(job_dir)
    input_dir = job_path / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    # Write new YAMLs (rename with _N suffix on collision)
    written = []
    for filename, content in yaml_files:
        target = input_dir / filename
        if target.exists():
            stem = Path(filename).stem
            ext = Path(filename).suffix  # .yaml
            n = 1
            while (input_dir / f"{stem}_{n}{ext}").exists():
                n += 1
            filename = f"{stem}_{n}{ext}"
            target = input_dir / filename
        target.write_text(content)
        written.append(filename)

    # Find next available job script number
    n = 1
    while (job_path / f"job_{n}.sh").exists():
        n += 1
    script_name = f"job_{n}.sh"
    (job_path / script_name).write_text(script_content)

    return job_path, script_name
