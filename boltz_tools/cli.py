"""Interactive CLI for setting up Boltz prediction jobs.

Step-based wizard with go-back navigation, colors, and a review screen.
"""

import os
import readline  # noqa: F401 — enables arrow-key history for input()
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .generate import (
    ProteinEntity,
    DnaEntity,
    RnaEntity,
    LigandEntity,
    PocketConstraint,
    AffinityProperty,
    TemplateEntry,
    BoltzParams,
    SlurmParams,
    VariantSet,
    build_yaml,
    build_yaml_variants,
    build_job_script,
    setup_job_variants,
    setup_job_resume,
    estimate_tokens,
    recommend_gpu,
    recommend_time,
    extract_template_chains,
    validate_protein_sequence,
    validate_smiles,
    validate_ccd,
    validate_dna_sequence,
    validate_rna_sequence,
    parse_fasta,
    parse_boltz_yaml,
)
from .cluster import BOLTZ_JOBS_DIR, BOLTZ_CACHE_DIR, PYTHON_MODULE, submit_job
from .tui import (
    GoBack,
    JumpToStep,
    bold,
    dim,
    success,
    warning,
    error,
    info,
    highlight,
    value,
    hrule,
    section_header,
    step_progress_bar,
    print_banner,
    entity_table,
    print_yaml,
    print_bash,
    styled_prompt,
    styled_confirm,
    styled_choice,
    styled_select,
    print_error,
    print_warning,
    print_success,
    print_info,
    print_value,
    reset_back_hints,
    _show_back_hint,
)


# ---------------------------------------------------------------------------
# Step labels (for progress bar)
# ---------------------------------------------------------------------------

STEP_LABELS = ["Entities", "Constraints", "Properties", "Templates", "Boltz", "Slurm", "Review"]
NUM_STEPS = len(STEP_LABELS)


# ---------------------------------------------------------------------------
# Wizard state
# ---------------------------------------------------------------------------

@dataclass
class WizardState:
    """Mutable state passed through all wizard steps."""
    job_name: str
    base_dir: str
    entities: list = field(default_factory=list)
    used_ids: set = field(default_factory=set)
    constraints: list = field(default_factory=list)
    properties: list = field(default_factory=list)
    boltz: BoltzParams = field(default_factory=BoltzParams)
    slurm: Optional[SlurmParams] = None
    gpu_rec: Optional[dict] = None
    time_rec: Optional[dict] = None
    templates: list = field(default_factory=list)
    variant_set: Optional[VariantSet] = None
    resume_mode: bool = False


# ---------------------------------------------------------------------------
# Chain ID utilities
# ---------------------------------------------------------------------------

def _next_chain_ids(n: int, used_ids: set) -> list:
    """Return the next *n* available uppercase chain letters."""
    ids = []
    for code in range(ord('A'), ord('Z') + 1):
        if len(ids) == n:
            break
        letter = chr(code)
        if letter not in used_ids:
            ids.append(letter)
    if len(ids) < n:
        raise ValueError(
            f"Cannot auto-assign {n} chain IDs — only "
            f"{26 - len(used_ids)} letters remaining."
        )
    return ids


def _parse_copies_or_ids(val: str, used_ids: set):
    """Parse the copies/chain-IDs prompt value.

    Accepts:
        ""       -> 1 copy, auto-assign
        "3"      -> 3 copies, auto-assign
        "A,B"    -> custom chain IDs (comma-separated)
        "A B"    -> custom chain IDs (space-separated)
        "A, B"   -> also works

    Returns (ids: list[str], error: str|None).
    """
    val = val.strip()
    if not val:
        val = "1"

    if val.isdigit():
        n = int(val)
        if n < 1:
            return None, "Need at least 1 copy."
        try:
            ids = _next_chain_ids(n, used_ids)
        except ValueError as exc:
            return None, str(exc)
        return ids, None

    # Accept commas, spaces, or both as separators
    ids = [x.strip().upper() for x in val.replace(",", " ").split() if x.strip()]
    if not ids:
        return None, "No chain IDs provided."
    for cid in ids:
        if len(cid) != 1 or not cid.isalpha():
            return None, f"Invalid chain ID '{cid}' — must be a single letter A-Z."
    dupes = set(ids) & used_ids
    if dupes:
        return None, f"Chain ID(s) already used: {', '.join(sorted(dupes))}."
    if len(ids) != len(set(ids)):
        return None, "Duplicate chain IDs."
    return ids, None


# ---------------------------------------------------------------------------
# Multi-value input helper
# ---------------------------------------------------------------------------

def _is_file_path(s: str) -> bool:
    """True if *s* looks like a file path (starts with / or ./)."""
    return s.startswith("/") or s.startswith("./")


def _read_values(first_prompt, validator_fn):
    """Read one or more validated values interactively.

    First line uses styled_prompt (GoBack support). Subsequent lines use
    plain input("  > "). A blank line terminates.

    File loading triggers:
      - typing ``file`` (then prompted for path)
      - entering a path starting with ``/`` or ``./``

    *validator_fn* has signature (raw: str) -> (cleaned: str, error_or_None).
    Warnings (error starting with "Warning:") are displayed but kept.

    Returns (values, names) where *values* is list[str] and *names* is
    list[str] (from FASTA headers) or None for plain text / interactive input.
    Raises GoBack only from the first line.
    """
    values = []
    names = None

    # First value — GoBack-capable prompt
    raw = styled_prompt(first_prompt)
    if not raw:
        return [], None

    if raw.lower() == "file":
        return _read_values_from_file(validator_fn)
    if _is_file_path(raw):
        return _read_values_from_file(validator_fn, path=raw)

    cleaned, err = validator_fn(raw)
    if err and not err.startswith("Warning:"):
        print_error(err)
        return [], None
    if err:
        print_warning(err)
    values.append(cleaned)

    # Subsequent values — plain input
    while True:
        try:
            line = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            break
        if line.lower() == "file":
            file_vals, file_names = _read_values_from_file(validator_fn)
            values.extend(file_vals)
            if file_names is not None:
                # FASTA names only apply if all values came from this file
                if len(values) == len(file_vals):
                    names = file_names
            break
        if _is_file_path(line):
            file_vals, file_names = _read_values_from_file(validator_fn, path=line)
            values.extend(file_vals)
            if file_names is not None:
                if len(values) == len(file_vals):
                    names = file_names
            break
        cleaned, err = validator_fn(line)
        if err and not err.startswith("Warning:"):
            print_error(f"  skipped: {err}")
            continue
        if err:
            print_warning(err)
        values.append(cleaned)

    return values, names


def _read_values_from_file(validator_fn, path=None):
    """Load values from a file, validate each line. Returns (values, names).

    If *path* is None, prompts for one. Checks file existence before reading.

    If the file is in FASTA format (lines starting with '>'), sequences are
    parsed with headers used as names. Returns (values, names) where *names*
    is a list of FASTA header names (parallel to *values*), or None for
    plain-text files.
    """
    if path is None:
        try:
            path = input("  File path: ").strip()
        except (EOFError, KeyboardInterrupt):
            return [], None

    if not os.path.isfile(path):
        print_error(f"File not found: {path}")
        return [], None

    try:
        text = open(path).read()
    except OSError as exc:
        print_error(f"Cannot read file: {exc}")
        return [], None

    # Detect FASTA format: any non-empty line starting with '>'
    is_fasta = any(
        line.strip().startswith(">") for line in text.splitlines() if line.strip()
    )

    if is_fasta:
        entries = parse_fasta(text)
        if not entries:
            print_warning("FASTA file has no sequences.")
            return [], None

        values = []
        names = []
        n_errors = 0
        for name, seq in entries:
            cleaned, err = validator_fn(seq)
            if err and not err.startswith("Warning:"):
                n_errors += 1
                if n_errors <= 3:
                    print_error(f"  {name}: {err}")
                continue
            if err:
                print_warning(f"  {name}: {err}")
            values.append(cleaned)
            names.append(name)

        if n_errors > 3:
            print(dim(f"  ... and {n_errors - 3} more errors"))
        if values:
            print_success(f"  {len(values)} sequences loaded from FASTA" +
                           (f" ({n_errors} skipped)" if n_errors else ""))
        else:
            print_warning("No valid sequences in FASTA file.")
        return values, names

    # Plain text: one value per line
    values = []
    n_errors = 0
    for lineno, raw in enumerate(text.splitlines(), 1):
        raw = raw.strip()
        if not raw:
            continue
        cleaned, err = validator_fn(raw)
        if err and not err.startswith("Warning:"):
            n_errors += 1
            if n_errors <= 3:
                print_error(f"  line {lineno}: {err}")
            continue
        if err:
            print_warning(err)
        values.append(cleaned)

    if n_errors > 3:
        print(dim(f"  ... and {n_errors - 3} more errors"))
    if values:
        print_success(f"  {len(values)} values loaded from file" +
                       (f" ({n_errors} skipped)" if n_errors else ""))
    else:
        print_warning("No valid values in file.")
    return values, None


# ---------------------------------------------------------------------------
# Entity collection helpers
# ---------------------------------------------------------------------------

def _collect_protein(used_ids: set):
    """Collect a protein entity with one or more sequences.

    Returns (entity, values, names) or (None, [], None).
    Raises GoBack from first prompt.
    *values* is non-empty only when multiple sequences are provided.
    *names* is a list of FASTA header names or None.
    """
    print(dim("  Sequence(s) — one per line, blank line when done. /path to load file."))
    sequences, names = _read_values("Sequence", validate_protein_sequence)
    if not sequences:
        print_warning("Empty sequence, skipping.")
        return None, [], None

    raw = styled_prompt("Chains (count or IDs, e.g. 2 or A,B)", "1")
    ids, err = _parse_copies_or_ids(raw, used_ids)
    if err:
        print_error(f"{err} Try again.")
        return None, [], None

    used_ids.update(ids)
    entity = ProteinEntity(ids=ids, sequence=sequences[0])
    if len(sequences) > 1:
        print_success(
            f"  + protein {','.join(ids)} "
            f"({len(sequences)} sequences for screening)"
        )
    else:
        print_success(f"  + protein {','.join(ids)} ({len(sequences[0])} residues)")
    return entity, sequences if len(sequences) > 1 else [], names


def _collect_ligand(used_ids: set):
    """Collect a ligand entity with one or more values.

    Returns (entity, values, names) or (None, [], None).
    Raises GoBack from first prompt.
    *values* is non-empty only when multiple values are provided.
    *names* is a list of FASTA header names or None.
    """
    stype = styled_select(
        "Format",
        [("s", "miles"), ("c", "cd")],
        default=0,
    )

    if stype in ("s", "smiles"):
        print(dim("  SMILES — one per line, blank line when done. /path to load file."))
        smiles_list, names = _read_values("SMILES", validate_smiles)
        if not smiles_list:
            print_warning("No SMILES provided, skipping.")
            return None, [], None
        raw = styled_prompt("Chains (count or IDs, e.g. 2 or A,B)", "1")
        ids, err = _parse_copies_or_ids(raw, used_ids)
        if err:
            print_error(f"{err} Try again.")
            return None, [], None
        used_ids.update(ids)
        entity = LigandEntity(ids=ids, smiles=smiles_list[0])
        if len(smiles_list) > 1:
            print_success(
                f"  + ligand {','.join(ids)} "
                f"({len(smiles_list)} SMILES for screening)"
            )
        else:
            print_success(f"  + ligand {','.join(ids)} (SMILES)")
        return entity, smiles_list if len(smiles_list) > 1 else [], names

    elif stype in ("c", "ccd"):
        print(dim("  CCD code(s) — one per line, blank line when done. /path to load file."))
        ccd_list, names = _read_values("CCD code", validate_ccd)
        if not ccd_list:
            print_warning("No CCD code provided, skipping.")
            return None, [], None
        raw = styled_prompt("Chains (count or IDs, e.g. 2 or A,B)", "1")
        ids, err = _parse_copies_or_ids(raw, used_ids)
        if err:
            print_error(f"{err} Try again.")
            return None, [], None
        used_ids.update(ids)
        entity = LigandEntity(ids=ids, ccd=ccd_list[0])
        if len(ccd_list) > 1:
            print_success(
                f"  + ligand {','.join(ids)} "
                f"({len(ccd_list)} CCD codes for screening)"
            )
        else:
            print_success(f"  + ligand {','.join(ids)} (CCD: {ccd_list[0]})")
        return entity, ccd_list if len(ccd_list) > 1 else [], names

    else:
        print_error("Unknown format, use 's' or 'c'.")
        return None, [], None


def _collect_dna(used_ids: set):
    """Collect a DNA entity with one or more sequences.

    Returns (entity, values, names) or (None, [], None).
    Raises GoBack from first prompt.
    *values* is non-empty only when multiple sequences are provided.
    *names* is a list of FASTA header names or None.
    """
    print(dim("  DNA sequence(s) (ACGT) — one per line, blank line when done. /path to load file."))
    sequences, names = _read_values("Sequence", validate_dna_sequence)
    if not sequences:
        print_warning("Empty sequence, skipping.")
        return None, [], None

    raw = styled_prompt("Chains (count or IDs, e.g. 2 or A,B)", "1")
    ids, err = _parse_copies_or_ids(raw, used_ids)
    if err:
        print_error(f"{err} Try again.")
        return None, [], None

    used_ids.update(ids)
    entity = DnaEntity(ids=ids, sequence=sequences[0])
    if len(sequences) > 1:
        print_success(
            f"  + DNA {','.join(ids)} "
            f"({len(sequences)} sequences for screening)"
        )
    else:
        print_success(f"  + DNA {','.join(ids)} ({len(sequences[0])} nt)")
    return entity, sequences if len(sequences) > 1 else [], names


def _collect_rna(used_ids: set):
    """Collect an RNA entity with one or more sequences.

    Returns (entity, values, names) or (None, [], None).
    Raises GoBack from first prompt.
    *values* is non-empty only when multiple sequences are provided.
    *names* is a list of FASTA header names or None.
    """
    print(dim("  RNA sequence(s) (ACGU) — one per line, blank line when done. /path to load file."))
    sequences, names = _read_values("Sequence", validate_rna_sequence)
    if not sequences:
        print_warning("Empty sequence, skipping.")
        return None, [], None

    raw = styled_prompt("Chains (count or IDs, e.g. 2 or A,B)", "1")
    ids, err = _parse_copies_or_ids(raw, used_ids)
    if err:
        print_error(f"{err} Try again.")
        return None, [], None

    used_ids.update(ids)
    entity = RnaEntity(ids=ids, sequence=sequences[0])
    if len(sequences) > 1:
        print_success(
            f"  + RNA {','.join(ids)} "
            f"({len(sequences)} sequences for screening)"
        )
    else:
        print_success(f"  + RNA {','.join(ids)} ({len(sequences[0])} nt)")
    return entity, sequences if len(sequences) > 1 else [], names


# ---------------------------------------------------------------------------
# Entity registration and variant tracking
# ---------------------------------------------------------------------------

def _register_entity(state: WizardState, entity, values, names=None):
    """Add entity to state and record multi-values in variant_set if needed.

    *names* is an optional list of human-readable names (e.g. from FASTA
    headers) parallel to *values*.
    """
    if entity is None:
        return
    state.entities.append(entity)
    idx = len(state.entities) - 1
    if values:
        if state.variant_set is None:
            state.variant_set = VariantSet(variants={})
        state.variant_set.variants[idx] = values
        if names is not None and len(names) == len(values):
            state.variant_set.variant_names[idx] = names
    entity_table(state.entities, variant_counts=_get_variant_counts(state))


def _get_variant_counts(state: WizardState):
    """Return {entity_index: count} dict or None for entity_table display."""
    if state.variant_set is None:
        return None
    return {idx: len(vals) for idx, vals in state.variant_set.variants.items()}


# ---------------------------------------------------------------------------
# Step 1: Entities
# ---------------------------------------------------------------------------

def step_entities(state: WizardState):
    """Collect proteins, ligands, etc. Multi-value input is inline."""
    section_header("Entities", 1, NUM_STEPS)
    step_progress_bar(0, NUM_STEPS, STEP_LABELS)
    _show_back_hint("entities")

    # Re-entry: show existing entities
    if state.entities:
        print(f"  Current entities:")
        entity_table(state.entities, variant_counts=_get_variant_counts(state))
        if state.variant_set:
            for idx, vals in state.variant_set.variants.items():
                print_info(
                    f"  Entity {idx + 1}: {len(vals)} values for screening"
                )
        keep = styled_confirm("Keep existing entities?", default=True)
        if keep:
            # Allow adding more
            print(dim("  Add more entities, or type 'd' when done."))
        else:
            state.entities = []
            state.used_ids = set()
            state.variant_set = None
            print_info("Entities cleared, starting fresh.")
    else:
        print(dim("  Paste multiple values for screening, or enter a /path to load from file"))

    while True:
        try:
            default_idx = 4 if state.entities else 0
            etype = styled_select(
                "Entity type",
                [("p", "rotein"), ("l", "igand"), ("D", "NA"), ("R", "NA"), ("d", "one")],
                default=default_idx,
            )
        except GoBack:
            if not state.entities:
                raise
            else:
                break

        if etype in ("d", "done", ""):
            if not state.entities:
                print_error("Need at least one entity.")
                continue
            break

        elif etype in ("p", "protein"):
            try:
                entity, values, names = _collect_protein(state.used_ids)
            except GoBack:
                print(dim("  (cancelled, staying in entity step)"))
                continue
            _register_entity(state, entity, values, names)

        elif etype in ("l", "ligand"):
            try:
                entity, values, names = _collect_ligand(state.used_ids)
            except GoBack:
                print(dim("  (cancelled, staying in entity step)"))
                continue
            _register_entity(state, entity, values, names)

        elif etype in ("D", "DNA"):
            try:
                entity, values, names = _collect_dna(state.used_ids)
            except GoBack:
                print(dim("  (cancelled, staying in entity step)"))
                continue
            _register_entity(state, entity, values, names)

        elif etype in ("R", "RNA"):
            try:
                entity, values, names = _collect_rna(state.used_ids)
            except GoBack:
                print(dim("  (cancelled, staying in entity step)"))
                continue
            _register_entity(state, entity, values, names)

        else:
            print_error("Unknown type. Use p / l / D / R / d.")


# ---------------------------------------------------------------------------
# Step 2: Constraints
# ---------------------------------------------------------------------------

def step_constraints(state: WizardState):
    """Optionally add pocket constraints."""
    section_header("Constraints", 2, NUM_STEPS)
    step_progress_bar(1, NUM_STEPS, STEP_LABELS)

    ligand_ids = []
    for e in state.entities:
        if isinstance(e, LigandEntity):
            ligand_ids.extend(e.ids)

    if not ligand_ids:
        print(dim("  No ligands — skipping constraints."))
        state.constraints = []
        return

    _show_back_hint("constraints")

    # Show entity table for reference
    entity_table(state.entities)

    # Re-entry: show existing constraints
    if state.constraints:
        print(f"  Current constraints: {len(state.constraints)}")
        for i, c in enumerate(state.constraints, 1):
            print(f"    {i}. pocket on {value(c.binder)} "
                  f"({len(c.contacts)} contacts, "
                  f"max_dist={c.max_distance}, force={c.force})")
        keep = styled_confirm("Keep existing constraints?", default=True)
        if not keep:
            state.constraints = []

    while True:
        try:
            add = styled_confirm("Add pocket constraint?", default=False)
        except GoBack:
            raise
        if not add:
            break

        try:
            binder = styled_prompt(
                f"Binder chain ({', '.join(ligand_ids)})"
            ).strip().upper()
        except GoBack:
            continue

        if binder not in ligand_ids:
            print_error(f"'{binder}' is not a ligand chain. Skipping.")
            continue

        contacts = []
        print(dim("  Add contacts (chain,residue). Enter 'done' to finish."))
        while True:
            try:
                c_str = styled_prompt("Contact (e.g. A,96)", "done")
            except GoBack:
                break
            if c_str.lower() == "done":
                break
            parts = [x.strip() for x in c_str.split(",")]
            if len(parts) == 2:
                try:
                    contacts.append((parts[0].upper(), int(parts[1])))
                except ValueError:
                    print_error("Residue must be an integer.")
            else:
                print_error("Format: chain,residue  (e.g. A,96)")

        if not contacts:
            print_warning("No contacts added, skipping constraint.")
            continue

        try:
            max_dist = float(styled_prompt("Max distance", "6.0"))
            force = styled_confirm("Force?", default=True)
        except GoBack:
            continue

        state.constraints.append(PocketConstraint(
            binder=binder, contacts=contacts,
            force=force, max_distance=max_dist,
        ))
        print_success(
            f"  + pocket constraint on {binder} ({len(contacts)} contacts)"
        )


# ---------------------------------------------------------------------------
# Step 3: Properties
# ---------------------------------------------------------------------------

def step_properties(state: WizardState):
    """Optionally request affinity prediction."""
    section_header("Properties", 3, NUM_STEPS)
    step_progress_bar(2, NUM_STEPS, STEP_LABELS)

    ligand_ids = []
    for e in state.entities:
        if isinstance(e, LigandEntity):
            ligand_ids.extend(e.ids)

    if not ligand_ids:
        print(dim("  No ligands — skipping affinity."))
        state.properties = []
        return

    _show_back_hint("properties")

    # Smart default: if only 1 ligand, auto-select it
    default_binder = ligand_ids[0] if len(ligand_ids) == 1 else None

    # Re-entry: show existing
    if state.properties:
        for p in state.properties:
            print(f"  Current: affinity for binder {value(p.binder)}")
        keep = styled_confirm("Keep existing affinity setting?", default=True)
        if keep:
            return
        state.properties = []

    if default_binder:
        print_info(f"Only one ligand chain: {value(default_binder)}")

    want = styled_confirm("Predict affinity?", default=False)
    if not want:
        return

    if default_binder:
        use_default = styled_confirm(
            f"Use {value(default_binder)} as binder?", default=True,
        )
        if use_default:
            state.properties.append(AffinityProperty(binder=default_binder))
            print_success(f"  + affinity for {default_binder}")
            return

    binder = styled_prompt(
        f"Binder chain ({', '.join(ligand_ids)})"
    ).strip().upper()
    if binder in ligand_ids:
        state.properties.append(AffinityProperty(binder=binder))
        print_success(f"  + affinity for {binder}")
    else:
        print_error(f"'{binder}' is not a ligand chain, skipping.")


# ---------------------------------------------------------------------------
# Step 4: Templates
# ---------------------------------------------------------------------------

def step_templates(state: WizardState):
    """Optionally add structural templates for guided prediction."""
    section_header("Templates", 4, NUM_STEPS)
    step_progress_bar(3, NUM_STEPS, STEP_LABELS)

    # Prerequisite: need at least one protein entity
    has_protein = any(isinstance(e, ProteinEntity) for e in state.entities)
    if not has_protein:
        print(dim("  No protein entities — skipping templates."))
        state.templates = []
        return

    _show_back_hint("templates")

    # Re-entry: show existing templates
    if state.templates:
        print(bold("  Current templates:"))
        for i, t in enumerate(state.templates, 1):
            fname = os.path.basename(t.file_path)
            chains_str = ",".join(t.chain_ids) if t.chain_ids else "auto"
            tpl_str = ",".join(t.template_ids) if t.template_ids else "auto"
            force_str = f", force={t.threshold}Å" if t.force else ""
            print(f"    {i}. {value(fname)} ({t.file_format}) → chains {chains_str} ← {tpl_str}{force_str}")
        keep = styled_confirm("Keep existing templates?", default=True)
        if keep:
            return
        state.templates = []

    # Collect protein chain IDs for mapping
    protein_chain_ids = []
    for e in state.entities:
        if isinstance(e, ProteinEntity):
            protein_chain_ids.extend(e.ids)

    while True:
        try:
            add = styled_confirm("Add structural template?", default=False)
        except GoBack:
            raise
        if not add:
            break

        # File path
        file_path = None
        file_format = None
        template_chains = []
        while True:
            try:
                raw_path = styled_prompt("Template file path")
            except GoBack:
                break
            resolved = os.path.abspath(raw_path)
            if not os.path.isfile(resolved):
                print_error(f"File not found: {resolved}")
                continue
            ext = os.path.splitext(resolved)[1].lower()
            if ext == ".cif":
                file_format = "cif"
            elif ext == ".pdb":
                file_format = "pdb"
            else:
                print_error(f"Unsupported extension '{ext}' — must be .cif or .pdb")
                continue
            template_chains = extract_template_chains(resolved)
            if not template_chains:
                print_error("Could not extract chains from file (is it a valid structure?)")
                continue
            file_path = resolved
            print_info(f"  Found chains: {', '.join(template_chains)}")
            break

        if file_path is None:
            continue

        # Chain ID mapping — input chains
        chain_ids = None
        if len(protein_chain_ids) == 1:
            # Only one protein chain — auto-select
            use_single = styled_confirm(
                f"Map to protein chain {value(protein_chain_ids[0])}?", default=True,
            )
            if use_single:
                chain_ids = [protein_chain_ids[0]]
        else:
            skip_opt = ("skip", " (auto-detect)")
            chain_options = [(cid, "") for cid in protein_chain_ids] + [skip_opt]
            selected_chains = []
            while True:
                try:
                    choice = styled_select(
                        "Input chain" if not selected_chains else "Add chain",
                        chain_options,
                        default=len(chain_options) - 1,
                    )
                except GoBack:
                    break
                if choice == "skip":
                    break
                if choice in protein_chain_ids and choice not in selected_chains:
                    selected_chains.append(choice)
                    print_info(f"  Selected: {', '.join(selected_chains)}")
                    if not styled_confirm("Add another chain?", default=False):
                        break
                elif choice in selected_chains:
                    print_warning(f"Chain {choice} already selected.")
                else:
                    break
            if selected_chains:
                chain_ids = selected_chains

        # Template ID mapping — chains in template file
        template_ids = None
        if chain_ids and len(template_chains) > 0:
            print(dim(f"  Map each input chain to a template chain from: {', '.join(template_chains)}"))
            skip_opt = ("skip", " (auto-match)")
            mapped = []
            for cid in chain_ids:
                tpl_options = [(tc, "") for tc in template_chains] + [skip_opt]
                # Default to same letter if available, else skip
                default_idx = len(tpl_options) - 1
                for ti, tc in enumerate(template_chains):
                    if tc == cid:
                        default_idx = ti
                        break
                try:
                    choice = styled_select(
                        f"Template chain for {cid}",
                        tpl_options,
                        default=default_idx,
                    )
                except GoBack:
                    mapped = []
                    break
                if choice == "skip":
                    mapped = []
                    break
                mapped.append(choice)
            if mapped:
                template_ids = mapped

        # Force
        force = False
        threshold = None
        force_choice = styled_select(
            "Force backbone constraint?",
            [("n", "o"), ("y", "es")],
            default=0,
        )
        if force_choice in ("y", "yes"):
            force = True
            while True:
                try:
                    raw_thresh = styled_prompt("Threshold (Å)", "2.5")
                    threshold = float(raw_thresh)
                    break
                except ValueError:
                    print_error("Must be a number.")
                except GoBack:
                    break
            if threshold is None:
                force = False

        entry = TemplateEntry(
            file_path=file_path,
            file_format=file_format,
            chain_ids=chain_ids,
            template_ids=template_ids,
            force=force,
            threshold=threshold,
        )
        state.templates.append(entry)
        fname = os.path.basename(file_path)
        print_success(f"  + template {fname}")

        if not styled_confirm("Add another template?", default=False):
            break


# ---------------------------------------------------------------------------
# Step 5: Boltz parameters
# ---------------------------------------------------------------------------

def step_boltz(state: WizardState):
    """Prompt for Boltz prediction parameters."""
    section_header("Boltz Parameters", 5, NUM_STEPS)
    step_progress_bar(4, NUM_STEPS, STEP_LABELS)
    _show_back_hint("boltz")

    bp = state.boltz

    # Smart default: suggest potentials if force constraints or templates exist
    has_constraint_force = any(
        getattr(c, 'force', False) for c in state.constraints
    )
    has_template_force = any(
        getattr(t, 'force', False) for t in state.templates
    )
    has_force = has_constraint_force or has_template_force
    if has_constraint_force and not bp.use_potentials:
        print_info("Force constraints detected — recommending use_potentials.")
    if has_template_force and not bp.use_potentials:
        print_warning(
            "Template force requires use_potentials — without it, "
            "force/threshold will be silently ignored."
        )

    bp.recycling_steps = int(
        styled_prompt("Recycling steps", bp.recycling_steps)
    )
    bp.diffusion_samples = int(
        styled_prompt("Diffusion samples", bp.diffusion_samples)
    )

    potentials_default = True if has_force else bp.use_potentials
    print(dim("  Enables physics-based steering (VDW, chirality, bond geometry, template force)."))
    bp.use_potentials = styled_confirm("Use potentials?", default=potentials_default)

    if not bp.use_potentials and has_template_force:
        print_warning(
            "Template force=true will have no effect without use_potentials!"
        )


# ---------------------------------------------------------------------------
# Step 6: Slurm settings
# ---------------------------------------------------------------------------

def step_slurm(state: WizardState):
    """Compute resource recommendation and prompt for Slurm parameters."""
    section_header("Slurm Settings", 6, NUM_STEPS)
    step_progress_bar(5, NUM_STEPS, STEP_LABELS)
    _show_back_hint("slurm")

    # Always recompute recommendations from current state
    total_tokens = estimate_tokens(state.entities)
    n_variants = 1
    if state.variant_set:
        for vals in state.variant_set.variants.values():
            n_variants *= len(vals)
    state.gpu_rec = recommend_gpu(total_tokens)
    state.time_rec = recommend_time(total_tokens, state.boltz, n_variants=n_variants,
                                    gpu_rec=state.gpu_rec)

    gpu_rec = state.gpu_rec
    time_rec = state.time_rec

    # Build defaults from recommendation (or previous values on re-entry)
    if state.slurm is None:
        sp = SlurmParams(job_name=state.job_name)
    else:
        sp = state.slurm
    sp.gpus_per_node = gpu_rec["gpu_sbatch"]
    sp.mem = gpu_rec["mem"]
    sp.time = time_rec["time"]
    sp.partition = time_rec["partition"]

    # Display recommended settings as a single overview
    print(f"  {bold('Recommended settings')}")
    print_value("  Tokens", total_tokens)
    print_value("  GPU", sp.gpus_per_node)
    print_value("  Memory", sp.mem)
    print_value("  Time", f"{sp.time}  (~{time_rec['estimated_minutes']} min estimated)")
    print_value("  Partition", sp.partition)
    print_value("  CPUs", sp.cpus_per_task)
    if gpu_rec["boltz_extra_flags"]:
        print_value("  Extra flags", " ".join(gpu_rec["boltz_extra_flags"]))
    for w in gpu_rec["warnings"] + time_rec["warnings"]:
        print_warning(w)
    print()

    # Accept all or customize
    accept = styled_confirm("Accept these settings?", default=True)
    if not accept:
        sp.time = styled_prompt("Time limit", sp.time)
        sp.partition = styled_prompt("Partition", sp.partition)
        sp.gpus_per_node = styled_prompt(
            "GPUs (e.g. 1 or gpu:a100:1)", sp.gpus_per_node,
        )
        sp.mem = styled_prompt("Memory", sp.mem)
        sp.cpus_per_task = int(
            styled_prompt("CPUs per task", sp.cpus_per_task)
        )

    state.slurm = sp

    # Auto-set no_kernels if V100 selected (no Tensor Core support for trifast)
    if "v100" in sp.gpus_per_node.lower():
        state.boltz.no_kernels = True


# ---------------------------------------------------------------------------
# Step 7: Review
# ---------------------------------------------------------------------------

def step_review(state: WizardState):
    """Show full summary and generated files. Allow jumping back."""
    section_header("Review", 7, NUM_STEPS)
    step_progress_bar(6, NUM_STEPS, STEP_LABELS)

    # Summary
    print(bold("  Entities:"))
    entity_table(state.entities, variant_counts=_get_variant_counts(state))

    if state.variant_set:
        vs = state.variant_set
        print(bold("  Variant Screening:"))
        n_combos = 1
        for idx in sorted(vs.variants.keys()):
            vals = vs.variants[idx]
            n_combos *= len(vals)
            entity = state.entities[idx]
            etype = type(entity).__name__.replace("Entity", "")
            chains = ",".join(entity.ids)
            # Show first 3 + last as sample
            sample = vals[:3]
            sample_str = ", ".join(
                (v[:30] + "..." if len(v) > 30 else v) for v in sample
            )
            if len(vals) > 4:
                last = vals[-1]
                sample_str += f", ... , {last[:30] + '...' if len(last) > 30 else last}"
            elif len(vals) == 4:
                last = vals[-1]
                sample_str += f", {last[:30] + '...' if len(last) > 30 else last}"
            print_value(
                f"    Entity {idx + 1} ({etype} {chains})",
                f"{len(vals)} values"
            )
            print(dim(f"      {sample_str}"))
        if len(vs.variants) > 1:
            print_value("    Combinations", n_combos)
        print_value("    YAML files", n_combos)
        print()

    if state.constraints:
        print(bold("  Constraints:"))
        for c in state.constraints:
            print(f"    pocket on {value(c.binder)} — "
                  f"{len(c.contacts)} contacts, "
                  f"max_dist={c.max_distance}, force={c.force}")
        print()

    if state.properties:
        print(bold("  Properties:"))
        for p in state.properties:
            print(f"    affinity binder: {value(p.binder)}")
        print()

    if state.templates:
        print(bold("  Templates:"))
        for t in state.templates:
            fname = os.path.basename(t.file_path)
            chains_str = ",".join(t.chain_ids) if t.chain_ids else "auto"
            tpl_str = ",".join(t.template_ids) if t.template_ids else "auto"
            force_str = f", force={t.threshold}Å" if t.force else ""
            print(f"    {value(fname)} ({t.file_format}) → chains {chains_str} ← {tpl_str}{force_str}")
        print()

    print(bold("  Boltz:"))
    bp = state.boltz
    print_value("    Recycling steps", bp.recycling_steps)
    print_value("    Diffusion samples", bp.diffusion_samples)
    print_value("    Potentials", bp.use_potentials)
    print()

    print(bold("  Slurm:"))
    sp = state.slurm
    print_value("    Partition", sp.partition)
    print_value("    GPUs", sp.gpus_per_node)
    print_value("    Time", sp.time)
    print_value("    Memory", sp.mem)
    print()

    # Generated files
    job_dir = f"{state.base_dir}/{state.job_name}"
    script_content = build_job_script(
        job_dir=job_dir,
        cache_dir=BOLTZ_CACHE_DIR,
        boltz_params=state.boltz,
        slurm_params=state.slurm,
        python_module=PYTHON_MODULE,
    )

    hrule()
    if state.variant_set:
        yaml_files = build_yaml_variants(
            state.entities,
            state.constraints or None,
            state.properties or None,
            state.variant_set,
            state.job_name,
            templates=state.templates or None,
        )
        n_combos = len(yaml_files)
        yaml_content = yaml_files[0][1]  # first combo for preview
        print(bold(f"  Generated YAML (1 of {n_combos}):"))
    else:
        yaml_content = build_yaml(
            state.entities,
            state.constraints or None,
            state.properties or None,
            templates=state.templates or None,
        )
        print(bold("  Generated YAML:"))
    print()
    print_yaml(yaml_content)

    hrule()

    # Navigation — primary action selector
    print()
    action = styled_select(
        "Action",
        [("S", "ubmit"), ("E", "dit step"), ("V", "iew script")],
        default=0, allow_back=True,
    )

    if action == "S":
        return  # proceed to write & submit

    if action == "V":
        hrule()
        print(bold("  Generated job script:"))
        print()
        print_bash(script_content)
        hrule()
        step_review(state)
        return

    # action == "E" — pick which step to edit
    try:
        step_choice = styled_select(
            "Edit step",
            [
                ("1", " Entities"),
                ("2", " Constraints"),
                ("3", " Properties"),
                ("4", " Templates"),
                ("5", " Boltz"),
                ("6", " Slurm"),
            ],
            default=0, allow_back=True,
        )
        raise JumpToStep(int(step_choice) - 1)
    except GoBack:
        # Back from step picker → re-show review (not jump to Slurm)
        step_review(state)


# ---------------------------------------------------------------------------
# File writing and submission
# ---------------------------------------------------------------------------

def step_write_and_submit(state: WizardState):
    """Write files and optionally submit the job."""
    print()
    job_dir = f"{state.base_dir}/{state.job_name}"
    script_content = build_job_script(
        job_dir=job_dir,
        cache_dir=BOLTZ_CACHE_DIR,
        boltz_params=state.boltz,
        slurm_params=state.slurm,
        python_module=PYTHON_MODULE,
    )

    if state.variant_set:
        yaml_files = build_yaml_variants(
            state.entities,
            state.constraints or None,
            state.properties or None,
            state.variant_set,
            state.job_name,
            templates=state.templates or None,
        )
        n = len(yaml_files)
        if state.resume_mode:
            confirm_msg = f"Add {n} YAML files to existing job?"
        else:
            confirm_msg = f"Write {n} YAML files + job script?"
    else:
        yaml_content = build_yaml(
            state.entities,
            state.constraints or None,
            state.properties or None,
            templates=state.templates or None,
        )
        yaml_files = [(f"{state.job_name}.yaml", yaml_content)]
        if state.resume_mode:
            confirm_msg = "Add YAML file to existing job?"
        else:
            confirm_msg = "Write files?"

    if styled_confirm(confirm_msg, default=True, allow_back=False):
        if state.resume_mode:
            result_dir, script_name = setup_job_resume(
                job_dir, yaml_files, script_content,
            )
            print_success(
                f"Added {len(yaml_files)} YAML file(s) to {result_dir}/input/"
            )
            print_success(f"Job script written as {script_name}")

            if styled_confirm("Submit job now?", default=False, allow_back=False):
                try:
                    job_id = submit_job(str(result_dir), script_name=script_name)
                    print_success(f"Submitted batch job {job_id}")
                    print_info("A clean log will be generated automatically when the job finishes.")
                except RuntimeError as exc:
                    print_error(f"Submission failed: {exc}")
            else:
                print_info(f"To submit: sbatch {result_dir}/{script_name}")
        else:
            result_dir = setup_job_variants(
                state.base_dir, state.job_name, yaml_files, script_content,
            )
            print_success(f"Files written to {result_dir}/ ({len(yaml_files)} YAML file(s))")

            if styled_confirm("Submit job now?", default=False, allow_back=False):
                try:
                    job_id = submit_job(str(result_dir))
                    print_success(f"Submitted batch job {job_id}")
                    print_info("A clean log will be generated automatically when the job finishes.")
                except RuntimeError as exc:
                    print_error(f"Submission failed: {exc}")
            else:
                print_info(f"To submit: sbatch {result_dir}/job.sh")
    else:
        print(dim("  Aborted."))


# ---------------------------------------------------------------------------
# Main wizard loop
# ---------------------------------------------------------------------------

STEPS = [
    step_entities,
    step_constraints,
    step_properties,
    step_templates,
    step_boltz,
    step_slurm,
    step_review,
]


def main():
    # --help / -h
    if len(sys.argv) >= 2 and sys.argv[1] in ('-h', '--help'):
        print(f"Usage: boltz-setup [JOB_NAME]")
        print(f"       boltz-setup --resume [JOB_NAME]")
        print(f"       boltz-setup log <job_dir> [--file <slurm.out>]")
        print()
        print("Interactive wizard for setting up Boltz structure prediction jobs.")
        print(f"Jobs are created under {BOLTZ_JOBS_DIR}/")
        print()
        print("Options:")
        print("  --resume [JOB_NAME]  Add new inputs to an existing job directory.")
        print("                       Old YAML files are kept; boltz skips already-processed inputs.")
        print("                       Offers to pre-populate wizard state from an existing YAML,")
        print("                       so you can tweak entities/constraints rather than re-entering.")
        print("                       If JOB_NAME is omitted, shows a picker of existing jobs.")
        print()
        print("  log <job_dir>        Parse a finished job's slurm-*.out into a clean summary.")
        print("                       Runs automatically at job end; can also be run manually.")
        print()
        print("Steps:")
        print("  1. Entities    — proteins, ligands, DNA/RNA (multi-value for screening)")
        print("  2. Constraints — pocket constraints for ligand binding")
        print("  3. Properties  — affinity prediction")
        print("  4. Templates   — structural templates for guided prediction")
        print("  5. Boltz       — recycling steps, diffusion samples, potentials")
        print("  6. Slurm       — auto-recommended GPU/time/partition; accept all or customize")
        print("  7. Review      — preview YAML + job script, then write and submit")
        print()
        print("Job scripts:")
        print("  Written as job.sh (new jobs) or job_1.sh, job_2.sh, ... (resumed jobs).")
        print("  On submission, the script renames itself to job_<slurm_id>.sh so")
        print("  multiple runs in the same directory are easy to tell apart.")
        print()
        print("Examples:")
        print("  boltz-setup myjob                # set up a new job")
        print("  boltz-setup --resume myjob       # add inputs to existing job")
        print("  boltz-setup --resume             # pick from existing jobs")
        print("  boltz-setup log myjob            # parse results after completion")
        sys.exit(0)

    # Detect --resume flag
    resume_mode = False
    resume_name = None
    argv_rest = list(sys.argv[1:])
    if "--resume" in argv_rest:
        resume_mode = True
        idx = argv_rest.index("--resume")
        argv_rest.pop(idx)
        # Next arg (if any and not a flag) is the job name
        if idx < len(argv_rest) and not argv_rest[idx].startswith("-"):
            resume_name = argv_rest.pop(idx)

    # Intro
    print()
    if resume_mode:
        print(f"  {bold('Boltz Setup')} — resume mode (add inputs to existing job)")
    else:
        print(f"  {bold('Boltz Setup')} — structure prediction job wizard")
    print(dim("  Configure entities, constraints, and compute settings, then submit."))
    print(dim("  Run with --help for detailed usage info."))

    base_dir = BOLTZ_JOBS_DIR
    resume_parsed = None

    if resume_mode:
        # Pick or validate job directory
        if resume_name:
            job_name = resume_name
            job_dir = os.path.join(base_dir, job_name)
            if not os.path.isdir(job_dir):
                print_error(f"Job directory not found: {job_dir}")
                sys.exit(1)
        else:
            # List existing job dirs, sorted by mtime (newest first)
            try:
                dirs = sorted(
                    [d for d in Path(base_dir).iterdir() if d.is_dir()],
                    key=lambda p: p.stat().st_mtime, reverse=True,
                )
            except FileNotFoundError:
                dirs = []
            if not dirs:
                print_error(f"No job directories found in {base_dir}")
                sys.exit(1)

            # Show as a numbered select
            options = [(d.name, "") for d in dirs[:20]]
            print()
            hrule()
            job_name = styled_select(
                "Resume job", options, default=0, allow_back=False,
            )
            job_dir = os.path.join(base_dir, job_name)
            if not os.path.isdir(job_dir):
                print_error(f"Job directory not found: {job_dir}")
                sys.exit(1)

        # Count existing YAMLs and offer to pre-populate from one
        input_dir = Path(job_dir) / "input"
        existing_yamls = sorted(input_dir.glob("*.yaml")) if input_dir.is_dir() else []
        print_info(f"Resuming job: {job_name} ({len(existing_yamls)} existing YAML files)")

        # YAML picker: offer to base new inputs on an existing YAML
        resume_parsed = None
        if existing_yamls:
            yaml_options = [(y.name, "") for y in existing_yamls]
            yaml_options.append(("Start fresh", ""))
            print()
            picked = styled_select(
                "Base on existing YAML?",
                yaml_options,
                default=0,
                allow_back=False,
            )
            if picked != "Start fresh":
                # Find the matching file
                picked_path = input_dir / picked
                if picked_path.is_file():
                    try:
                        content = picked_path.read_text()
                        resume_parsed = parse_boltz_yaml(content)
                    except Exception as exc:
                        print_warning(f"Could not parse {picked}: {exc}")
    else:
        if argv_rest:
            job_name = argv_rest[0]
        else:
            print()
            hrule()
            try:
                job_name = styled_prompt("Job name", allow_back=False)
            except SystemExit:
                raise
            if not job_name:
                print_error("Job name is required.")
                sys.exit(1)

        # Check for existing job directory
        job_dir = os.path.join(base_dir, job_name)
        if os.path.isdir(job_dir):
            print_warning(f"Directory already exists: {job_dir}")
            if not styled_confirm("Overwrite?", default=False, allow_back=False):
                print(dim("  Choose a different job name."))
                sys.exit(0)

    state = WizardState(job_name=job_name, base_dir=base_dir, resume_mode=resume_mode)

    # Pre-populate state from parsed YAML (resume mode)
    if resume_mode and resume_parsed is not None:
        ents, cons, props, tmpls, uids = resume_parsed
        state.entities = ents
        state.used_ids = uids
        state.constraints = cons
        state.properties = props
        state.templates = tmpls
        # Summary of what was loaded
        parts = [f"{len(ents)} entities"]
        if cons:
            parts.append(f"{len(cons)} constraints")
        if props:
            parts.append("affinity=yes")
        if tmpls:
            parts.append(f"{len(tmpls)} templates")
        print_success(f"  Loaded from YAML: {', '.join(parts)}")

    reset_back_hints()

    print_banner(job_name, base_dir)

    step = 0
    try:
        while step < NUM_STEPS:
            try:
                STEPS[step](state)
                step += 1
            except GoBack:
                if step == 0:
                    print_warning("Already at the first step.")
                else:
                    step -= 1
            except JumpToStep as j:
                step = j.step_index
    except KeyboardInterrupt:
        print()
        print(dim("  Interrupted. Goodbye."))
        sys.exit(1)
    except SystemExit:
        raise
    except EOFError:
        print()
        print(dim("  EOF. Goodbye."))
        sys.exit(0)

    # All steps complete — write and submit
    try:
        step_write_and_submit(state)
    except KeyboardInterrupt:
        print()
        print(dim("  Interrupted. Goodbye."))
        sys.exit(1)


if __name__ == "__main__":
    main()
