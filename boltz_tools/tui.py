"""Terminal UI primitives — ANSI formatting, styled prompts, layout components.

Zero dependencies beyond stdlib. Colors auto-disable when stdout is not a tty.
"""

import os
import re
import sys

# ---------------------------------------------------------------------------
# Color system
# ---------------------------------------------------------------------------

USE_COLOR = sys.stdout.isatty()


class _Style:
    """ANSI escape code constants."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"

    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    GRAY = "\033[90m"

    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_CYAN = "\033[96m"


S = _Style


def style(text, *codes):
    """Apply ANSI codes to text. Returns plain text if colors are disabled."""
    if not USE_COLOR or not codes:
        return str(text)
    prefix = "".join(codes)
    return f"{prefix}{text}{S.RESET}"


# Semantic helpers
def bold(text):
    return style(text, S.BOLD)


def dim(text):
    return style(text, S.DIM)


def success(text):
    return style(text, S.GREEN)


def warning(text):
    return style(text, S.YELLOW)


def error(text):
    return style(text, S.BRIGHT_RED)


def info(text):
    return style(text, S.CYAN)


def highlight(text):
    return style(text, S.BRIGHT_CYAN)


def header(text):
    return style(text, S.BOLD, S.BLUE)


def value(text):
    return style(text, S.BRIGHT_CYAN)


# ---------------------------------------------------------------------------
# Navigation exceptions
# ---------------------------------------------------------------------------

class GoBack(Exception):
    """User typed 'back' — navigate to the previous step."""


class JumpToStep(Exception):
    """User wants to jump to a specific step (from review screen)."""
    def __init__(self, step_index):
        self.step_index = step_index
        super().__init__(f"jump to step {step_index}")


# ---------------------------------------------------------------------------
# Layout components
# ---------------------------------------------------------------------------

def _term_width():
    """Get terminal width, capped at 72."""
    try:
        return min(72, os.get_terminal_size().columns)
    except (AttributeError, ValueError, OSError):
        return 72


def hrule(char="\u2500"):
    """Print a horizontal rule."""
    print(dim(char * _term_width()))


def section_header(title, step, total):
    """Display a step header between horizontal rules."""
    hrule()
    badge = f" Step {step}/{total}"
    line = f"{badge} \u2500\u2500 {title}"
    print(header(line))
    hrule()


def step_progress_bar(current, total, labels):
    """Compact one-liner: completed (green), current (bold cyan), future (dim).

    current is 0-indexed internally but displayed 1-indexed.
    """
    parts = []
    for i, label in enumerate(labels):
        num = f"{i + 1}"
        if i < current:
            parts.append(success(f"[{num}]"))
        elif i == current:
            parts.append(highlight(f"[{num} {label}]"))
        else:
            parts.append(dim(f"[{num}]"))
    print(" ".join(parts))
    print()


def print_banner(job_name, base_dir):
    """Startup banner with job name and directory."""
    print()
    hrule()
    print(header(f"  Boltz Job Setup: {job_name}"))
    print(dim(f"  Directory: {base_dir}/{job_name}/"))
    hrule()
    print()


# ---------------------------------------------------------------------------
# Entity table
# ---------------------------------------------------------------------------

def entity_table(entities, variant_counts=None):
    """Print a formatted entity table.

    Columns: #, Type, Chains, Details.
    If *variant_counts* is set (dict of {entity_index: count}), append
    `` x{N}`` to the type column for rows with multiple values.
    """
    from .generate import ProteinEntity, DnaEntity, RnaEntity, LigandEntity

    if not entities:
        print(dim("  (no entities yet)"))
        return

    # Column widths
    w_num = 3
    w_type = 14
    w_chain = 8
    w_detail = max(20, _term_width() - w_num - w_type - w_chain - 10)

    hdr = (
        f" {'#':<{w_num}} | {'Type':<{w_type}} | {'Chains':<{w_chain}} | Details"
    )
    rule = "\u2500"
    sep = (
        f" {rule * w_num}+{rule * (w_type + 1)}+"
        f"{rule * (w_chain + 1)}+{rule * w_detail}"
    )

    print(dim(hdr))
    print(dim(sep))

    for i, e in enumerate(entities, 1):
        chains = ", ".join(e.ids)
        if isinstance(e, ProteinEntity):
            etype = "Protein"
            detail = f"{len(e.sequence)} residues"
        elif isinstance(e, DnaEntity):
            etype = "DNA"
            detail = f"{len(e.sequence)} nt"
        elif isinstance(e, RnaEntity):
            etype = "RNA"
            detail = f"{len(e.sequence)} nt"
        elif isinstance(e, LigandEntity):
            etype = "Ligand"
            if e.smiles:
                s = e.smiles if len(e.smiles) <= 40 else e.smiles[:37] + "..."
                detail = f"SMILES: {s}"
            elif e.ccd:
                detail = f"CCD: {e.ccd}"
            else:
                detail = ""
        else:
            etype = "?"
            detail = ""

        # Mark entities with multiple values
        if variant_counts and (i - 1) in variant_counts:
            etype = etype + f" x{variant_counts[i - 1]}"

        num_s = f" {i:<{w_num}}"
        type_s = f" {etype:<{w_type}}"
        chain_s = f" {chains:<{w_chain}}"
        detail_s = f" {detail}"

        print(f"{num_s}|{type_s}|{chain_s}|{detail_s}")

    print()


# ---------------------------------------------------------------------------
# Syntax-highlighted output
# ---------------------------------------------------------------------------

def print_yaml(text):
    """Print YAML with keys in cyan and comments dimmed."""
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            print(dim(line))
        else:
            # Color the key portion (word followed by colon)
            m = re.match(r'^(\s*(?:- )?)(\w[\w\s]*?)(:)(.*)', line)
            if m:
                indent, key, colon, rest = m.groups()
                print(f"{indent}{info(key)}{colon}{rest}")
            else:
                print(line)


def print_bash(text):
    """Print bash script with comments dimmed and exports in green."""
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            print(dim(line))
        elif stripped.startswith("export "):
            print(success(line))
        else:
            print(line)


# ---------------------------------------------------------------------------
# Styled prompts
# ---------------------------------------------------------------------------

_back_hint_shown = set()


def _show_back_hint(step_key=None):
    """Print the back hint once per step."""
    if step_key is not None and step_key in _back_hint_shown:
        return
    if step_key is not None:
        _back_hint_shown.add(step_key)
    print(dim("  (type 'back' to return to previous step)"))


def reset_back_hints():
    """Clear the set of shown back hints (for fresh wizard runs)."""
    _back_hint_shown.clear()


def _check_back(val):
    """Raise GoBack if val is a back command."""
    if val.lower().strip() in ("back", "b", "prev"):
        raise GoBack()


def styled_prompt(msg, default=None, allow_back=True):
    """Prompt with colored default value.

    If input is 'back'/'b'/'prev', raises GoBack.
    Catches EOFError/KeyboardInterrupt for clean exit.
    """
    if default is not None:
        suffix = f" [{value(str(default))}]"
    else:
        suffix = ""
    try:
        val = input(f"  {msg}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        raise SystemExit(0)
    if allow_back:
        _check_back(val)
    return val if val else (str(default) if default is not None else "")


def styled_confirm(msg, default=True, allow_back=True):
    """Yes/no prompt with arrow-key selection.

    Returns bool. Raises GoBack on Escape (if allow_back).
    Falls back to text input if not a tty.
    """
    if not USE_COLOR or not sys.stdin.isatty():
        # Text fallback
        if default:
            hint = f"{highlight('Y')}/{dim('n')}"
        else:
            hint = f"{dim('y')}/{highlight('N')}"
        try:
            val = input(f"  {msg} [{hint}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            raise SystemExit(0)
        if allow_back:
            _check_back(val)
        if not val:
            return default
        return val.lower() in ("y", "yes")

    default_idx = 0 if default else 1
    result = styled_select(
        msg, [("y", "es"), ("n", "o")],
        default=default_idx, allow_back=allow_back,
    )
    return result == "y"


def styled_choice(msg, options, default=None, allow_back=True):
    """Prompt for a choice from a list of options.

    options: list of (key, label) tuples, e.g. [("p", "protein"), ("l", "ligand")]
    Returns the selected key.
    """
    opts_str = "  ".join(f"[{highlight(k)}]{label}" for k, label in options)
    if default is not None:
        suffix = f" [{value(default)}]"
    else:
        suffix = ""
    try:
        val = input(f"  {msg} ({opts_str}){suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        raise SystemExit(0)
    if allow_back:
        _check_back(val)
    return val.lower() if val else (default or "")


def styled_select(msg, options, default=0, allow_back=True):
    """Inline arrow-key selector for fixed choices.

    options: list of (key, label) tuples, e.g. [("p", "rotein"), ("l", "igand")]
    default: index of pre-selected option (0-based).
    Keys: Left/Right arrows to move, Enter to confirm, letter keys to jump+confirm.
    Escape raises GoBack if *allow_back* is True.

    Falls back to styled_choice if stdin is not a tty.
    Returns the selected key (lowercase).
    """
    if not USE_COLOR or not sys.stdin.isatty():
        default_key = options[default][0] if default < len(options) else None
        return styled_choice(msg, options, default=default_key, allow_back=allow_back)

    import select as _select
    import termios
    import tty

    selected = default
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    def render():
        parts = []
        for i, (key, label) in enumerate(options):
            full = f"{key}{label}"
            if i == selected:
                parts.append(highlight(f"[{full}]"))
            else:
                parts.append(dim(f" {full} "))
        line = " ".join(parts)
        hints = dim("\u2190/\u2192")
        if allow_back:
            option_keys = {k.lower() for k, _ in options}
            if 'b' not in option_keys:
                hints += " " + dim("b=back")
        sys.stdout.write(f"\r\033[K  {msg}: {line}  {hints}")
        sys.stdout.flush()

    try:
        tty.setcbreak(fd)
        render()

        while True:
            b = os.read(fd, 1)
            if not b:
                continue
            ch = b[0]  # byte value

            if ch in (13, 10):  # Enter — confirm
                key, label = options[selected]
                sys.stdout.write(f"\r\033[K  {msg}: {highlight(f'{key}{label}')}\n")
                sys.stdout.flush()
                return key
            elif ch == 0x1b:  # Escape sequence
                if _select.select([fd], [], [], 0.3)[0]:
                    rest = os.read(fd, 2)
                    if rest == b'[C':  # Right arrow
                        selected = (selected + 1) % len(options)
                        render()
                    elif rest == b'[D':  # Left arrow
                        selected = (selected - 1) % len(options)
                        render()
                    else:
                        # Unrecognized escape sequence — drain and go back
                        while _select.select([fd], [], [], 0.05)[0]:
                            os.read(fd, 1)
                        if allow_back:
                            sys.stdout.write('\n')
                            raise GoBack()
                        render()
                else:
                    # Bare Escape — go back
                    if allow_back:
                        sys.stdout.write('\n')
                        raise GoBack()
                    render()
            elif ch == 0x03:  # Ctrl+C
                sys.stdout.write('\n')
                raise KeyboardInterrupt
            elif ch == 0x04:  # Ctrl+D
                sys.stdout.write('\n')
                raise EOFError
            else:
                # Letter key — select matching option and confirm immediately
                c = chr(ch)
                for i, (key, label) in enumerate(options):
                    if c.lower() == key.lower():
                        selected = i
                        sys.stdout.write(
                            f"\r\033[K  {msg}: {highlight(f'{key}{label}')}\n"
                        )
                        sys.stdout.flush()
                        return key
                # 'b' as go-back (when not an option key)
                if c.lower() == 'b' and allow_back:
                    option_keys = {k.lower() for k, _ in options}
                    if 'b' not in option_keys:
                        sys.stdout.write('\n')
                        raise GoBack()
                # Unknown key — ignore
    except (KeyboardInterrupt, EOFError):
        sys.stdout.write('\n')
        raise
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def print_error(msg):
    """Print an error message."""
    print(f"  {error('Error:')} {msg}")


def print_warning(msg):
    """Print a warning message."""
    print(f"  {warning('Warning:')} {msg}")


def print_success(msg):
    """Print a success message."""
    print(f"  {success(msg)}")


def print_info(msg):
    """Print an info message."""
    print(f"  {info(msg)}")


def print_value(label, val):
    """Print a label: value pair."""
    print(f"  {label}: {value(str(val))}")
