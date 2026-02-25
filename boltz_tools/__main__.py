import sys

HELP = """\
boltz_tools — Boltz structure prediction workflow toolkit

Usage:
  boltz-setup <job_name>                Set up and submit a Boltz job (interactive wizard)
  boltz-setup --resume [JOB_NAME]       Add new inputs to an existing job
  boltz-setup log <job_dir> [--file f]  Parse Slurm output into a clean log

Commands:
  <job_name>    Launch the job setup wizard. Walks through entities, constraints,
                Boltz parameters, and Slurm settings, then writes files and submits.
                Jobs are created under $BOLTZ_JOBS_DIR (default: /scratch/$LOGNAME/boltz_jobs/).

  yaml          (boltz-setup-yaml)
                Non-interactive YAML generation. Takes all entity/constraint/property
                definitions as flags. Use for scripting and batch pipelines.
                See: boltz-setup-yaml --help

  --resume      Add new YAML inputs to an existing job directory. Old inputs are
                kept; boltz automatically skips already-processed inputs on re-run.
                Offers to pre-populate wizard state from an existing YAML in the job,
                so you can tweak entities/constraints rather than re-entering everything.
                If JOB_NAME is omitted, shows a picker of existing job directories.
                Scripts are written as job_1.sh, job_2.sh, etc. (never overwriting job.sh).

  log           Parse a finished job's slurm-*.out and produce a clean summary file
                named <name>_<STATUS>_<jobid>.log with timing, resource usage, and
                confidence scores. Runs automatically at the end of each job, but
                can also be run manually.

                <job_dir> can be an absolute path or just a job name.

Job scripts:
  On submission (whether via the wizard or manual sbatch), each script renames
  itself to embed the Slurm job ID: job.sh -> job_12345.sh, job_1.sh -> job_1_12345.sh.
  This makes it easy to match scripts to squeue/sacct entries in busy job directories.

Slurm settings:
  GPU type, memory, time limit, and partition are auto-recommended based on sequence
  length and boltz parameters. The wizard shows a single overview — press Enter to
  accept all, or 'n' to customize individual settings.

Options:
  -h, --help    Show this help message

Examples:
  boltz-setup myjob                     # set up a new job
  boltz-setup --resume myjob            # add inputs to existing job
  boltz-setup --resume                  # pick from existing jobs
  boltz-setup log myjob                 # parse results after completion
  boltz-setup log /scratch/$LOGNAME/boltz_jobs/myjob --file slurm-12345.out
"""

if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
    print(HELP)
    sys.exit(0)
elif len(sys.argv) > 1 and sys.argv[1] == "log":
    from .logparse import log_main
    log_main()
else:
    from .cluster import check_boltz_installation
    installed, latest, needs_update = check_boltz_installation()

    if installed is None:
        print("\033[33m  Warning: boltz is not installed.\033[0m")
        print("  Install with:  pip install --user boltz")
        ans = input("  Continue anyway? [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            sys.exit(1)
    elif needs_update:
        print(f"\033[33m  Update available: boltz {installed} → {latest}\033[0m")
        print("  Update with:  pip install --user --upgrade boltz")
        print()

    from .cli import main
    main()
