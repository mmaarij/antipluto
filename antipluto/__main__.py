from __future__ import annotations

# Path bootstrap — ensures `import antipluto` resolves correctly when this file
# is executed directly as `python -m antipluto` without `pip install -e .`.
# __file__ is antipluto/__main__.py, so .parent.parent is the project root.
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

"""
__main__.py — CLI entry point for the Anti-PLUTO framework.
=========================================================

Anti-PLUTO — Anti-Phishing Lexical Utilities and Threat Observation.

Usage
-----
::

    # Run via module invocation (no install required):
    python -m antipluto <command>

    # Run via entry point (after: pip install -e .):
    antipluto <command>

    # ── Preprocessing ──────────────────────────────────────────────────────────
    # Run the full ingestion pipeline with the default configuration:
    antipluto preprocess

    # Run a single source only:
    antipluto preprocess --source nazario

    # Override export mode (full = all 20 MeAJOR features, minimal = core NLP only):
    antipluto preprocess --mode full

    # ── PII Masking ────────────────────────────────────────────────────────────
    # Apply PII masking to all preprocessed cohorts:
    antipluto mask --input datasets/datasets_preprocessed/

    # Mask a single JSONL file:
    antipluto mask --input datasets/datasets_preprocessed/human_written_phishing.jsonl

    # Mask LLM cohorts into a custom output directory:
    antipluto mask --input datasets/llm_cohorts/ --output-dir datasets/datasets_masked/

    # Best accuracy: transformer NER model with GPU batching:
    antipluto mask --input datasets/datasets_preprocessed/ --spacy-model en_core_web_trf --batch-size 64

    # ── Validation & Stats ─────────────────────────────────────────────────────
    antipluto validate --input datasets/datasets_preprocessed/human_written_benign.jsonl
    antipluto stats    --input datasets/datasets_preprocessed/human_written_phishing.jsonl

Windows Multiprocessing Guard
------------------------------
The ``if __name__ == \"__main__\":`` guard at the bottom of this file is
**mandatory** on Windows. Python's \"spawn\" start method for multiprocessing
re-imports the top-level script in each worker process. Without the guard,
each worker would attempt to re-launch the CLI and recursively spawn more
workers, causing an exponential process storm and eventual crash.

Never remove or bypass this guard.
"""

import io
import logging
import sys
from pathlib import Path
from typing import Optional

import click

# Ensure UTF-8 stdout/stderr across Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("antipluto")

# ---------------------------------------------------------------------------
# Default configuration path
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "default.yaml"


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(version="2.0.0", prog_name="antipluto")
def cli() -> None:
    """
    Anti-PLUTO — Anti-Phishing Lexical Utilities and Threat Observation.

    An end-to-end framework for email corpus preprocessing, PII masking,
    LLM cohort generation, dual-branch phishing classification, baseline
    evaluation, and real-time prediction serving.
    """


# ---------------------------------------------------------------------------
# `preprocess` command
# ---------------------------------------------------------------------------


@cli.command("preprocess")
@click.option(
    "--config",
    "config_path",
    default=str(_DEFAULT_CONFIG),
    show_default=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the YAML configuration file.",
)
@click.option(
    "--source",
    "source_filter",
    default=None,
    type=click.Choice(
        ["enron", "nazario", "spamassassin", "trec07", "nigerian_fraud"],
        case_sensitive=False,
    ),
    help="Process only the specified source. Omit to run all enabled sources.",
)
@click.option(
    "--mode",
    "mode_override",
    default=None,
    type=click.Choice(["full", "minimal", "custom"], case_sensitive=False),
    help="Override dataset export mode ('full', 'minimal', or 'custom').",
)
def preprocess_command(
    config_path: Path, source_filter: str | None, mode_override: str | None
) -> None:
    """
    Ingest raw email corpora and write cleaned JSONL datasets.

    Reads raw email corpora, applies cleaning and optional PII masking,
    validates records against the EmailRecord schema, and writes results to
    human_written_benign.jsonl and human_written_phishing.jsonl.

    \b
    Examples:
      antipluto preprocess
      antipluto preprocess --mode full
      antipluto preprocess --source nazario
      antipluto preprocess --config configs/custom.yaml
    """
    from antipluto.preprocessing.pipeline import PreprocessingPipeline

    logger.info(
        "CLI › Invoking pipeline. config=%s source=%s mode=%s",
        config_path,
        source_filter,
        mode_override,
    )
    pipeline = PreprocessingPipeline(config_path=config_path)
    pipeline.run(source_filter=source_filter, mode_override=mode_override)


# ---------------------------------------------------------------------------
# `mask` command
# ---------------------------------------------------------------------------


@cli.command("mask")
@click.option(
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help=(
        "Path to a single .jsonl file OR a directory of .jsonl files. "
        "When a directory is given, all .jsonl files inside it are masked."
    ),
)
@click.option(
    "--output-dir",
    "output_dir",
    default=None,
    type=click.Path(path_type=Path),
    help=(
        "Directory to write masked output files to. "
        "Defaults to datasets/datasets_masked/ relative to the project root "
        "defined in the config file. Output filenames match the input filenames."
    ),
)
@click.option(
    "--config",
    "config_path",
    default=str(_DEFAULT_CONFIG),
    show_default=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the YAML configuration file.",
)
@click.option(
    "--spacy-model",
    "spacy_model",
    default=None,
    help="Override the spaCy model name from the config (e.g. en_core_web_trf).",
)
@click.option(
    "--batch-size",
    "batch_size",
    default=32,
    show_default=True,
    type=int,
    help=(
        "Number of records per NER mini-batch. "
        "Increase (e.g. 64-128) for GPU runs; decrease if you hit out-of-memory errors."
    ),
)
@click.option(
    "--resume",
    "resume",
    is_flag=True,
    default=False,
    help=(
        "Resume a previously interrupted masking run. "
        "Counts the records already written to each output file and skips "
        "that many records in the input, then appends from where it left off. "
        "Safe to run even if the output file does not yet exist (starts fresh)."
    ),
)
def mask_command(
    input_path: Path,
    output_dir: Path | None,
    config_path: Path,
    spacy_model: str | None,
    batch_size: int,
    resume: bool,
) -> None:
    """
    Apply MeAJOR-compatible PII anonymisation to one or more existing JSONL files.

    Reads each record, applies the two-stage masking pipeline (spaCy NER + regex)
    to the 'subject' and 'body' fields, and writes anonymised records to the
    output directory. Original files are never modified.

    The 20 MeAJOR anonymisation tokens used are:

        [PGP], [EMOJI], [SYMBOL], [NAME], [USERNAME], [INITIALS],
        [EMAIL_ADDRESS], [PHONE_NUMBER], [ADDRESS], [ORGANIZATION],
        [URL], [IP_ADDRESS], [FILE_PATH], [FILE_NAME], [FILE],
        [DATE], [TIME], [FINANCIAL_INFO], [PRODUCT], [REFERENCE_NUMBER]

    \b
    Examples:
      antipluto mask --input datasets/datasets_preprocessed/
      antipluto mask --input datasets/datasets_preprocessed/human_written_phishing.jsonl
      antipluto mask --input datasets/llm_cohorts/ --output-dir datasets/datasets_masked/
      antipluto mask --input datasets/datasets_preprocessed/ --spacy-model en_core_web_trf --batch-size 64
      antipluto mask --input datasets/datasets_preprocessed/ --resume
    """
    import json
    import gc
    import yaml
    from antipluto.masking.masker import PIIMasker

    # How often (in batches) to force a full GC cycle.
    # spaCy's StringStore/vocab grows at the process level and is never freed
    # between batches; periodic GC reclaims orphaned Doc objects and prevents
    # gradual heap growth that slows long runs.
    GC_INTERVAL = 10

    # ── Resolve config ─────────────────────────────────────────────────────────
    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    project_root = Path(config["project_root"])
    model_name = spacy_model or config.get("masking", {}).get(
        "spacy_model", "en_core_web_sm"
    )

    # ── Resolve output directory ───────────────────────────────────────────────
    if output_dir is None:
        output_dir = project_root / "datasets" / "datasets_masked"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Collect input files ────────────────────────────────────────────────────
    if input_path.is_dir():
        input_files = sorted(input_path.glob("*.jsonl"))
        if not input_files:
            click.echo(f"No .jsonl files found in {input_path}", err=True)
            sys.exit(1)
    else:
        input_files = [input_path]

    click.echo(f"\nMasking {len(input_files)} file(s) -> {output_dir}")
    click.echo(f"spaCy model : {model_name}")
    click.echo(f"Batch size  : {batch_size}\n")

    # ── Load masker (expensive — done once for all files) ─────────────────────
    masker = PIIMasker(spacy_model=model_name, enabled=True)

    total_records = 0
    total_errors = 0
    batch_counter = 0  # tracks batches across all files for GC_INTERVAL

    for src_file in input_files:
        dst_file = output_dir / src_file.name

        # ── Resume: count already-written records ──────────────────────────
        already_done = 0
        if resume and dst_file.exists():
            with open(dst_file, "r", encoding="utf-8") as chk:
                for ln in chk:
                    if ln.strip():
                        already_done += 1
            click.echo(
                f"  {src_file.name}: resuming — skipping {already_done:,} already-masked records."
            )
        elif dst_file.exists() and not resume:
            # Non-resume mode: warn and overwrite.
            click.echo(
                f"  {src_file.name}: output exists — overwriting (use --resume to continue)."
            )

        open_mode = "a" if resume else "w"
        file_records = already_done  # so the final tally is accurate
        file_errors = 0

        # Fast non-parsing line count so we can show n/total progress.
        input_total = sum(1 for ln in open(src_file, "r", encoding="utf-8") if ln.strip())

        # ── Streaming batch loop ───────────────────────────────────────────
        # Reads the input line-by-line so multi-hundred-MB files never fully
        # load into memory. Records before `already_done` are skipped cheaply
        # (JSON-parsed but not sent to the NER model).
        batch_records: list[dict] = []
        batch_line_nums: list[int] = []
        input_pos = 0  # position in the input stream (across JSON-valid lines)

        def flush_batch(
            batch: list[dict],
            line_nums: list[int],
            out_fh,
        ) -> tuple[int, int]:
            """Mask and write one batch. Returns (written, errors)."""
            subjects = [r.get("subject", "") or "" for r in batch]
            bodies = [r.get("body", "") or "" for r in batch]
            try:
                masked_subjects, masked_bodies = masker.mask_records_batch(
                    subjects, bodies, batch_size=batch_size
                )
            except Exception as exc:
                logger.warning(
                    "mask › %s batch at line %d failed: %s",
                    src_file.name, line_nums[0], exc,
                )
                return 0, len(batch)
            written = 0
            for rec, ms, mb in zip(batch, masked_subjects, masked_bodies):
                rec["subject"] = ms
                rec["body"] = mb
                out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
            out_fh.flush()  # ensure data hits disk so resume can pick up cleanly
            return written, 0

        with open(dst_file, open_mode, encoding="utf-8") as out_fh:
            with open(src_file, "r", encoding="utf-8") as in_fh:
                for line_num, line in enumerate(in_fh, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        logger.warning(
                            "mask › %s line %d bad JSON: %s",
                            src_file.name, line_num, exc,
                        )
                        file_errors += 1
                        continue

                    input_pos += 1

                    # Skip records already written in a previous run.
                    if input_pos <= already_done:
                        continue

                    batch_records.append(record)
                    batch_line_nums.append(line_num)

                    if len(batch_records) >= batch_size:
                        written, errs = flush_batch(batch_records, batch_line_nums, out_fh)
                        file_records += written
                        file_errors += errs
                        batch_records = []
                        batch_line_nums = []
                        batch_counter += 1
                        if batch_counter % GC_INTERVAL == 0:
                            gc.collect()
                        # Overwrite the same line so the CLI stays clean.
                        click.echo(
                            f"\r  {src_file.name}: {file_records:,}/{input_total:,} masked...",
                            nl=False,
                        )

                # Flush any remaining partial batch.
                if batch_records:
                    written, errs = flush_batch(batch_records, batch_line_nums, out_fh)
                    file_records += written
                    file_errors += errs

        total_records += file_records
        total_errors += file_errors
        # Move to new line after the in-place progress counter.
        click.echo()
        click.echo(
            f"  {src_file.name:<45s}  "
            f"{file_records:>8,} records  "
            f"{file_errors} errors  -> {dst_file.name}"
        )

    click.echo(
        f"\nDone. Total masked: {total_records:,} records | Errors: {total_errors}"
    )


# ---------------------------------------------------------------------------
# `validate` command
# ---------------------------------------------------------------------------


@cli.command("validate")
@click.option(
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the .jsonl file to validate.",
)
@click.option(
    "--config",
    "config_path",
    default=str(_DEFAULT_CONFIG),
    show_default=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the YAML configuration file (used for schema settings).",
)
def validate_command(input_path: Path, config_path: Path) -> None:
    """
    Validate an existing JSONL file against the EmailRecord schema.

    Reads each line, parses the JSON, and runs it through the Pydantic
    validator. Prints a summary of valid and invalid records. Exits
    with code 1 if any invalid records are found.

    \b
    Example:
      antipluto validate --input datasets/datasets_preprocessed/human_written_benign.jsonl
    """
    from antipluto.preprocessing.pipeline import PreprocessingPipeline

    pipeline = PreprocessingPipeline(config_path=config_path)
    stats = pipeline.validate_file(input_path)

    click.echo(
        f"\nValidation Results — {input_path.name}\n"
        f"  Total  : {stats['total']}\n"
        f"  Valid  : {stats['valid']}\n"
        f"  Invalid: {stats['invalid']}\n"
    )

    if stats["invalid"] > 0:
        logger.warning(
            "CLI › Validation completed with %d invalid records.", stats["invalid"]
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# `stats` command
# ---------------------------------------------------------------------------


@cli.command("stats")
@click.option(
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the .jsonl file to inspect.",
)
def stats_command(input_path: Path) -> None:
    """
    Display record count and label/source distribution for a JSONL file.

    Scans the file line-by-line without loading it entirely into memory.

    \b
    Example:
      antipluto stats --input datasets/datasets_preprocessed/human_written_benign.jsonl
    """
    import json
    from collections import Counter

    total = 0
    source_counts: Counter = Counter()
    label_counts: Counter = Counter()
    errors = 0

    with open(input_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                total += 1
                source_counts[record.get("source", "unknown")] += 1
                label_counts[record.get("label", "?")] += 1
            except json.JSONDecodeError:
                errors += 1

    click.echo(f"\nStats — {input_path.name}")
    click.echo(f"  Total records : {total}")
    click.echo(f"  Parse errors  : {errors}")
    click.echo("\n  Label distribution:")
    for label, count in sorted(label_counts.items()):
        label_name = "benign" if label == 0 else "phishing"
        click.echo(f"    {label} ({label_name:8s}) : {count:>8,}")
    click.echo("\n  Source distribution:")
    for source, count in source_counts.most_common():
        click.echo(f"    {source:<15s} : {count:>8,}")
    click.echo()


# ---------------------------------------------------------------------------
# `generate` command
# ---------------------------------------------------------------------------


@cli.command("generate")
@click.option(
    "--cohort",
    "cohort",
    required=True,
    type=click.Choice(["benign", "phishing"], case_sensitive=False),
    help="Which LLM cohort to generate.",
)
@click.option(
    "--config",
    "config_path",
    default=str(_DEFAULT_CONFIG),
    show_default=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the YAML configuration file.",
)
@click.option(
    "--api-key",
    "api_key",
    default=None,
    help=(
        "Global API key override applied to all providers. "
        "Per-provider keys are configured via api_key_env in the YAML config."
    ),
)
@click.option(
    "--models",
    "models_override",
    default=None,
    multiple=True,
    help=(
        "Filter models to run. Only models matching these names (from the config) "
        "will be used. Repeat to specify multiple: --models gpt-5-mini --models Phi-4"
    ),
)
@click.option(
    "--n-samples",
    "n_samples",
    default=None,
    type=int,
    help="Override the total number of samples to generate (default: from config).",
)
@click.option(
    "--batch-size",
    "batch_size",
    default=None,
    type=int,
    help="Override emails per API call (default: from config).",
)
@click.option(
    "--language",
    "language_filter",
    default=None,
    help="Override language filter for seed sampling, e.g. 'en' (default: from config).",
)
@click.option(
    "--model",
    "model_filter",
    default=None,
    help="Use only this single model, ignoring the models list.",
)
@click.option(
    "--overwrite",
    "overwrite",
    is_flag=True,
    default=False,
    help="Overwrite existing output file instead of resuming.",
)
def generate_command(
    cohort: str,
    config_path: Path,
    api_key: str | None,
    models_override: tuple[str, ...],
    n_samples: int | None,
    batch_size: int | None,
    language_filter: str | None,
    model_filter: str | None,
    overwrite: bool,
) -> None:
    """
    Generate an LLM email cohort via configured provider APIs.

    Reads seed emails from the human-written JSONL dataset, sends them
    in batches to the configured LLM models (routed to their respective
    provider backends), and writes schema-validated output to the
    configured output directory.

    Total samples are divided evenly across all models in the config.
    Generation is resumable -- interrupted runs continue from where they
    left off (use --overwrite to start fresh).

    \b
    Examples:
      antipluto generate --cohort benign
      antipluto generate --cohort phishing --api-key <key>
      antipluto generate --cohort benign --model gpt-5-mini --n-samples 500
      antipluto generate --cohort phishing --models gpt-5-mini --models Phi-4
      antipluto generate --cohort benign --overwrite
    """
    import yaml
    from antipluto.generation.generator import CohortGenerator

    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    project_root = Path(config["project_root"])
    gen_cfg: dict = config.get("generation", {})

    # Apply CLI overrides.
    if models_override:
        # Filter the existing models list to only include the specified names.
        filtered = [
            m for m in gen_cfg.get("models", [])
            if isinstance(m, dict) and m.get("name") in models_override
        ]
        if not filtered:
            click.echo(
                f"Error: No configured models match --models {list(models_override)}. "
                f"Available: {[m['name'] for m in gen_cfg.get('models', []) if isinstance(m, dict)]}",
                err=True,
            )
            raise SystemExit(1)
        gen_cfg["models"] = filtered
    if n_samples is not None:
        gen_cfg["n_samples"] = n_samples
    if batch_size is not None:
        gen_cfg["batch_size"] = batch_size
    if language_filter is not None:
        gen_cfg["language_filter"] = language_filter

    generator = CohortGenerator(
        config=gen_cfg,
        project_root=project_root,
        api_key=api_key,
    )
    generator.run(
        cohort=cohort,  # type: ignore[arg-type]
        model_filter=model_filter,
        n_samples_override=n_samples,
        overwrite=overwrite,
    )


# ---------------------------------------------------------------------------
# `trace` command
# ---------------------------------------------------------------------------


@cli.command("trace")
@click.option(
    "--record",
    "record_str",
    default=None,
    help="Synthetic JSON record string to trace.",
)
@click.option(
    "--file",
    "file_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to a JSON file containing a single synthetic record.",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_CONFIG,
    show_default=True,
    help="Path to YAML configuration file.",
)
def trace_command(
    record_str: str | None,
    file_path: Path | None,
    config_path: Path,
) -> None:
    """
    Trace a synthetic email record back to its original human seed email.

    Outputs the full original human email record as JSON.

    \b
    Examples:
      antipluto trace --record '{"source": "gpt-5-mini", "seed_idx": 42, "label": 1}'
      antipluto trace --file synthetic_sample.json
    """
    import json
    import yaml
    from antipluto.generation.provenance import ProvenanceTracer

    if record_str is None and file_path is None:
        raise click.UsageError("Provide either --record '<json>' or --file <path>.")

    if file_path is not None:
        content = file_path.read_text(encoding="utf-8").strip()
    else:
        content = record_str or ""

    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    project_root = Path(config["project_root"])
    tracer = ProvenanceTracer(
        config=config,
        config_path=config_path,
        project_root=project_root,
    )

    try:
        original = tracer.trace(content)
        click.echo(json.dumps(original, indent=2, ensure_ascii=False))
    except Exception as exc:
        raise click.ClickException(f"Trace failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Classifier Commands (Train, Export, Predict)
# ---------------------------------------------------------------------------


@cli.command("train")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_CONFIG,
    show_default=True,
    help="Path to YAML configuration file.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory for checkpoints and evaluation results.",
)
@click.option(
    "--sample-size",
    type=int,
    default=None,
    help="Number of records to sample per cohort (default: 16,000, -1 for all).",
)
@click.option(
    "--epochs",
    type=int,
    default=None,
    help="DeBERTa fine-tuning epochs (default from config: 3).",
)
@click.option(
    "--batch-size",
    type=int,
    default=None,
    help="Batch size for training and embedding extraction (default: 16).",
)
@click.option(
    "--resume/--no-resume",
    default=True,
    show_default=True,
    help="Resume training from existing checkpoints if available.",
)
@click.option(
    "--classifier",
    "classifier",
    type=click.Choice(["xgboost", "rf", "mlp", "cnn", "all"], case_sensitive=False),
    default="xgboost",
    show_default=True,
    help="Classifier fusion head to train (xgboost, rf, mlp, cnn, or all).",
)
@click.option(
    "--force-retrain-deberta",
    is_flag=True,
    default=False,
    help="Force retraining DeBERTa even if a completed checkpoint exists.",
)
def train_command(
    config_path: Path,
    output_dir: Path | None,
    sample_size: int | None,
    epochs: int | None,
    batch_size: int | None,
    resume: bool,
    classifier: str,
    force_retrain_deberta: bool,
) -> None:
    """
    Train the Anti-PLUTO dual-branch classifier with the selected fusion head (XGBoost, RF, MLP, 1D-CNN).

    Supports phase-level state saving and pause/resume at any epoch or stage.
    """
    import yaml
    from antipluto.classifier.trainer import ClassifierTrainer

    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    cls_cfg = config.get("classifier", {})
    project_root = Path(config.get("project_root", "."))

    out_dir = output_dir or (project_root / cls_cfg.get("output_dir", "models"))
    data_dir = project_root / cls_cfg.get("data_dir", "datasets/datasets_masked")

    samples = sample_size if sample_size is not None else cls_cfg.get("samples_per_cohort", 16000)
    if samples <= 0:
        samples = None

    sem_cfg = cls_cfg.get("semantic", {})
    xgb_cfg = cls_cfg.get("fusion", {})

    trainer = ClassifierTrainer(
        output_dir=out_dir,
        data_dir=data_dir,
        samples_per_cohort=samples,
        model_name=sem_cfg.get("model_name", "microsoft/deberta-v3-small"),
        epochs=epochs if epochs is not None else sem_cfg.get("epochs", 3),
        batch_size=batch_size if batch_size is not None else sem_cfg.get("batch_size", 16),
        learning_rate=float(sem_cfg.get("learning_rate", 2e-5)),
        classifier=classifier,
        xgb_n_estimators=xgb_cfg.get("n_estimators", 500),
        xgb_learning_rate=float(xgb_cfg.get("learning_rate", 0.08)),
        xgb_max_depth=xgb_cfg.get("max_depth", 6),
        xgb_device=xgb_cfg.get("device", "cpu"),
        seed=cls_cfg.get("seed", 42),
    )

    try:
        results = trainer.run(resume=resume, force_deberta_retrain=force_retrain_deberta)
        if results and "four_class" in results:
            acc = results["four_class"]["accuracy"]
            f1 = results["four_class"]["macro_f1"]
            click.echo(f"\nTraining successfully finished! 4-Class Accuracy: {acc:.4f}, Macro F1: {f1:.4f}")
        else:
            click.echo("\nTraining successfully completed!")
    except Exception as exc:
        logger.exception("Training failed: %s", exc)
        raise click.ClickException(f"Classifier training failed: {exc}") from exc


@cli.command("compare")
@click.option(
    "--model-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("models"),
    show_default=True,
    help="Directory containing trained models and cached splits.",
)
@click.option(
    "--force-retrain",
    is_flag=True,
    default=False,
    help="Force retraining all heads even if checkpoints exist.",
)
def compare_command(model_dir: Path, force_retrain: bool) -> None:
    """
    Benchmark and compare all classifier heads (XGBoost, RF, MLP, 1D-CNN) side-by-side.

    Evaluates all heads on the exact same 6,400 unseen test samples and generates
    comparative tables for 4-class accuracy, phishing recall, FPR, attribution, and latency.
    """
    from antipluto.classifier.benchmark import run_benchmark

    try:
        run_benchmark(model_dir=model_dir, force_retrain=force_retrain)
    except Exception as exc:
        logger.exception("Benchmark comparison failed: %s", exc)
        raise click.ClickException(f"Benchmark comparison failed: {exc}") from exc


@cli.command("export")
@click.option(
    "--model-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("models"),
    show_default=True,
    help="Directory containing trained model checkpoints.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Directory to save exported ONNX models (default: <model-dir>/onnx).",
)
@click.option(
    "--classifier",
    "classifier",
    type=click.Choice(["xgboost", "rf", "mlp", "cnn", "all"], case_sensitive=False),
    default="all",
    show_default=True,
    help="Classifier head(s) to export to ONNX.",
)
def export_command(model_dir: Path, output_dir: Path | None, classifier: str) -> None:
    """
    Export fine-tuned DeBERTa and classifier heads to ONNX format for portable deployment.
    """
    from antipluto.classifier.exporter import (
        export_deberta_onnx,
        export_xgboost_onnx,
        export_rf_onnx,
        export_mlp_onnx,
        export_cnn_onnx,
    )
    from antipluto.classifier.heads import (
        XGBoostFusionHead,
        RandomForestFusionHead,
        MLPFusionHead,
        CNNFusionHead,
        AVAILABLE_HEADS,
    )
    from antipluto.classifier.benchmark import CHECKPOINT_CONFIGS
    from antipluto.classifier.stylometric import StylometricExtractor

    model_dir.mkdir(parents=True, exist_ok=True)
    out_dir = output_dir or (model_dir / "onnx")
    out_dir.mkdir(parents=True, exist_ok=True)

    deberta_dir = model_dir / "checkpoints" / "deberta" / "final_model"
    tfidf_path = model_dir / "checkpoints" / "tfidf_pipeline.joblib"

    if not tfidf_path.exists() or not deberta_dir.exists():
        raise click.ClickException(
            f"Base model artifacts not found in '{model_dir}'. Run training first."
        )

    click.echo("Exporting DeBERTa to ONNX...")
    export_deberta_onnx(
        model_dir=deberta_dir,
        output_path=out_dir / "deberta_cls.onnx",
    )

    stylo = StylometricExtractor.load(tfidf_path)
    total_features = stylo.total_features + 768

    heads_to_export = AVAILABLE_HEADS if classifier.lower() == "all" else [classifier.lower()]

    for h in heads_to_export:
        cfg = CHECKPOINT_CONFIGS[h]
        ckpt_path = model_dir / cfg["rel_path"]
        if not ckpt_path.exists():
            click.echo(f"Skipping {h.upper()} (checkpoint {ckpt_path} not found).")
            continue

        click.echo(f"Exporting {h.upper()} to ONNX...")
        clf = cfg["class"].load(ckpt_path)
        out_onnx_name = f"{h}_fusion.onnx"
        clf.export_onnx(out_dir / out_onnx_name, num_features=total_features)

    click.echo(f"ONNX export completed! Models saved in: {out_dir}")


@cli.command("predict")
@click.option("--subject", default="", help="Email subject line.")
@click.option("--body", default="", help="Email body text.")
@click.option(
    "--file",
    "file_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to file containing email content or JSON record.",
)
@click.option(
    "--classifier",
    "classifier",
    type=click.Choice(["xgboost", "rf", "mlp", "cnn"], case_sensitive=False),
    default="xgboost",
    show_default=True,
    help="Classifier head to use for inference (default: xgboost).",
)
@click.option(
    "--mode",
    "mode",
    type=click.Choice(["fusion", "semantic", "stylometric"], case_sensitive=False),
    default="fusion",
    show_default=True,
    help="Inference representation mode: 'fusion' (dual-branch), 'semantic' (DeBERTa-only), or 'stylometric' (TF-IDF only).",
)
@click.option(
    "--model-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("models"),
    show_default=True,
    help="Directory containing trained model checkpoints.",
)
@click.option(
    "--mask/--no-mask",
    default=True,
    show_default=True,
    help="Apply MeAJOR PII masking (URLs, emails, names) before classification.",
)
def predict_command(
    subject: str,
    body: str,
    file_path: Path | None,
    classifier: str,
    mode: str,
    model_dir: Path,
    mask: bool,
) -> None:
    """
    Classify a single email into one of the 4 cohorts (Human Benign, Human Phish, LLM Benign, LLM Phish).
    """
    import json
    from antipluto.api.service import InferenceService

    model_dir = Path(model_dir).resolve()

    if file_path is not None:
        raw_text = file_path.read_text(encoding="utf-8").strip()
        try:
            data = json.loads(raw_text)
            subject = data.get("subject", subject)
            body = data.get("body", body)
        except json.JSONDecodeError:
            if not body:
                body = raw_text

    if not subject and not body:
        raise click.UsageError("Provide email content via --subject, --body, or --file.")

    click.echo(f"Initializing InferenceService in {mode.upper()} mode (classifier: {classifier.upper()})...")
    service = InferenceService(
        model_dir=model_dir,
        classifier_name=classifier,
        default_mode=mode,
    )
    if not mask:
        service.pipeline.masker.enabled = False

    try:
        service.load_artifacts()
    except Exception as exc:
        raise click.ClickException(f"Failed to load model artifacts: {exc}") from exc

    res = service.predict(subject=subject, body=body, mode=mode)

    click.echo("\n" + "=" * 54)
    click.echo(f"  Mode           : {res.metadata.mode.upper()}")
    click.echo(f"  Classifier     : {res.metadata.classifier.upper()}")
    click.echo(f"  Prediction     : {res.prediction}")
    click.echo(f"  Is Phishing    : {'YES' if res.is_phishing else 'NO'} (Prob: {res.phishing_probability * 100:.2f}%)")
    click.echo(f"  Is LLM Gen     : {'YES' if res.is_llm_generated else 'NO'}")
    click.echo(f"  Latency        : {res.metadata.latency_ms:.2f} ms")
    click.echo("-" * 54)
    for name, conf in res.confidence_scores.items():
        click.echo(f"    {name:18s}: {conf * 100:6.2f}%")
    click.echo("=" * 54)


# ---------------------------------------------------------------------------
# antipluto baseline
# ---------------------------------------------------------------------------


@cli.command("baseline")
@click.option(
    "--engine",
    type=click.Choice(["all", "spamassassin", "rspamd", "antipluto"], case_sensitive=False),
    default="all",
    show_default=True,
    help="Target email security engine(s) to benchmark.",
)
@click.option(
    "--max-samples",
    type=int,
    default=None,
    help="Limit benchmark to the first N test samples (useful for quick dry-runs).",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output path for benchmark metrics JSON (defaults to models/evaluation/external_baselines_comparison.json).",
)
def baseline_cmd(engine: str, max_samples: Optional[int], output: Optional[Path]) -> None:
    """Benchmark Anti-PLUTO against industry-standard baselines (SpamAssassin & Rspamd).

    Evaluates on the exact 6,400 unseen test emails wrapped in standardized
    RFC 5322 MIME envelopes with neutral transmission headers.
    """
    from antipluto.baseline import evaluate_external_baselines

    engines = ["spamassassin", "rspamd", "antipluto"] if engine.lower() == "all" else [engine.lower()]
    evaluate_external_baselines(
        project_root=Path.cwd(),
        engines=engines,
        max_samples=max_samples,
        output_json=output,
    )


# ---------------------------------------------------------------------------
# antipluto serve
# ---------------------------------------------------------------------------


@cli.command("serve")
@click.option(
    "--host",
    type=str,
    default="127.0.0.1",
    show_default=True,
    help="Host address to bind the REST API server to.",
)
@click.option(
    "--port",
    type=int,
    default=8000,
    show_default=True,
    help="Port to listen on.",
)
@click.option(
    "--model-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default="models",
    show_default=True,
    help="Path to models directory containing checkpoints/.",
)
@click.option(
    "--classifier",
    type=click.Choice(["xgboost", "rf", "mlp", "cnn"], case_sensitive=False),
    default="xgboost",
    show_default=True,
    help="Active classification head to serve.",
)
@click.option(
    "--mode",
    type=click.Choice(["fusion", "semantic", "stylometric"], case_sensitive=False),
    default="fusion",
    show_default=True,
    help="Default inference representation mode served by the API (fusion, semantic, or stylometric).",
)
@click.option(
    "--reload",
    is_flag=True,
    default=False,
    help="Enable auto-reload for local development.",
)
def serve_cmd(
    host: str,
    port: int,
    model_dir: Path,
    classifier: str,
    mode: str,
    reload: bool,
) -> None:
    """Launch the production FastAPI inference server for Anti-PLUTO.

    Exposes REST endpoints for real-time email security prediction:
      - POST /api/v1/predict     (JSON email payloads)
      - POST /api/v1/predict/raw (Raw RFC 5322 MIME .eml uploads)
      - GET  /api/v1/health      (System health and model readiness)
      - GET  /docs               (Interactive OpenAPI / Swagger documentation)
    """
    import uvicorn
    from antipluto.api.app import create_app

    click.echo("\n" + "=" * 60)
    click.echo("  Starting Anti-PLUTO Email Security API Server")
    click.echo("=" * 60)
    click.echo(f"  Host       : {host}")
    click.echo(f"  Port       : {port}")
    click.echo(f"  Classifier : {classifier.upper()}")
    click.echo(f"  Mode       : {mode.upper()}")
    click.echo(f"  Model Dir  : {model_dir}")
    click.echo(f"  Swagger UI : http://{host}:{port}/docs")
    click.echo("=" * 60 + "\n")

    app = create_app(model_dir=model_dir, classifier=classifier, default_mode=mode)
    uvicorn.run(app, host=host, port=port, reload=reload)



# ---------------------------------------------------------------------------
# Windows multiprocessing spawn guard — DO NOT REMOVE
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # This guard is required on Windows to prevent recursive worker spawning
    # when ProcessPoolExecutor uses the "spawn" start method.
    # See: https://docs.python.org/3/library/multiprocessing.html#the-spawn-and-forkserver-start-methods
    cli()

