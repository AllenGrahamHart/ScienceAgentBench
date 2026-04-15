"""
Modal-based parallel evaluation for ScienceAgentBench.

Replicates the local eval harness (evaluation/harness/run_evaluation.py) but runs
all 102 instances in parallel on Modal cloud compute, reducing wall-clock time
from 16-20 hours to ~30-60 minutes.

Usage:
    # One-time: upload benchmark data to Modal Volume
    modal run run_eval_modal.py::upload_benchmark

    # Run evaluation (all 102 instances)
    modal run run_eval_modal.py::main \
        --pred-program-path pred_programs_cc_run1 \
        --log-fname eval_cc_run1_modal.jsonl \
        --run-id cc_run1_modal

    # Run subset
    modal run run_eval_modal.py::main \
        --pred-program-path pred_programs_cc_run1 \
        --log-fname eval_test_modal.jsonl \
        --run-id cc_test_modal \
        --instance-ids "3,8,20"
"""

from __future__ import annotations

import json
import os
import subprocess
import shutil
import sys
from pathlib import Path

import modal

app = modal.App("sab-eval")

# Base image built from the same Dockerfile used by the local harness.
# Modal caches this layer so it's only built once.
sab_image = (
    modal.Image.from_dockerfile("Dockerfile.sab_base")
    .add_local_file("compute_scores.py", "/testbed/compute_scores.py")
    .add_local_file("gpt4_visual_judge.py", "/testbed/gpt4_visual_judge.py")
)

benchmark_vol = modal.Volume.from_name("sab-benchmark", create_if_missing=True)

# ---------------------------------------------------------------------------
# Version pinning maps — replicates the sed commands from dockerfiles.py
# ---------------------------------------------------------------------------
VERSION_PINS = {
    "numpy": "numpy<2.0",
    "scipy": "scipy<1.14.0",
    "matplotlib": "matplotlib<3.8.0",
    "torch": "torch<=2.3.0",
    "tensorflow": "tensorflow<=2.17.0",
    "rdkit": "rdkit<=2023.09.5",
    "tf_keras": "tf_keras<=2.17.0",
    "pymatgen": "pymatgen<=2024.5.1",
    "oggm": "oggm<=1.6.1",
}

PACKAGE_RENAMES = {
    "scvi": "scvi-tools",
    "iris": "scitools-iris",
    "skimage": "scikit-image",
}

REMOVE_PACKAGES = {"benchmark"}

# Extra companion deps keyed by package name
COMPANION_DEPS = {
    "oggm": ["salem", "tables", "geopandas"],
    "scanpy": ["scikit-misc", "leidenalg"],
    "biopsykit": ["mne"],
}

# Post-install steps keyed by package name
POST_INSTALL = {
    "deepchem": ["dgl", "-f", "https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html"],
    "DeepPurpose": ["git+https://github.com/bp-kelley/descriptastorus"],
}


def _install_instance_deps(pred_program_path: str) -> str:
    """Run pipreqs on the predicted program, apply version pinning, install deps.

    Returns the pip install stdout+stderr log.
    """
    pip = "/opt/miniconda3/bin/pip"
    pipreqs = "/opt/miniconda3/bin/pipreqs"
    reqs_file = "/testbed/instance_requirements.txt"

    # Extract imports via pipreqs
    result = subprocess.run(
        [pipreqs, pred_program_path, f"--savepath={reqs_file}", "--mode", "no-pin"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"pipreqs warning: {result.stderr}", file=sys.stderr)

    # Read extracted packages
    if os.path.exists(reqs_file):
        with open(reqs_file) as f:
            pkgs = [line.strip() for line in f if line.strip()]
    else:
        pkgs = []

    # --- Package transformations (mirrors dockerfiles.py lines 51-83) ---

    # Check for qsprpred special case (early return in original)
    if "qsprpred" in pkgs:
        subprocess.run(
            [pip, "install", "git+https://github.com/martin-sicho/papyrus-scaffold-visualizer.git@main"],
            capture_output=True,
        )
        subprocess.run([pip, "install", "kaleido"], capture_output=True)
        # Still install the rest of the requirements
        subprocess.run(
            [pip, "install", "--exists-action", "i", "--no-cache-dir", "-r", reqs_file],
            capture_output=True,
        )
        return "qsprpred special path"

    # Renames (scvi -> scvi-tools, iris -> scitools-iris, etc.)
    new_pkgs = []
    for pkg in pkgs:
        base = pkg.split("==")[0].split(">=")[0].split("<=")[0].split("<")[0].split(">")[0]
        if base in REMOVE_PACKAGES:
            continue
        if base in PACKAGE_RENAMES:
            new_pkgs.append(PACKAGE_RENAMES[base])
        else:
            new_pkgs.append(pkg)
    pkgs = new_pkgs

    # Companion deps
    pkg_names = {p.split("==")[0].split(">=")[0].split("<=")[0].split("<")[0].split(">")[0] for p in pkgs}
    for trigger, extras in COMPANION_DEPS.items():
        if trigger in pkg_names:
            pkgs.extend(extras)

    # Version pinning
    pinned = []
    for pkg in pkgs:
        base = pkg.split("==")[0].split(">=")[0].split("<=")[0].split("<")[0].split(">")[0]
        if base in VERSION_PINS:
            pinned.append(VERSION_PINS[base])
        else:
            pinned.append(pkg)
    pkgs = pinned

    # Write back
    with open(reqs_file, "w") as f:
        f.write("\n".join(pkgs) + "\n")

    # Install
    result = subprocess.run(
        [pip, "install", "--exists-action", "i", "--no-cache-dir", "-r", reqs_file],
        capture_output=True,
        text=True,
    )
    install_log = result.stdout + "\n" + result.stderr

    # Post-install steps (deepchem -> dgl, DeepPurpose -> descriptastorus)
    for trigger, pip_args in POST_INSTALL.items():
        if trigger in pkg_names:
            r = subprocess.run(
                [pip, "install"] + pip_args,
                capture_output=True,
                text=True,
            )
            install_log += f"\n[post-install {trigger}] {r.stdout}\n{r.stderr}"

    return install_log


@app.function(
    image=sab_image,
    volumes={"/testbed/benchmark": benchmark_vol},
    timeout=1800,
    memory=16384,
)
def evaluate_instance(instance_data: dict) -> dict:
    """Evaluate a single SAB instance inside Modal.

    Args:
        instance_data: dict with keys:
            - instance_id: str
            - test_spec: dict (serialized TestSpec fields)
            - pred_program_content: str (source code of predicted program)
            - pred_program_name: str (e.g. "pred_foo.py")
            - openai_api_key: str (optional, for visual judge tasks)

    Returns:
        dict with keys: instance_id, result_tuple, stdout, stderr, install_log
    """
    os.chdir("/testbed")

    instance_id = instance_data["instance_id"]
    test_spec = instance_data["test_spec"]
    pred_content = instance_data["pred_program_content"]
    pred_name = instance_data["pred_program_name"]  # e.g. "pred_foo.py"
    gold_program_name = test_spec["gold_program_name"]

    # Set OpenAI API key for visual judge tasks (passed through args, not secrets)
    openai_key = instance_data.get("openai_api_key", "")
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key

    print(f"=== Evaluating instance {instance_id} ({gold_program_name}) ===")

    # 1. Write predicted program to expected locations
    program_eval_dir = Path("/testbed/program_to_eval")
    program_eval_dir.mkdir(parents=True, exist_ok=True)
    (program_eval_dir / pred_name).write_text(pred_content)
    # Also needs __init__.py for python -m imports
    (program_eval_dir / "__init__.py").touch()

    pred_programs_dir = Path("/testbed/pred_programs")
    pred_programs_dir.mkdir(parents=True, exist_ok=True)
    (pred_programs_dir / pred_name).write_text(pred_content)

    # 2. Install instance-specific dependencies
    install_log = _install_instance_deps("/testbed/program_to_eval/")

    # 3. Set up instance_path (input.json, output/, pred_results/)
    instance_path = Path("/testbed/instance_path")
    (instance_path / "input").mkdir(parents=True, exist_ok=True)
    (instance_path / "output").mkdir(parents=True, exist_ok=True)
    (instance_path / "pred_results").mkdir(parents=True, exist_ok=True)

    with open(instance_path / "input" / "input.json", "w") as f:
        json.dump(test_spec, f)

    # 4. compute_scores.py creates the pred_results symlink itself, so just
    #    ensure no stale file/dir exists at that path.
    pred_results_link = Path("/testbed/pred_results")
    if pred_results_link.is_symlink():
        pred_results_link.unlink()
    elif pred_results_link.exists():
        shutil.rmtree(pred_results_link)

    # 5. Reload the volume to ensure benchmark data is visible
    benchmark_vol.reload()

    # 6. Patch gpt4_visual_judge.py with custom retry logic for 429 rate limits.
    #    With 64 visual tasks hitting the API concurrently, the SDK's built-in
    #    exponential backoff (max ~8s between retries) is too short.  We inject a
    #    _call_with_retry() wrapper that starts at 60s and doubles up to 480s,
    #    giving the rate-limit window time to reset.
    #    Guard with "_call_with_retry" check so the patch is idempotent.
    judge_path = Path("/testbed/gpt4_visual_judge.py")
    if judge_path.exists():
        judge_code = judge_path.read_text()
        if "_call_with_retry" not in judge_code:
            # Add imports for retry logic
            judge_code = judge_code.replace(
                "from openai import OpenAI, AzureOpenAI",
                "import time\nimport random\nfrom openai import OpenAI, AzureOpenAI, RateLimitError",
            )
            # Disable SDK-level retry; we handle it ourselves
            judge_code = judge_code.replace(
                "client = OpenAI()",
                "client = OpenAI(max_retries=0, timeout=120.0)",
            )
            judge_code = judge_code.replace(
                "client = AzureOpenAI(",
                "client = AzureOpenAI(max_retries=0, timeout=120.0, ",
            )
            # Route both create() calls through the retry wrapper
            # (must happen BEFORE injecting the helper, which itself
            #  calls client.chat.completions.create)
            judge_code = judge_code.replace(
                "client.chat.completions.create(",
                "_call_with_retry(",
            )
            # Inject the retry helper before score_figure()
            retry_helper = '''
def _call_with_retry(**kwargs):
    """Retry wrapper with long backoff for 429 rate limits."""
    wait = 60
    for attempt in range(10):
        try:
            return client.chat.completions.create(**kwargs)
        except RateLimitError:
            if attempt == 9:
                raise
            jitter = random.uniform(0, wait * 0.25)
            print(f"[visual_judge] 429 rate limit, waiting {wait + jitter:.0f}s (attempt {attempt + 1}/10)")
            time.sleep(wait + jitter)
            wait = min(wait * 2, 480)

'''
            judge_code = judge_code.replace(
                "def score_figure(",
                retry_helper + "def score_figure(",
            )
            judge_path.write_text(judge_code)

    # 7. Run compute_scores.py (replicates: python compute_scores.py)
    result = subprocess.run(
        ["python", "/testbed/compute_scores.py"],
        capture_output=True,
        text=True,
        timeout=1200,  # 20-min overall timeout for compute_scores
        cwd="/testbed",
    )

    stdout = result.stdout
    stderr = result.stderr
    print(f"[{instance_id}] stdout: {stdout[:500]}")
    if stderr:
        print(f"[{instance_id}] stderr: {stderr[:500]}", file=sys.stderr)

    # 8. Read result.json
    result_path = instance_path / "output" / "result.json"
    if result_path.exists():
        with open(result_path) as f:
            result_tuple = json.load(f)
    else:
        result_tuple = [0, 0.0, 0, f"No result.json produced. stderr: {stderr[:1000]}"]

    return {
        "instance_id": instance_id,
        "result_tuple": result_tuple,
        "stdout": stdout[-2000:],  # Truncate to keep serialization small
        "stderr": stderr[-2000:],
        "install_log": install_log[-2000:],
    }


@app.local_entrypoint()
def upload_benchmark(benchmark_path: str = "benchmark"):
    """One-time upload of benchmark/ directory to the Modal Volume.

    This uploads ~3.8GB of data (datasets, eval_programs, gold_programs).
    """
    vol = modal.Volume.from_name("sab-benchmark", create_if_missing=True)
    bp = Path(benchmark_path)

    if not bp.exists():
        print(f"Error: benchmark path {bp.resolve()} does not exist")
        sys.exit(1)

    # Create a persistent temp file for __init__.py stubs (must outlive batch_upload)
    import tempfile
    tmp_dir = tempfile.mkdtemp()
    init_file = Path(tmp_dir) / "__init__.py"
    init_file.write_text("")

    with vol.batch_upload(force=True) as batch:
        for subdir in ["datasets", "eval_programs", "gold_programs"]:
            src = bp / subdir
            if not src.exists():
                print(f"Warning: {src} does not exist, skipping")
                continue
            print(f"Uploading {src}/ -> /{subdir}/ ...")
            batch.put_directory(str(src), f"/{subdir}")

        # Upload __init__.py stubs for Python imports
        batch.put_file(str(init_file), "/__init__.py")
        batch.put_file(str(init_file), "/eval_programs/__init__.py")

    # Clean up after batch is done
    shutil.rmtree(tmp_dir)
    print("Upload complete.")


@app.local_entrypoint()
def main(
    pred_program_path: str = "pred_programs_cc_run1",
    log_fname: str = "eval_cc_run1_modal.jsonl",
    run_id: str = "cc_run1_modal",
    openai_api_key: str = "",
    instance_ids: str = "",
    max_concurrent: int = 0,
    benchmark_path: str = "benchmark",
    dataset_name: str = "osunlp/ScienceAgentBench",
    split: str = "validation",
):
    """Run SAB evaluation in parallel on Modal.

    Args:
        pred_program_path: Local directory containing predicted programs.
        log_fname: Output JSONL file path (same format as local harness).
        run_id: Run identifier.
        openai_api_key: OpenAI API key for visual judge tasks.
        instance_ids: Comma-separated instance IDs to evaluate (empty = all).
        max_concurrent: Maximum number of Modal eval tasks to run at once.
            0 means launch all selected instances together.
        benchmark_path: Local path to benchmark/ directory.
        dataset_name: HuggingFace dataset name.
        split: Dataset split.
    """
    from datasets import load_dataset

    # Resolve the OpenAI API key (CLI arg takes precedence over env var)
    resolved_openai_key = openai_api_key or os.environ.get("OPENAI_API_KEY", "")
    if not resolved_openai_key:
        print("Warning: No OPENAI_API_KEY set. Visual judge tasks will fail.")

    # Parse instance_ids filter
    filter_ids = set()
    if instance_ids:
        filter_ids = {s.strip() for s in instance_ids.split(",") if s.strip()}

    # Load HuggingFace dataset
    print(f"Loading dataset {dataset_name} (split={split})...")
    dataset = load_dataset(dataset_name, split=split)
    num_instances = len(dataset)
    print(f"Dataset has {num_instances} instances.")

    # Resume support: read existing log to skip already-evaluated instances
    evaluated_indices = set()
    evaluated_logs = [None] * num_instances
    log_path = Path(log_fname)
    if log_path.exists():
        with open(log_path) as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if line:
                    evaluated_indices.add(idx)
                    evaluated_logs[idx] = json.loads(line)
        print(f"Resuming: {len(evaluated_indices)} instances already evaluated.")

    # Build instance data list
    pred_dir = Path(pred_program_path)
    if not pred_dir.exists():
        print(f"Error: pred_program_path {pred_dir.resolve()} does not exist")
        sys.exit(1)

    instance_id_to_idx = {}
    instances_to_run = []

    for idx, example in enumerate(dataset):
        instance_id = str(example["instance_id"])
        instance_id_to_idx[instance_id] = idx

        # Skip if already evaluated
        if idx in evaluated_indices:
            continue

        # Skip if not in filter
        if filter_ids and instance_id not in filter_ids:
            continue

        gold_program_name = example["gold_program_name"]
        pred_name = f"pred_{gold_program_name}"
        pred_file = pred_dir / pred_name

        if not pred_file.exists():
            print(f"Warning: No predicted program for instance {instance_id} "
                  f"(expected {pred_file}), skipping.")
            continue

        pred_content = pred_file.read_text()

        # Build test_spec dict (same fields as TestSpec dataclass)
        test_spec = {
            "instance_id": instance_id,
            "domain": example["domain"],
            "subtask_categories": example["subtask_categories"],
            "github_name": example["github_name"],
            "task_inst": example["task_inst"],
            "domain_knowledge": example["domain_knowledge"],
            "dataset_folder_tree": example["dataset_folder_tree"],
            "dataset_preview": example["dataset_preview"],
            "src_file_or_path": example["src_file_or_path"],
            "gold_program_name": gold_program_name,
            "output_fname": example["output_fname"],
            "benchmark_path": "/testbed/benchmark",
            "eval_script_name": example["eval_script_name"],
            "pred_program_path": "/testbed/pred_programs",
            "arch": "x86_64",
        }

        instances_to_run.append({
            "instance_id": instance_id,
            "test_spec": test_spec,
            "pred_program_content": pred_content,
            "pred_program_name": pred_name,
            "openai_api_key": resolved_openai_key,
        })

    if not instances_to_run:
        print("No instances to run.")
        return

    print(f"Running {len(instances_to_run)} instances on Modal...")
    ids_to_run = [d["instance_id"] for d in instances_to_run]
    print(f"Instance IDs: {ids_to_run}")

    # Launch all instances in parallel via Modal map, or in bounded batches
    # when max_concurrent is set to reduce external API pressure.
    results = []
    if max_concurrent and max_concurrent > 0:
        for start in range(0, len(instances_to_run), max_concurrent):
            batch = instances_to_run[start : start + max_concurrent]
            batch_ids = [d["instance_id"] for d in batch]
            print(
                f"Starting eval batch {start // max_concurrent + 1}: "
                f"{len(batch)} instances {batch_ids}"
            )
            batch_results = list(evaluate_instance.map(batch, return_exceptions=True))
            results.extend(batch_results)
    else:
        results = list(evaluate_instance.map(instances_to_run, return_exceptions=True))

    # Process results
    num_success = 0
    num_failed = 0
    for r in results:
        if isinstance(r, Exception):
            print(f"Exception: {r}")
            num_failed += 1
            continue

        iid = r["instance_id"]
        idx = instance_id_to_idx.get(iid)
        if idx is None:
            print(f"Warning: unknown instance_id {iid}")
            num_failed += 1
            continue

        result_tuple = r["result_tuple"]
        evaluated_logs[idx] = result_tuple

        # Determine pass/fail for display
        if isinstance(result_tuple, (list, tuple)) and len(result_tuple) >= 4:
            success = result_tuple[2]
            status = "PASS" if success == 1 else "FAIL"
        else:
            status = "FAIL"

        print(f"  [{iid}] {status}: {result_tuple}")

        if status == "PASS":
            num_success += 1
        else:
            num_failed += 1

    # Write JSONL log (same format as local harness)
    # Find a default codebert_score for missing entries
    default_cbs = 0.0
    default_log_info = "Not evaluated"
    for log in evaluated_logs:
        if isinstance(log, (list, tuple)) and len(log) >= 4:
            valid_program, cbs, success_rate, log_info = log
            if success_rate == 0 and valid_program == 0:
                default_cbs = cbs
                default_log_info = log_info
                break

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        for log in evaluated_logs:
            if log is None:
                entry = {
                    "valid_program": 0,
                    "codebert_score": default_cbs,
                    "success_rate": 0,
                    "log_info": default_log_info,
                }
            elif isinstance(log, dict):
                entry = log
            else:
                valid_program, cbs, success_rate, log_info = log
                entry = {
                    "valid_program": valid_program,
                    "codebert_score": cbs,
                    "success_rate": success_rate,
                    "log_info": log_info,
                }
            f.write(json.dumps(entry) + "\n")

    total = num_success + num_failed
    print(f"\nFinished. {num_success}/{total} passed, {num_failed}/{total} failed.")
    print(f"Results written to {log_path}")
