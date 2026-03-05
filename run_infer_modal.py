"""
Modal-based parallel Claude Code inference for ScienceAgentBench.

Replicates the local Docker inference (run_infer_claude_code.py) but runs
all 102 instances in parallel on Modal Sandboxes, avoiding local OOM and
enabling higher concurrency.

Usage:
    # Run all 102 tasks (uses ANTHROPIC_API_KEY and ANTHROPIC_BASE_URL from env)
    modal run run_infer_modal.py::main \
        --out-fpath pred_programs_cc_run2 \
        --log-fname cc_run2_infer.jsonl

    # Run subset
    modal run run_infer_modal.py::main \
        --out-fpath pred_programs_cc_test \
        --log-fname cc_test_infer.jsonl \
        --instance-ids "3,8,20"

    # With explicit proxy URL
    modal run run_infer_modal.py::main \
        --out-fpath pred_programs_cc_run2 \
        --log-fname cc_run2_infer.jsonl \
        --anthropic-base-url "https://pp-api-ec82a10d0c5d226c.elb.us-west-2.amazonaws.com"
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import modal

app = modal.App("sab-infer")

# Build the Claude Code image on Modal from the same Dockerfiles used locally.
# Modal caches layers, so after the first build (~15-20 min) it's instant.
sab_cc_image = (
    modal.Image.from_dockerfile("Dockerfile.sab_base")
    .dockerfile_commands(
        [
            # Node.js (required by Claude Code CLI)
            "RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash -"
            " && apt-get install -y nodejs"
            " && rm -rf /var/lib/apt/lists/*",
            # Claude Code CLI
            "RUN npm install -g @anthropic-ai/claude-code",
            # Pre-create directory tree (benchmark/ left empty for volume mount)
            "RUN mkdir -p /testbed/pred_results /testbed/logs",
        ]
    )
)

# Reuse the same benchmark volume as run_eval_modal.py
# Mount at /benchmark_vol to avoid conflict with Dockerfile-created paths,
# then symlink into /testbed/benchmark at sandbox startup.
benchmark_vol = modal.Volume.from_name("sab-benchmark", create_if_missing=True)
VOLUME_MOUNT = "/benchmark_vol"


# ---------------------------------------------------------------------------
# Instruction builder — mirrors Harbor adapter's _build_instruction exactly
# (adapters/scienceagentbench/adapter.py:244-291)
# ---------------------------------------------------------------------------

def build_instruction(
    task_inst: str,
    domain_knowledge: str,
    dataset_folder_tree: str,
    dataset_preview: str,
    gold_program_name: str,
    output_fname: str,
    dataset_folder_name: str,
) -> str:
    """Build the reframed instruction for terminal agents."""
    parts = []
    parts.append(
        "You are tasked with a scientific computing problem. "
        "Write a self-contained Python program to solve it.\n"
    )
    parts.append(f"## Task\n\n{task_inst}\n")

    if domain_knowledge:
        parts.append(f"## Domain Knowledge\n\n{domain_knowledge}\n")

    parts.append(
        f"## Input Data\n\n"
        f"The input datasets are located in `/testbed/benchmark/datasets/{dataset_folder_name}/`.\n\n"
        f"Directory structure:\n```\n{dataset_folder_tree}\n```\n"
    )

    if dataset_preview:
        parts.append(f"### Data Preview\n\n{dataset_preview}\n")

    parts.append(
        f"## Output Requirements\n\n"
        f"- Save your program as `/testbed/{gold_program_name}`\n"
        f"- The program must write its output to `/testbed/pred_results/{output_fname}`\n"
        f"- The program must be self-contained and runnable with `python /testbed/{gold_program_name}`\n"
        f"- Use the conda environment `sab` (already activated) which has common scientific packages pre-installed\n"
    )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Modal Sandbox-based inference for a single instance
# ---------------------------------------------------------------------------

def run_instance_on_modal(
    instance_data: dict,
    app: modal.App,
) -> dict:
    """Run Claude Code inference for one SAB instance in a Modal Sandbox.

    Args:
        instance_data: dict with keys from the HuggingFace dataset row plus
            anthropic_api_key, anthropic_base_url, model_name, timeout.

    Returns:
        dict with instance_id, status, predicted_program (source code), error, etc.
    """
    instance_id = instance_data["instance_id"]
    model_name = instance_data["model_name"]
    timeout = instance_data["timeout"]

    log_entry = {
        "instance_id": instance_id,
        "model_name": model_name,
        "start_time": datetime.now(timezone.utc).isoformat(),
        "status": "error",
        "predicted_program": None,
        "error": None,
    }

    # Build env vars for Claude Code (mirrors run_infer_claude_code.py:174-198)
    env = {
        "IS_SANDBOX": "1",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "FORCE_AUTO_BACKGROUND_TASKS": "1",
        "ENABLE_BACKGROUND_TASKS": "1",
    }

    api_key = instance_data.get("anthropic_api_key", "")
    if api_key:
        env["ANTHROPIC_API_KEY"] = api_key

    base_url = instance_data.get("anthropic_base_url", "")
    if base_url:
        env["ANTHROPIC_BASE_URL"] = base_url
        env["ANTHROPIC_MODEL"] = model_name
        env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = model_name
        env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = model_name
        env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = model_name
        env["CLAUDE_CODE_SUBAGENT_MODEL"] = model_name
    else:
        model = model_name.split("/")[-1]
        env["ANTHROPIC_MODEL"] = model

    # Build instruction
    dataset_folder_tree = instance_data["dataset_folder_tree"]
    dataset_folder_name = dataset_folder_tree.split("\n")[0].split("-- ")[-1].rstrip("/")
    gold_program_name = instance_data["gold_program_name"]

    instruction = build_instruction(
        task_inst=instance_data["task_inst"],
        domain_knowledge=instance_data.get("domain_knowledge", ""),
        dataset_folder_tree=dataset_folder_tree,
        dataset_preview=instance_data.get("dataset_preview", ""),
        gold_program_name=gold_program_name,
        output_fname=instance_data["output_fname"],
        dataset_folder_name=dataset_folder_name,
    )

    sandbox = None
    try:
        # Create Modal Sandbox with env vars set at sandbox level
        sandbox = modal.Sandbox.create(
            image=sab_cc_image,
            app=app,
            timeout=timeout + 300,  # buffer beyond agent timeout
            cpu=2,
            memory=8192,
            secrets=[modal.Secret.from_dict(env)],
            volumes={VOLUME_MOUNT: benchmark_vol},
        )

        print(f"  [{instance_id}] Sandbox created: {sandbox.object_id}")

        # Symlink volume into expected path and set up dirs
        setup_cmd = (
            # Symlink benchmark volume
            "rm -rf /testbed/benchmark && "
            f"ln -s {VOLUME_MOUNT} /testbed/benchmark && "
            # Ensure __init__.py stubs for imports
            "touch /testbed/benchmark/__init__.py && "
            "find /testbed/benchmark/datasets -type d "
            "-exec sh -c 'touch \"$1/__init__.py\"' _ {} \\; && "
            # Claude config dirs
            "mkdir -p /root/.claude/debug /root/.claude/projects/-app "
            "/root/.claude/shell-snapshots /root/.claude/statsig "
            "/root/.claude/todos"
        )
        sandbox.exec("bash", "-c", setup_cmd).wait()

        # Write instruction file
        p = sandbox.open("/testbed/instruction.md", "w")
        p.write(instruction)
        p.close()

        # Verify instruction file was written before proceeding
        check = sandbox.exec("bash", "-c",
            "wc -c < /testbed/instruction.md")
        size = "".join(check.stdout).strip()
        check.wait()
        print(f"  [{instance_id}] Instruction file: {size} bytes")

        # Run Claude Code (wrapped in `script` to provide a pseudo-TTY,
        # which Claude Code requires to produce output in Modal sandboxes).
        # Use -p with $(cat ...) for agentic mode (piped stdin = chat mode only).
        claude_cmd = (
            "script -qc '"
            'claude --verbose --output-format stream-json '
            '--permission-mode bypassPermissions '
            '-p \"$(cat /testbed/instruction.md)\" '
            "2>&1 | tee /testbed/logs/claude-code.txt' /dev/null"
        )

        print(f"  [{instance_id}] Running Claude Code (timeout={timeout}s)...")
        process = sandbox.exec(
            "bash", "-lc", claude_cmd,
            workdir="/testbed",
            timeout=timeout,
        )

        # Collect output
        stdout_lines = []
        try:
            for line in process.stdout:
                stdout_lines.append(line)
        except Exception:
            pass

        stderr_lines = []
        try:
            for line in process.stderr:
                stderr_lines.append(line)
        except Exception:
            pass

        try:
            return_code = process.wait()
        except modal.exception.SandboxTimeoutError:
            log_entry["error"] = f"Timed out after {timeout}s"
            log_entry["timed_out"] = True
            log_entry["end_time"] = datetime.now(timezone.utc).isoformat()
            return log_entry

        stdout = "".join(stdout_lines)
        stderr = "".join(stderr_lines)

        log_entry["exit_code"] = return_code
        log_entry["timed_out"] = False
        log_entry["stdout_tail"] = stdout[-2000:]
        log_entry["stderr_tail"] = stderr[-1000:]

        # Extract predicted program
        try:
            p = sandbox.open(f"/testbed/{gold_program_name}", "r")
            pred_content = p.read()
            p.close()
            log_entry["predicted_program"] = pred_content
            log_entry["status"] = "success"
            print(f"  [{instance_id}] Success — extracted {gold_program_name}")
        except Exception as e:
            log_entry["error"] = (
                f"Predicted program /testbed/{gold_program_name} not found. "
                f"Claude Code may not have created it. Error: {e}"
            )
            print(f"  [{instance_id}] FAIL — {gold_program_name} not found")

    except Exception as e:
        log_entry["error"] = str(e)
        print(f"  [{instance_id}] ERROR: {e}")

    finally:
        if sandbox:
            try:
                sandbox.terminate()
            except Exception:
                pass

    log_entry["end_time"] = datetime.now(timezone.utc).isoformat()
    return log_entry


@app.local_entrypoint()
def main(
    model_name: str = "claude-haiku-4-5",
    out_fpath: str = "pred_programs_cc_run2",
    log_fname: str = "cc_run2_infer.jsonl",
    dataset_name: str = "osunlp/ScienceAgentBench",
    split: str = "validation",
    instance_ids: str = "",
    timeout: int = 3600,
    max_concurrent: int = 16,
    anthropic_base_url: str = "",
    force: bool = False,
):
    """Run Claude Code inference on ScienceAgentBench via Modal Sandboxes.

    Args:
        model_name: Model name for Claude Code.
        out_fpath: Output directory for predicted programs.
        log_fname: JSONL log file for inference results.
        dataset_name: HuggingFace dataset name.
        split: Dataset split.
        instance_ids: Comma-separated instance IDs (empty = all).
        timeout: Timeout in seconds per instance.
        max_concurrent: Maximum concurrent Modal Sandboxes.
        anthropic_base_url: Custom Anthropic API base URL.
        force: Re-run even if log entry exists.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from datasets import load_dataset

    # Resolve API credentials
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    base_url = anthropic_base_url or os.environ.get("ANTHROPIC_BASE_URL", "")

    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    print(f"API key: {api_key[:12]}...{api_key[-4:]}")
    print(f"Base URL: {base_url or '(direct Anthropic API)'}")
    print(f"Model: {model_name}")

    # Load dataset
    print(f"Loading dataset: {dataset_name} (split={split})")
    dataset = load_dataset(dataset_name, split=split)
    print(f"Loaded {len(dataset)} instances")

    # Filter by instance_ids
    if instance_ids:
        id_set = {s.strip() for s in instance_ids.split(",") if s.strip()}
        examples = [ex for ex in dataset if str(ex["instance_id"]) in id_set]
        print(f"Filtered to {len(examples)} instances")
    else:
        examples = list(dataset)

    # Resume: skip already-completed instances
    completed_ids: set[str] = set()
    log_path = Path(log_fname)
    if log_path.exists() and not force:
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    if entry.get("status") == "success":
                        completed_ids.add(str(entry["instance_id"]))
        if completed_ids:
            print(f"Resuming: {len(completed_ids)} instances already completed")

    examples_to_run = [
        ex for ex in examples if str(ex["instance_id"]) not in completed_ids
    ]

    if not examples_to_run:
        print("All instances already completed. Use --force to re-run.")
        return

    # Output directory
    out_dir = Path(out_fpath)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build instance data payloads
    payloads = []
    for ex in examples_to_run:
        payloads.append({
            "instance_id": str(ex["instance_id"]),
            "task_inst": ex["task_inst"],
            "domain_knowledge": ex.get("domain_knowledge", ""),
            "dataset_folder_tree": ex["dataset_folder_tree"],
            "dataset_preview": ex.get("dataset_preview", ""),
            "gold_program_name": ex["gold_program_name"],
            "output_fname": ex["output_fname"],
            "anthropic_api_key": api_key,
            "anthropic_base_url": base_url,
            "model_name": model_name,
            "timeout": timeout,
        })

    print(
        f"Running {len(payloads)} instances on Modal "
        f"(max_concurrent={max_concurrent}, timeout={timeout}s)"
    )

    # Run instances concurrently using ThreadPoolExecutor
    # Each thread creates a Modal Sandbox (the Sandbox itself runs remotely)
    num_success = 0
    num_failed = 0

    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = {
            executor.submit(run_instance_on_modal, payload, app): payload
            for payload in payloads
        }

        for i, future in enumerate(as_completed(futures)):
            log_entry = future.result()
            iid = log_entry["instance_id"]
            status = log_entry["status"]

            print(f"  [{i + 1}/{len(payloads)}] Instance {iid}: {status}")

            # Write log entry (append)
            with open(log_fname, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            # Save predicted program to disk
            if log_entry.get("predicted_program"):
                gold_name = None
                for ex in examples_to_run:
                    if str(ex["instance_id"]) == iid:
                        gold_name = ex["gold_program_name"]
                        break
                if gold_name:
                    pred_name = f"pred_{gold_name}"
                    (out_dir / pred_name).write_text(log_entry["predicted_program"])

            if status == "success":
                num_success += 1
            else:
                num_failed += 1

    total = num_success + num_failed
    print(f"\nFinished. {num_success}/{total} succeeded, {num_failed}/{total} failed.")
    print(f"Predicted programs saved to {out_dir}/")
    print(f"Log written to {log_fname}")
