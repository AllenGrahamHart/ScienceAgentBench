#!/usr/bin/env python3
"""
Run Claude Code inference on ScienceAgentBench tasks.

This script is the "Track A" component of parity experiments: it runs Claude
Code inside Docker containers that match the Harbor adapter environment, then
outputs predicted programs in the format expected by SAB's evaluation harness.

Usage:
    # Build the image first
    docker build -f Dockerfile.claude_code -t sab-claude-code:latest .

    # Run on specific tasks
    python run_infer_claude_code.py \
        --instance_ids 5 16 21 \
        --out_fpath pred_programs_cc_test \
        --log_fname cc_test_infer.jsonl

    # Run all 102 tasks
    python run_infer_claude_code.py \
        --out_fpath pred_programs_cc_run1 \
        --log_fname cc_run1_infer.jsonl

    # With proxy (for 2077AI-sponsored runs)
    python run_infer_claude_code.py \
        --anthropic_base_url "http://pp-api-ec82a10d0c5d226c.elb.us-west-2.amazonaws.com:3000" \
        --out_fpath pred_programs_cc_run1 \
        --log_fname cc_run1_infer.jsonl
"""

from __future__ import annotations

import argparse
import docker
import json
import os
import shutil
import signal
import sys
import tarfile
import tempfile
import threading
import time
import traceback

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from datasets import load_dataset


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
        f"The input dataset is located at `benchmark/datasets/{dataset_folder_name}/` "
        f"(relative to the working directory `/testbed/`).\n\n"
        f"**Directory structure:**\n```\n{dataset_folder_tree}\n```\n"
    )

    # Truncate dataset_preview if very long (>2000 chars)
    if dataset_preview:
        preview = dataset_preview[:2000]
        if len(dataset_preview) > 2000:
            preview += "\n... (truncated)"
        parts.append(f"**Data preview:**\n```\n{preview}\n```\n")

    parts.append(
        f"## Output Requirements\n\n"
        f"- Write your solution as a Python program named `{gold_program_name}`\n"
        f"- Save it to `/testbed/{gold_program_name}`\n"
        f"- The program must produce the output file at `{output_fname}` "
        f"(relative to `/testbed/`)\n"
        f"- Make sure to create the `pred_results/` directory before writing output\n"
        f"- The program must be self-contained and runnable with "
        f"`cd /testbed && python {gold_program_name}`\n"
        f"- Install any required dependencies before running\n"
    )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Helper: create a tar archive in memory for docker cp
# ---------------------------------------------------------------------------

def make_tar(src_path: Path, arcname: str) -> BytesIO:
    """Create an in-memory tar archive of src_path."""
    buf = BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        tar.add(str(src_path), arcname=arcname)
    buf.seek(0)
    return buf


def make_tar_from_string(content: str, arcname: str) -> BytesIO:
    """Create an in-memory tar archive containing a single text file."""
    buf = BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        data = content.encode("utf-8")
        info = tarfile.TarInfo(name=arcname)
        info.size = len(data)
        tar.addfile(info, BytesIO(data))
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------
# Per-instance runner
# ---------------------------------------------------------------------------

def run_instance(
    example: dict,
    args: argparse.Namespace,
    client: docker.DockerClient,
) -> dict:
    """Run Claude Code on a single SAB instance inside a Docker container.

    Returns a log dict with instance_id, status, predicted program path, etc.
    """
    instance_id = str(example["instance_id"])
    gold_program_name = example["gold_program_name"]
    dataset_folder_tree = example["dataset_folder_tree"]
    # Extract top-level folder name from tree (e.g. "|-- dkpes/" → "dkpes")
    dataset_folder_name = dataset_folder_tree.split("\n")[0].split("-- ")[-1].rstrip("/")

    log_entry: dict = {
        "instance_id": instance_id,
        "model_name": args.model_name,
        "start_time": datetime.now(timezone.utc).isoformat(),
        "status": "error",
        "predicted_program": None,
        "error": None,
    }

    container = None
    try:
        # 1. Build instruction (identical to Harbor adapter)
        instruction = build_instruction(
            task_inst=example["task_inst"],
            domain_knowledge=example.get("domain_knowledge", ""),
            dataset_folder_tree=dataset_folder_tree,
            dataset_preview=example.get("dataset_preview", ""),
            gold_program_name=gold_program_name,
            output_fname=example["output_fname"],
            dataset_folder_name=dataset_folder_name,
        )

        # 2. Build env vars matching Harbor's claude_code.py:769-853
        env = {
            "IS_SANDBOX": "1",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "FORCE_AUTO_BACKGROUND_TASKS": "1",
            "ENABLE_BACKGROUND_TASKS": "1",
        }

        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if api_key:
            env["ANTHROPIC_API_KEY"] = api_key

        base_url = args.anthropic_base_url or os.environ.get("ANTHROPIC_BASE_URL", "")
        if base_url:
            env["ANTHROPIC_BASE_URL"] = base_url
            # With custom base URL, keep full model name and set all aliases
            env["ANTHROPIC_MODEL"] = args.model_name
            env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = args.model_name
            env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = args.model_name
            env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = args.model_name
            env["CLAUDE_CODE_SUBAGENT_MODEL"] = args.model_name
        else:
            # Direct Anthropic API — strip provider prefix if present
            model = args.model_name.split("/")[-1]
            env["ANTHROPIC_MODEL"] = model

        # 3. Start container
        container = client.containers.run(
            "sab-claude-code:latest",
            command="sleep infinity",
            detach=True,
            working_dir="/testbed",
            environment=env,
            mem_limit=args.mem_limit,
            name=f"sab_cc_{instance_id}_{int(time.time())}",
        )
        print(f"  [{instance_id}] Container started: {container.short_id}")

        # 4. Copy dataset into container
        dataset_src = Path(args.datasets_path) / dataset_folder_name
        if dataset_src.is_dir():
            tar = make_tar(dataset_src, arcname=dataset_folder_name)
            container.put_archive("/testbed/benchmark/datasets/", tar)
            # Ensure __init__.py files exist for importability
            container.exec_run(
                'bash -c "touch /testbed/benchmark/datasets/__init__.py && '
                "find /testbed/benchmark/datasets -type f -name '*.py' "
                "! -name '__init__.py' "
                "-exec sh -c 'dir=$(dirname \"$1\"); "
                "while [ \"$dir\" != \"/testbed/benchmark/datasets\" ] && "
                '[ "$dir" != "/testbed/benchmark" ]; do '
                "touch \"$dir/__init__.py\"; dir=$(dirname \"$dir\"); "
                "done' _ {} \\;\""
            )
        else:
            print(f"  [{instance_id}] WARNING: dataset folder not found at {dataset_src}")

        # 5. Write instruction to a file (avoids shell escaping issues)
        tar = make_tar_from_string(instruction, "instruction.md")
        container.put_archive("/testbed/", tar)

        # 6. Set up Claude config dirs (mirrors Harbor's setup command)
        setup_cmd = (
            "mkdir -p /root/.claude/debug /root/.claude/projects/-app "
            "/root/.claude/shell-snapshots /root/.claude/statsig "
            "/root/.claude/todos"
        )
        container.exec_run(f"bash -c '{setup_cmd}'")

        # 7. Run Claude Code (with timeout enforcement)
        claude_cmd = (
            'bash -lc "'
            "claude --verbose --output-format stream-json "
            "--permission-mode bypassPermissions "
            '-p \\"$(cat /testbed/instruction.md)\\" '
            '2>&1 | tee /testbed/logs/claude-code.txt"'
        )
        print(f"  [{instance_id}] Running Claude Code (timeout={args.timeout}s)...")

        # Run exec in a thread so we can enforce a timeout
        exec_result = [None, None]  # [exit_code, output]
        timed_out = False

        def _run_exec():
            exec_result[0], exec_result[1] = container.exec_run(
                claude_cmd,
                environment=env,
                workdir="/testbed",
                demux=True,
            )

        exec_thread = threading.Thread(target=_run_exec, daemon=True)
        exec_thread.start()
        exec_thread.join(timeout=args.timeout)

        if exec_thread.is_alive():
            timed_out = True
            print(f"  [{instance_id}] TIMEOUT after {args.timeout}s — stopping container")
            try:
                container.stop(timeout=10)
            except Exception:
                pass

        exit_code = exec_result[0]
        output = exec_result[1]

        if timed_out:
            log_entry["error"] = f"Timed out after {args.timeout}s"
            stdout = ""
            stderr = ""
        else:
            stdout = (output[0] or b"").decode("utf-8", errors="replace") if output else ""
            stderr = (output[1] or b"").decode("utf-8", errors="replace") if output else ""

        log_entry["exit_code"] = exit_code
        log_entry["timed_out"] = timed_out
        log_entry["stdout_tail"] = stdout[-2000:] if len(stdout) > 2000 else stdout
        log_entry["stderr_tail"] = stderr[-1000:] if len(stderr) > 1000 else stderr

        # 8. Copy predicted program out of container
        pred_program_name = f"pred_{gold_program_name}"
        out_dir = Path(args.out_fpath)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / pred_program_name

        try:
            bits, _ = container.get_archive(f"/testbed/{gold_program_name}")
            with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tmp:
                for chunk in bits:
                    tmp.write(chunk)
                tmp_path = tmp.name

            with tarfile.open(tmp_path) as tar:
                member = tar.getmembers()[0]
                f = tar.extractfile(member)
                if f:
                    out_path.write_bytes(f.read())
                    log_entry["predicted_program"] = str(out_path)
                    log_entry["status"] = "success"
                    print(f"  [{instance_id}] Predicted program saved to {out_path}")
                else:
                    log_entry["error"] = f"Could not extract {gold_program_name} from tar"
            os.unlink(tmp_path)
        except docker.errors.NotFound:
            log_entry["error"] = (
                f"Predicted program /testbed/{gold_program_name} not found in container"
            )
            print(f"  [{instance_id}] WARNING: {log_entry['error']}")
        except Exception as e:
            log_entry["error"] = f"Error extracting program: {e}"

        # Also save the full Claude Code log
        logs_dir = Path(args.out_fpath) / "logs" / instance_id
        logs_dir.mkdir(parents=True, exist_ok=True)
        try:
            bits, _ = container.get_archive("/testbed/logs/claude-code.txt")
            with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tmp:
                for chunk in bits:
                    tmp.write(chunk)
                tmp_path = tmp.name
            with tarfile.open(tmp_path) as tar:
                member = tar.getmembers()[0]
                f = tar.extractfile(member)
                if f:
                    (logs_dir / "claude-code.txt").write_bytes(f.read())
            os.unlink(tmp_path)
        except Exception:
            pass  # Log saving is best-effort

    except Exception as e:
        log_entry["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        print(f"  [{instance_id}] ERROR: {e}")
    finally:
        log_entry["end_time"] = datetime.now(timezone.utc).isoformat()
        if container:
            try:
                container.stop(timeout=5)
                container.remove(force=True)
            except Exception:
                pass

    return log_entry


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run Claude Code inference on ScienceAgentBench tasks"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="claude-haiku-4-5-20251001",
        help="Model name for Claude Code (default: claude-haiku-4-5-20251001)",
    )
    parser.add_argument(
        "--datasets_path",
        type=str,
        default="benchmark/datasets/",
        help="Path to benchmark datasets directory",
    )
    parser.add_argument(
        "--out_fpath",
        type=str,
        default="pred_programs_cc/",
        help="Output directory for predicted programs",
    )
    parser.add_argument(
        "--log_fname",
        type=str,
        default="cc_infer.jsonl",
        help="JSONL log file for inference results",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="osunlp/ScienceAgentBench",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Dataset split",
    )
    parser.add_argument(
        "--instance_ids",
        nargs="+",
        type=int,
        default=None,
        help="Specific instance IDs to run (space-separated)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Timeout in seconds per instance (default: 3600)",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )
    parser.add_argument(
        "--anthropic_base_url",
        type=str,
        default="",
        help="Custom Anthropic base URL (e.g. for 2077AI proxy)",
    )
    parser.add_argument(
        "--mem_limit",
        type=str,
        default="8g",
        help="Docker memory limit per container (default: 8g)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run even if log entry exists for an instance",
    )
    args = parser.parse_args()

    # Validate
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("WARNING: ANTHROPIC_API_KEY not set. Claude Code will fail to authenticate.")

    # Load dataset
    print(f"Loading dataset: {args.dataset_name} (split={args.split})")
    dataset = load_dataset(args.dataset_name, split=args.split)
    print(f"Loaded {len(dataset)} instances")

    # Filter by instance_ids if specified
    if args.instance_ids:
        instance_id_set = set(str(i) for i in args.instance_ids)
        examples = [ex for ex in dataset if str(ex["instance_id"]) in instance_id_set]
        print(f"Filtered to {len(examples)} instances: {sorted(args.instance_ids)}")
    else:
        examples = list(dataset)

    # Resume: load already-completed instance IDs from log
    completed_ids: set[str] = set()
    log_path = Path(args.log_fname)
    if log_path.exists() and not args.force:
        with open(log_path, "r") as f:
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

    print(
        f"Running {len(examples_to_run)} instances "
        f"(workers={args.max_workers}, timeout={args.timeout}s)"
    )

    # Docker client
    client = docker.from_env()

    # Verify image exists
    try:
        client.images.get("sab-claude-code:latest")
    except docker.errors.ImageNotFound:
        print(
            "ERROR: Docker image 'sab-claude-code:latest' not found.\n"
            "Build it first: docker build -f Dockerfile.claude_code -t sab-claude-code:latest ."
        )
        sys.exit(1)

    # Output directory
    Path(args.out_fpath).mkdir(parents=True, exist_ok=True)

    if args.max_workers <= 1:
        # Sequential execution
        for i, example in enumerate(examples_to_run):
            iid = example["instance_id"]
            print(f"\n[{i+1}/{len(examples_to_run)}] Instance {iid}")
            log_entry = run_instance(example, args, client)
            with open(args.log_fname, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
    else:
        # Parallel execution
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(run_instance, ex, args, client): ex
                for ex in examples_to_run
            }
            for i, future in enumerate(as_completed(futures)):
                log_entry = future.result()
                iid = log_entry["instance_id"]
                status = log_entry["status"]
                print(f"  [{i+1}/{len(examples_to_run)}] Instance {iid}: {status}")
                with open(args.log_fname, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")

    # Summary
    completed = 0
    failed = 0
    if log_path.exists():
        with open(log_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    if entry.get("status") == "success":
                        completed += 1
                    else:
                        failed += 1
    print(f"\nDone. {completed} succeeded, {failed} failed. Log: {args.log_fname}")


if __name__ == "__main__":
    main()
