import os
import subprocess
import sys
import time

import yaml

DEVICES_MODE = "cuda:0"
MAX_CONCURRENT_RUNS_PER_GPU = 1
CHECK_INTERVAL = 5


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_target_devices(mode):
    if mode == "both":
        return ["cuda:0", "cuda:1"]
    if mode in {"cuda:0", "cuda:1"}:
        return [mode]
    return [mode]


def main():
    config_path = "config.yaml"
    config = load_config(config_path)

    models = config.get("MODELS_TO_BENCHMARK", ["google/vit-base-patch16-224"])
    datasets = config.get("DATASETS", [])
    experiments = config.get("EXPERIMENTS", [])
    seeds = config.get("SEEDS", [42])
    target_devices = get_target_devices(DEVICES_MODE)

    task_queue = []
    for model in models:
        for dataset in datasets:
            for exp_index, experiment in enumerate(experiments):
                for seed in seeds:
                    exp_name = experiment.get("name", f"Exp_{exp_index}")
                    model_short = model.split("/")[-1]
                    task_queue.append({
                        "description": f"[{model_short}] {dataset} :: {exp_name} (seed {seed})",
                        "base_cmd": [
                            sys.executable,
                            "main.py",
                            "--config",
                            config_path,
                            "--dataset",
                            dataset,
                            "--exp_index",
                            str(exp_index),
                            "--model",
                            model,
                            "--seed",
                            str(seed),
                        ],
                    })

    total_tasks = len(task_queue)
    print(f"Prepared {total_tasks} tasks.")

    running_procs = []
    finished_count = 0

    try:
        while task_queue or running_procs:
            active_procs = []
            for proc, task_info, device in running_procs:
                if proc.poll() is None:
                    active_procs.append((proc, task_info, device))
                else:
                    finished_count += 1
                    icon = "OK" if proc.returncode == 0 else "FAIL"
                    print(f"\n[{icon}] [{device}] Done: {task_info['description']}")
            running_procs = active_procs

            for device in target_devices:
                current_on_device = sum(1 for _, _, dev in running_procs if dev == device)
                while current_on_device < MAX_CONCURRENT_RUNS_PER_GPU and task_queue:
                    next_task = task_queue.pop(0)
                    final_cmd = next_task["base_cmd"] + ["--device", device]
                    print(f"Starting on [{device}]: {next_task['description']}")
                    proc = subprocess.Popen(final_cmd, cwd=os.path.dirname(os.path.abspath(__file__)) or ".")
                    running_procs.append((proc, next_task, device))
                    current_on_device += 1
                    time.sleep(2)

            sys.stdout.write(
                f"\rRunning: {len(running_procs)} | Queue: {len(task_queue)} | Done: {finished_count}/{total_tasks}   "
            )
            sys.stdout.flush()
            time.sleep(CHECK_INTERVAL)
    except KeyboardInterrupt:
        for proc, _, _ in running_procs:
            proc.terminate()
        sys.exit(0)


if __name__ == "__main__":
    main()