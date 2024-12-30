#!/usr/bin/env python3

import subprocess
import psutil
import time
import sys
from typing import List
import pynvml
import itertools
import json

def shutdown_nvml():
    """
    Shuts down NVML to free up resources.
    """
    try:
        pynvml.nvmlShutdown()
    except pynvml.NVMLError as e:
        print(f"Error shutting down NVML: {e}")

def get_available_gpus(threshold: int = 10, min_free_mem: int = 4000) -> List[int]:
    """
    Detects and returns a list of GPU indices that are available based on the utilization and memory thresholds.

    Args:
        threshold (int, optional): Maximum GPU utilization percentage to consider a GPU as available. Defaults to 10.
        min_free_mem (int, optional): Minimum free memory in MB required to launch a job. Defaults to 4000.

    Returns:
        List[int]: List of available GPU indices.
    """
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError as e:
        print(f"Failed to initialize NVML: {e}")
        return []

    device_count = pynvml.nvmlDeviceGetCount()
    available_gpus = []
    for i in range(device_count):
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_mem_mb = gpu_memory.free / (1024**2)
            print(f"GPU {i}: Utilization={utilization.gpu}%, Free Memory={free_mem_mb:.2f}MB/{gpu_memory.total / (1024**2):.2f}MB")
            if utilization.gpu < threshold and free_mem_mb > min_free_mem:
                available_gpus.append(i)
        except pynvml.NVMLError as e:
            print(f"Failed to get info for GPU {i}: {e}")
            continue

    return available_gpus

def multi_gpu_launcher(commands: List[str], gpus: List[int]):
    """
    Launches the given commands on the specified GPUs, assigning one command per GPU.

    Args:
        commands (List[str]): List of shell commands to execute.
        gpus (List[int]): List of GPU indices to assign commands to.
    """
    print('\nStarting multi_gpu_launcher...')
    n_gpus = len(gpus)
    if n_gpus == 0:
        print("No free GPUs available based on the specified threshold. Exiting.")
        sys.exit(1)

    procs_by_gpu = {gpu: None for gpu in gpus}
    job_queue = list(commands)  # Commands to execute

    try:
        while job_queue or any(proc is not None for proc in procs_by_gpu.values()):
            for gpu in gpus:
                proc = procs_by_gpu[gpu]
                if proc is None or proc.poll() is not None:
                    if job_queue:
                        cmd = job_queue.pop(0)
                        # Prefix the command with CUDA_VISIBLE_DEVICES to assign it to the specific GPU
                        full_cmd = f"CUDA_VISIBLE_DEVICES={gpu} {cmd}"
                        print(f"\nLaunching on GPU {gpu}: {cmd}")
                        try:
                            # Start the subprocess
                            procs_by_gpu[gpu] = subprocess.Popen(full_cmd, shell=True)
                            print(f"Process started on GPU {gpu} with PID {procs_by_gpu[gpu].pid}")
                            # print('\n')
                        except Exception as e:
                            print(f"GPU {gpu}: Failed to start command. Error: {e}")
            # Sleep briefly to prevent excessive CPU usage
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nLauncher interrupted by user. Terminating running jobs...")
        for gpu, proc in procs_by_gpu.items():
            if proc is not None:
                print(f"Terminating process on GPU {gpu} with PID {proc.pid}")
                proc.terminate()
        shutdown_nvml()
        sys.exit(0)

    # Wait for all processes to complete
    for gpu, proc in procs_by_gpu.items():
        if proc is not None:
            proc.wait()
            print(f"\nJob on GPU {gpu} with PID {proc.pid} has finished.")

def load_config(config_path: str):
    """
    Loads the configuration from a JSON file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def generate_commands_from_config(config: dict):
    """
    Generates a list of commands based on the provided configuration.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        List[str]: List of shell commands.
    """
    commands = []
    for script, details in config.items():
        scripts_cmd = details["scripts"]
        algorithms = details["algorithms"]
        for algo, algo_params in algorithms.items():
            lrs = algo_params.get("lr", [0.1])
            batch_sizes = algo_params.get("batch_size", [128])
            flags_list = algo_params.get("flags", [[]])
            for lr, bs, flags in itertools.product(lrs, batch_sizes, flags_list):
                flags_str = ' '.join(flags)
                cmd = f"{scripts_cmd} --algorithm {algo} --lr {lr} --batch_size {bs}  {flags_str}"
                commands.append(cmd.strip())
    return commands

def generate_commands():
    """
    Loads configuration and generates a list of commands.

    Returns:
        List[str]: List of shell commands.
    """
    config_path = "exp_config.json"  # Path to your configuration file
    config = load_config(config_path)
    commands = generate_commands_from_config(config)
    return commands

def main():
    # Generate the list of commands dynamically
    commands = generate_commands()

    # Detect available GPUs with memory constraints
    available_gpus = get_available_gpus(threshold=10, min_free_mem=4000)  # Adjust thresholds as needed
    if not available_gpus:
        print("No available GPUs detected based on the utilization and memory thresholds. Exiting.")
        sys.exit(1)

    print(f"\nAvailable GPUs for launching jobs: {available_gpus}")

    # Launch the commands on available GPUs
    multi_gpu_launcher(commands, available_gpus)

    # Shutdown NVML after all jobs are done
    shutdown_nvml()

if __name__ == '__main__':
    main()
