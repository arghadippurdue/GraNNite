import os
import csv
import re
import subprocess
import time
import shutil


# User inputs
performance_counters_list = ['ON', 'OFF']
nireq = -1  # -1 --> tool determined, else specify exact value

# Folder paths
blob_folder = "ov_model"    # Path to the xml & bin model files (OpenVINO IR)
report_folder_base = "logs_ir_2024_10_25"     # Log folder name

# Create report folder if it doesn't exist
os.makedirs(report_folder_base, exist_ok=True)

profiling_file = "profiling.json"  # The profiling.json generated in the base folder
base_folder = os.getcwd()  # The current directory from which the script is run

# Regex patterns for extracting latency and throughput
latency_pattern = re.compile(r"\[ INFO \]\s+Average:\s+([\d\.]+)\s+ms")
throughput_pattern = re.compile(r"\[ INFO \]\s+Throughput:\s+([\d\.]+)\s+FPS")

def get_precision_args(file_name, device):
    if file_name.endswith("fp16.xml"):
        return "-ip f16 -op f16"
    elif file_name.endswith("int8.xml"):
        if device == "NPU":
            return "-ip i8 -op i8 -infer_precision i8"
        return "-ip i8 -op i8"
    return ""

def get_nireq_arg(nireq):
    return f"-nireq {nireq}" if nireq == 1 else ""

def move_and_rename_profiling(blob_name, hint, report_folder):
    profiling_src = os.path.join(base_folder, profiling_file)
    if os.path.exists(profiling_src):
        profiling_dest = os.path.join(report_folder, f"{blob_name}_hint_{hint}.json")
        shutil.move(profiling_src, profiling_dest)
        print(f"Moved and renamed {profiling_file} to {profiling_dest}")
    else:
        print(f"{profiling_file} not found in the base folder.")

def rename_report_files(report_folder, blob_name, hint):
    benchmark_report = os.path.join(report_folder, "benchmark_report.csv")
    detailed_report = os.path.join(report_folder, "benchmark_detailed_counters_report.csv")

    new_benchmark_report = os.path.join(report_folder, f"{blob_name}_hint_{hint}_report.csv")
    new_detailed_report = os.path.join(report_folder, f"{blob_name}_hint_{hint}_detailed_report.csv")

    if os.path.exists(benchmark_report):
        os.rename(benchmark_report, new_benchmark_report)
        print(f"Renamed {benchmark_report} to {new_benchmark_report}")
    else:
        print(f"{benchmark_report} not found.")

    if os.path.exists(detailed_report):
        os.rename(detailed_report, new_detailed_report)
        print(f"Renamed {detailed_report} to {new_detailed_report}")
    else:
        print(f"{detailed_report} not found.")

def extract_log_data(log_folder, output_csv):
    with open(output_csv, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['filename', 'latency', 'throughput'])

        for file_name in os.listdir(log_folder):
            if file_name.endswith(".txt"):
                file_path = os.path.join(log_folder, file_name)
                latency = throughput = None

                with open(file_path, 'r') as log_file:
                    for line in log_file:
                        if not latency:
                            latency_match = latency_pattern.search(line)
                            if latency_match:
                                latency = latency_match.group(1)
                        if not throughput:
                            throughput_match = throughput_pattern.search(line)
                            if throughput_match:
                                throughput = throughput_match.group(1)
                        if latency and throughput:
                            break

                if latency and throughput:
                    csv_writer.writerow([file_name, latency, throughput])
                    print(f"Extracted from {file_name}: Latency={latency} ms, Throughput={throughput} FPS")
                else:
                    print(f"Could not extract data from {file_name}")


# Devices and hints
devices = ["CPU", "GPU", "NPU"]
hints = ["latency", "tput"]

# Process each file in the blob folder
for file_name in os.listdir(blob_folder):
    for performance_counters in  performance_counters_list:
        if not file_name.endswith(".xml"):
            continue

        blob_path = os.path.join(blob_folder, file_name)
        blob_name = file_name[:-4]  # Remove ".xml" extension

        for device in devices:
            for hint in hints:
                precision_args = get_precision_args(file_name, device)
                nireq_arg = get_nireq_arg(nireq)

                report_folder = os.path.join(report_folder_base, "detailed_log" if performance_counters == "ON" else "benchmark_log")
                os.makedirs(report_folder, exist_ok=True)

                cmd = (
                    f"benchmark_app "
                    f"-m {blob_path} "
                    f"-hint {hint} "
                    f"-t 30 "
                    f"-report_folder {report_folder} "
                    f"--inference_only TRUE "
                    f"{precision_args} "
                    f"{nireq_arg} "
                )

                if performance_counters == "ON":
                    cmd += "-pc -pcsort simple_sort -report_type detailed_counters "

                cmd += f"-d {device} > {report_folder}/{blob_name}_{device}_hint_{hint}.txt"

                print(cmd)
                subprocess.run(cmd, shell=True)

                if performance_counters == "ON":
                    time.sleep(2)  # Wait to ensure reports are written
                    blob_name_1 = f"{blob_name}_{device}"
                    rename_report_files(report_folder, blob_name_1, hint)
                    move_and_rename_profiling(blob_name_1, hint, report_folder)


# Extract data from both folders
folders = ["benchmark_log", "detailed_log"]
for folder in folders:
    log_folder = os.path.join(report_folder_base, folder)
    output_csv = os.path.join(log_folder, "latency_throughput_report.csv")
    if os.path.exists(log_folder):
        extract_log_data(log_folder, output_csv)

print("Processing and data extraction completed.")
