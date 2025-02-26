# -*- coding: utf-8 -*-
#
# This script is a python executable computing a change detection algorithm on time series of SAR images.
# Usage: python change-detection.py --storage_path [PATH_TO_FOLDER_TO_STORE_RESULTS] --image [PATH_TO_FOLDER_WITH_IMAGES] --window [WINDOW_SIZE] --cores [NUMBER_OF_CORES_USED] --number_run [NUMBER_OF_RUNS] --robust [ROBUSTNESS ID]
#
# Author: Matthieu Verlynde
# Email: matthieu.verlynde@univ-smb.fr
# Date: 26 Feb 2025
# Version: 1.0.0

import os
import argparse
import subprocess
import pandas as pd


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", '-i', type=str, default='/home/verlyndem/Data/Selection/500x500x4')
    parser.add_argument("--window", '-w', type=int, required=True)
    parser.add_argument("--cores", '-c', type=float, required=True)
    parser.add_argument("--number_run", '-n', type=int, default=1)
    parser.add_argument("--storage_path", type=str, required=True)
    parser.add_argument("--robust", type=float, default=0)
    args = parser.parse_args()

    results_path = os.path.join(args.storage_path, "results.txt")
    times_path = os.path.join(args.storage_path, "times.txt")

    os.makedirs(os.path.join(args.storage_path, "output"), exist_ok=True)

    command = f"bash performance-tracking/experiments/conso/simulation_metrics_exec.sh {results_path} {times_path} {args.number_run} python3 performance-tracking/experiments/conso_change/cd_sklearn_pair_var.py --image {args.image} --window {args.window} --cores {int(args.cores)} --robust {int(args.robust)} --storage_path {args.storage_path}"

    result = subprocess.run(command, shell=True)

