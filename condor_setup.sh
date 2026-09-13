#!/bin/bash

# Source the GPU renaming script
source /nethome/arouvalis/PhD/retrieval_task/scripts/server_files/rename_gpus.sh

set -eux
# Get conda env name from argument, default to 'lsv' if none provided
CONDA_ENV="${1:-base}"

# 1) allow HF caches & tokens
#source /etc/profile.d/hf_cache.sh
export HF_HOME="/data/users/arouvalis/hf_home"
export HF_TOKEN=""

export WANDB_API_KEY=""
export WANDB_CACHE_DIR=/data/users/arouvalis/.cache/wandb
export WANDB_DATA_DIR=/data/users/arouvalis/.wandb

export PYTHON_BIN="/nethome/arouvalis/miniconda3/envs/a_domain_generation/bin"

