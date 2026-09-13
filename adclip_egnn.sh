
#!/bin/bash
source /nethome/arouvalis/PhD/retrieval_task/scripts/server_files/condor_setup.sh

export ADCLIP_CIF_DIR=/data/users_old/arouvalis/boltz2/adomain_substrate_complex/cif
SEED=$1

$PYTHON_BIN/python /nethome/arouvalis/PhD/retrieval_task/scripts/analysis/main/train.py \
    --seed $SEED
