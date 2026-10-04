#!/bin/bash
set -e

source /etc/profile
cd /home/gridsan/hcairney/ALD/smarties

# Initialize Conda, then load the existing project environment.
source "$(conda info --base)/etc/profile.d/conda.sh"
source ./env-cluster.sh

#export LD_LIBRARY_PATH="$PWD/build-burgers-asan/build/lib:$LD_LIBRARY_PATH"
#export ASAN_OPTIONS="detect_leaks=0:halt_on_error=1"

# Allow the MPI rank and its forked environments to use allocated CPUs.
export OMPI_MCA_hwloc_base_binding_policy=none

runname=initial_test_smoothing_train_001_eval_final

python -u bin/smarties.py \
  build-burgers/apps/burgers_sgs/ \
  apps/burgers_sgs/runs/rl/initial_test_smoothing/settings.json \
  --execname burgers_smarties \
  --runprefix apps/burgers_sgs/runs/rl/ \
  --runname "$runname" \
  --restart apps/burgers_sgs/runs/rl/initial_test_smoothing_train_001 \
  --nEvalEpisodes 4 \
  --nEnvironments 1 \
  --nThreads 1 \
  --args "--appSettings ../../../configs/burgers_rl_app_settings.txt --setupFolder ../initial_test_smoothing"