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

runname=initial_test_radius_3_train_002

python -u bin/smarties.py \
  build-burgers/apps/burgers_sgs/ \
  apps/burgers_sgs/runs/rl/initial_test_radius_3/settings.json \
  --execname burgers_smarties \
  --runprefix apps/burgers_sgs/runs/rl/ \
  --runname "$runname" \
  --restart apps/burgers_sgs/runs/rl/initial_test_radius_3_train_001 \
  --nProcesses 1 \
  --nLearners 1 \
  --nEnvironments 8 \
  --mpiProcsPerEnv 0 \
  --nThreads 4 \
  --nTrainSteps 1000000 \
  --args "--randSeed 5489 --appSettings ../../../configs/burgers_rl_app_settings.txt --setupFolder ../initial_test_radius_3 --redirectAppStdoutToFile 0"