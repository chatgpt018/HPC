#!/usr/bin/env bash
#SBATCH --job-name=market-similarity
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --output=market-similarity-%j.out
set -euo pipefail
# Submission example: sbatch --export=ALL,DATASET=/shared/inputs/reports.npz,RESULT=/shared/results/similarity.json infrastructure/slurm/similarity.sbatch
# Set these to real, accessible shared filesystem paths before submitting.
: "${DATASET:?Provide DATASET as shared input path}"
: "${RESULT:?Provide RESULT as shared output path}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$(dirname "$RESULT")"
srun --mpi=pmix python -m marketintel.mpi_tasks similarity \
  --input "$DATASET" --output "$RESULT" --expected-processes "$SLURM_NTASKS" --top-k 5
