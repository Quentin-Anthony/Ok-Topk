#!/bin/bash

# Sample Slurm job script
#   for TACC Longhorn Nodes
#
#------------------------------------------------------------------------------

#SBATCH -J bertkfc                 # Job name
#SBATCH -o sbatch_logs/bertkfc.o%j # Name of stdout output file
#SBATCH -N 4                      # Total # of nodes
#SBATCH -n 40                      # Total # of mpi tasks
#SBATCH -t 12:00:00                # Run time (hh:mm:ss)
#SBATCH -p gpu-a100
#SBATCH -A Deep-Learning-at-Sca    # Allocation

mkdir -p sbatch_logs
module unload spectrum_mpi

cd /scratch/06632/quentin1/work-topk
source setup.sh

export MV2_USE_CUDA=1
export MV2_ENABLE_AFFINITY=1
export MV2_THREADS_PER_PROCESS=2
export MV2_SHOW_CPU_BINDING=1
export MV2_CPU_BINDING_POLICY=hybrid
export MV2_HYBRID_BINDING_POLICY=spread
export MV2_USE_RDMA_CM=0
export MV2_SUPPORT_DL=1

module load cuda/11.4 cmake gcc/9.4.0
module unload intel impi python3
module list

conda activate py38_topk
which nvcc
nvidia-smi

which mpicc
which python

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
echo $MASTER_ADDR

srun python -m mpi4py main_bert.py \
        --module models.bert12.depth=4 \
        --max_seq_length 128 \
        --train_batch_size 8 \
        --train_path ./bert_data/wikipedia.segmented.nltk.txt \
        --bert_config_path configs/bert_config_bert-base-uncased.json \
        --vocab_path ./bert_data/bert-large-uncased-vocab.txt \
        --do_train \
        --do_lower_case \
        --num_minibatches 512 \
	--density 0.01 \
	--compressor 'topkA_stable' \
        --gradient_accumulation_steps 1 --dataparallel --config_path tests/depth=4/conf_32nodes.json


