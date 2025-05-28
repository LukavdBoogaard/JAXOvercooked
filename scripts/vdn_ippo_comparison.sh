#!/bin/bash
#SBATCH -p gpu_a100
#SBATCH --time 12:00:00
#SBATCH --gres gpu:1
#SBATCH --job-name=overcooked
#SBATCH -o /home/lvdenboogaard/slurm/%A_%a.out
#SBATCH --array=0-120%5

module load 2022
module load 2023
module load CUDA/12.4.0
module load Python/3.10.4-GCCcore-11.3.0
source /etc/bashrc
source ~/.bashrc
source ~/venv/bin/activate

seeds=(0 1 2 3 4 5)
seq_lengths=(5 15)
# num_envs=(64 128)
# test_interval=(0.01 0.05)


tid=$SLURM_ARRAY_TASK_ID  
n_seeds=${#seeds[@]}
n_seq_lengths=${#seq_lengths[@]}
# num_envs=${#num_envs[@]}
# n_num_steps=${#num_steps[@]}
# n_test_interval=${#test_interval[@]}


index() {                            # index tid base₀ base₁ …
    local val=$1; shift
    local bases=("$@")
    local i
    for (( i=0; i<${#bases[@]}; i++ )); do
        local stride=1
        for (( j=i+1; j<${#bases[@]}; j++ )); do
            stride=$(( stride * bases[j] ))
        done
        printf '%s ' $(( val / stride ))
        val=$(( val % stride ))
    done
}

# read i_seed i_architecture i_tag <<<"$(index "$tid" "$n_seeds" "$n_architectures" "$n_layouts")"/
read i_seed i_seq_length <<<"$(index "$tid" "$n_seeds" "$n_seq_lengths" )"


seed=${seeds[$i_seed]}
seq_length=${seq_lengths[$i_seq_length]}
num_env=${num_envs[$i_num_envs]}
num_steps=${num_steps[$i_num_steps]}
test_interval=${test_interval[$i_test_interval]}

echo "Running experiment with seed $seed, seq_length $seq_length"

PYTHONPATH=$HOME/JAXOvercooked python $HOME/JAXOvercooked/baselines/IPPO_CL.py \
  --seq_length ${seq_length} \
  --seed ${seed} \
  --tags $seed $seq_length easy_levels\
  --group "experiment 1" \
  --layouts "easy_levels" \
  --anneal_lr \

