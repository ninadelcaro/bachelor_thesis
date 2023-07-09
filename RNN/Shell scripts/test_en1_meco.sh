#!/bin/bash
#SBATCH -p GPU # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH -t 0-36:00 # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
#SBATCH --gres=gpu:1

source activate nina_bep
cd

#RNN
lang="en"
#converged epoch (todo: automatize this):
epoch=45
seed=2342
#en_rnn_path_to_data="../u802098/final_models/model_${lang}_0delay_seed_${seed}_epoch_${epoch}/accuracies_test.csv"
CUDA_LAUNCH_BLOCKING=1 python test_modified_tuning.py --language $lang --seed ${seed} --delay 1 --min_frequency 0 \
--path_to_model "$(realpath ../u802098/final_models/model_${lang}_1delay_seed_${seed}_epoch_${epoch})" \
--input_data "$(realpath ../u802098/en_meco_lemmas.txt)" 

conda deactivate