
alphas=(0.05 0.01 0.03)
betas=(0.5 1)
gammas=(0.05 0.08 0.02 0.09 0.3)

for alpha in ${alphas[@]}
do
    for beta in ${betas[@]}

    do
        for gamma in ${gammas[@]}

        do

            echo "#!/bin/bash" >> temprun.sh
            echo "#SBATCH --account=def-adityam" >> temprun.sh
            echo "#SBATCH --output=\"/scratch/raihan08/slurm-%j.out\"" >> temprun.sh
            # echo "#SBATCH --job-name=TMBIBTEX" >> temprun.sh
            echo "#SBATCH --cpus-per-task=4"  >> temprun.sh   # ask for 8 CPUs
            #echo "#SBATCH --gres=gpu:2" >> temprun.sh         # ask for 2 GPU
            # echo "#SBATCH --gres=gpu:1" >> temprun.sh         # ask for 2 GPU
            echo "#SBATCH --mem=32G" >> temprun.sh            # ask for 64 GB RAM
            echo "#SBATCH --time=24:00:00" >> temprun.sh
            echo "source ../../thesis/bin/activate" >>temprun.sh
            #echo "wandb login e0273d1f1df1e15bffa4b6bca33edb700bc9d54c" >>temprun.s
            echo "#SBATCH --mail-user=raihanseraj@gmail.com" >> temprun.sh
            echo "#SBATCH --mail-type=AL" >> temprun.sh
            echo "python approximate_dp.py --alpha ${alpha} --beta ${beta} --gamma ${gamma} --num_expectation_samples 500 --num_eval_runs 500 --Fmax 100 --num_bins_fatigue 20" >> temprun.sh
            eval "sbatch temprun.sh"
            rm temprun.sh

        done
    done
done


  
