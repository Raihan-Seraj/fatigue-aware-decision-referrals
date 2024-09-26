
alphas=(0.2 0.4 0.6 0.8)
gammas=(0.05)
betas=(0.2 0.4 0.6 0.8)


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
            echo "#SBATCH --time=30:00:00" >> temprun.sh
            echo "source ../../thesis/bin/activate" >>temprun.sh
            echo "#SBATCH --mail-user=raihanseraj@gmail.com" >> temprun.sh
            echo "#SBATCH --mail-type=AL" >> temprun.sh
            echo "python approximate_dp.py --alpha ${alpha} --beta ${beta} --gamma ${gamma} --num_expectation_samples 500 --Fmax 30 --num_bins_fatigue 20"  >> temprun.sh
            eval "sbatch temprun.sh"
            rm temprun.sh

    
        done
    done
done



  
