
betas=(1 2 5 7 9)
mus=(0.003 0.05 0.07 0.1)
lamdas=(0.007 0.03 0.07 0.1)

for beta in ${betas[@]}
do
    for mu in ${mus[@]}

    do
        for lamda in ${lamdas[@]}

        do

            echo "#!/bin/bash" >> temprun.sh
            echo "#SBATCH --account=def-adityam" >> temprun.sh
            echo "#SBATCH --output=\"/scratch/raihan08/slurm-%j.out\"" >> temprun.sh
            # echo "#SBATCH --job-name=TMBIBTEX" >> temprun.sh
            echo "#SBATCH --cpus-per-task=8"  >> temprun.sh   # ask for 8 CPUs
            #echo "#SBATCH --gres=gpu:2" >> temprun.sh         # ask for 2 GPU
            # echo "#SBATCH --gres=gpu:1" >> temprun.sh         # ask for 2 GPU
            echo "#SBATCH --mem=64G" >> temprun.sh            # ask for 64 GB RAM
            echo "#SBATCH --time=48:00:00" >> temprun.sh
            echo "source ../../thesis/bin/activate" >>temprun.sh
            echo "wandb login e0273d1f1df1e15bffa4b6bca33edb700bc9d54c" >>temprun.s
            echo "#SBATCH --mail-user=raihanseraj@gmail.com" >> temprun.sh
            echo "#SBATCH --mail-type=AL" >> temprun.sh
            echo "python approximate_dp.py --beta ${beta} --mu ${mu} --lamda ${lamda} --num_expectation_samples 500" >> temprun.sh
            eval "sbatch temprun.sh"
            rm temprun.sh

        done
    done
done


  
