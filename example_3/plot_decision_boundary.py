import numpy as np 
import json 
import matplotlib.pyplot as plt
import pandas as pd 
import os 
from utils import Utils
import matplotlib
import pickle
import argparse
matplotlib.use('Agg')

## get the parameter for any value of beta since we are using the same parameters for all value of beta

def plot_decision_boundary(args):
    plt.figure(figsize=(110, 110)) 
    

   

    
    H0 = 0

    H1 = args.d_0

    #params["d_0"]


    
    ctp = args.ctp
    ctn = args.ctn
    cfp = args.cfp
    cfn = args.cfn
    num_bins_fatigue = args.num_bins_fatigue
   
    
    
  


    ut = Utils(args,H0,H1)


     # compute the auto_cost for different posterior of the tasks

    posteriors_h1 = np.linspace(0,1,21)

    auto_costs= [min(
        ctn + (cfn - ctn) * automation_posterior_h1,
        cfp + (ctp - cfp) * automation_posterior_h1,
    ) for automation_posterior_h1 in posteriors_h1]
    
    plt.plot(posteriors_h1,auto_costs,'*--',label='automation cost',color='red', linewidth=20, markersize=50)

    F_states = np.round(np.linspace(0, 20, num_bins_fatigue + 1), 2)

    #F_states = [0]

    #all_workloads = [9]
    automation_posteriors = [[1-post_h1,post_h1] for post_h1 in posteriors_h1]

    # we are considering taskload form {0,..,20}
    for w_t in range(21):

    #for w_t in all_workloads:
        #computing the human cost for each taskload

        for F_t_idx, F_t in enumerate(F_states):

            hum_costs=[ut.compute_gamma(automation_posterior,w_t,F_t) for automation_posterior in automation_posteriors]

            
            plt.plot(posteriors_h1,hum_costs,color='blue', label='Human (F_t,w_t)=('+str(F_t)+','+str(w_t)+')',linewidth=2)

           

    
                


    plt.grid(True)  # Show grid
    plt.legend()
    plt.tick_params(axis='both', which='major', labelsize=60) 
    plt.xlabel('Posterior_H1',fontsize=60)
    plt.ylabel('cost',fontsize=60)
    plt.xlim(left=0)
    
    

    path1 = 'decision_boundary/beta '+str(args.beta)+'/gamma_'+str(args.gamma)+'/'

    if not os.path.exists(path1):
        try:
            os.makedirs(path1,exist_ok=True)
        
        except FileExistsError:
            pass
    
    

    plt.savefig(path1+'decision_boundary.pdf')

    plt.clf()
    plt.close()

    if args.plot_posterior:

        mega_obs_path = args.results_path + 'num_tasks '+str(args.num_tasks_per_batch)+'/beta '+str(args.beta)+'/gamma_'+str(args.gamma)+'/plot_analysis/cost_comparison/'+'/all_mega_batch.pkl'

        with open(mega_obs_path,'rb') as file:
            all_mega_obs = pickle.load(file)
        
        x_vals = np.arange(20)
        for run in range(len(all_mega_obs)):

            for timestep in range(len(all_mega_obs['Run-'+str(run+1)])):

                plt.scatter(np.arange(1,21,1), all_mega_obs['Run-'+str(run+1)][timestep][2],color='black', s=60)


        plt.xlabel('Number of tasks')
        plt.ylabel('Posterior H1 Value')

    
        plt.savefig(path1+'posterior_h1_freq.pdf')
        plt.clf()
        plt.close()


    


if __name__=='__main__':

    parser = argparse.ArgumentParser(description="Approximate Dynamic Program parameters.")

    parser.add_argument('--beta', type=float, default= 0.5, help='Exponent that influcences the extent to which workload affects the observation channel')
    parser.add_argument('--num_expectation_samples', type=int, default=10, help='Number of expectation samples to take for the approximate Dynamic Program.')
    parser.add_argument('--horizon', type=int, default=20, help='The length of the horizon.')
    parser.add_argument('--d_0',type=float, default= 5, help='The value of d0 in the experiment.')
    parser.add_argument('--prior',default=[0.8,0.2], help='A list containing the prior of [H0, H1].' )
    parser.add_argument('--num_tasks_per_batch', type=int, default=20, help='The total number of tasks in a batch.')
    parser.add_argument('--sigma_a',type=float, default=2.5, help='Automation observation channel variance.')
    parser.add_argument('--sigma_h', type=float, default=1.0, help='Human observation channel variance.')
    parser.add_argument('--w_0', type=int, default=15, help='The workload threshold beyond which fatigue recovery does not occur')
    parser.add_argument('--gamma', type=float, default=0.05, help='The rate at which the fatigue grows')
    parser.add_argument('--num_bins_fatigue', type=int, default=10, help='The number of bins to be used to discretize fatigue')
    
    # Arguments associated with costs
    parser.add_argument('--ctp',type=float, default=0, help='The cost associated with true positive rate')
    parser.add_argument('--ctn',type=float, default=0, help='The cost associated with true negative value.')
    parser.add_argument('--cfp', type=float, default=1.0, help='The cost associated with false positive rates')
    parser.add_argument('--cfn', type=float, default=1.0, help='The cost associated with false negative rates.')
    parser.add_argument('--cm', type=float, default=0.0, help='The cost associated with deferrals.')

    parser.add_argument('--wandb_sync', type=str, default='offline',help='whether to sync wandb results to cloud.')
    parser.add_argument('--use_wandb', type=bool, default=False, help='Whether to use wandb for logging')
    
    parser.add_argument('--results_path', type=str, default='results/', help='The name of the directory to save the results')

    parser.add_argument('--run_eval_only', type=bool, default=False)
    parser.add_argument('--num_eval_runs', type=int, default=10, help="Number of independent runs for monte carlo performance evaluation")
    
    parser.add_argument('--plot_posterior', type=bool, default=False, help='Flag whether to plot the posteriors or not')

    args = parser.parse_args()
    

    plot_decision_boundary(args)




       

