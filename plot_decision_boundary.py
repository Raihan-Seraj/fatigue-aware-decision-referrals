import numpy as np 
import json 
import matplotlib.pyplot as plt
import pandas as pd 
import os 
from utils import Utils
import matplotlib
import pickle
matplotlib.use('Agg')

## get the parameter for any value of beta since we are using the same parameters for all value of beta

def plot_decision_boundary(result_path, beta,plot_posterior=False):
    plt.figure(figsize=(110, 110)) 
    

   

    num_tasks_per_batch = 20

    mu = 0.03
    lamda = 0.05

    d_0 = 3

    H0 = 0

    H1 = d_0

    prior = [0.8,0.2]

    #params["d_0"]


    sigma_h = 1.0
    sigma_a = 1.5

    ctp = 0#params["ctp"]
    ctn = 0#params["ctn"]
    cfp = 1#params["cfp"]
    cfn = 1#params["cfn"]
    cm = 0
    num_bins_fatigue = 20
    T = 20
    num_expecation_samples = 100

    w_0 = 15#params["w_0"]


    ut = Utils(num_tasks_per_batch, mu, lamda, w_0, sigma_a, H0, H1, prior, d_0, beta, sigma_h, ctp, ctn, cfp, cfn, cm, num_bins_fatigue)


     # compute the auto_cost for different posterior of the tasks

    posteriors_h1 = np.linspace(0,1,21)

    auto_costs= [min(
        ctn + (cfn - ctn) * automation_posterior_h1,
        cfp + (ctp - cfp) * automation_posterior_h1,
    ) for automation_posterior_h1 in posteriors_h1]
    
    plt.plot(posteriors_h1,auto_costs,'*--',label='automation cost',color='red', linewidth=20, markersize=50)

    F_states = np.round(np.linspace(0, 1, num_bins_fatigue + 1), 2)

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
    
    

    path1 = 'decision_boundary/'

    if not os.path.exists(path1):

        os.makedirs(path1)
    
    path2 = path1+'beta '+str(beta)+'/'

    if not os.path.exists(path2):
        os.makedirs(path2)
    

    plt.savefig(path2+'decision_boundary.pdf')

    plt.clf()
    plt.close()

    if plot_posterior:

        mega_obs_path = result_path+'plot_analysis/cost_comparison/beta '+str(beta)+'/all_mega_batch.pkl'

        with open(mega_obs_path,'rb') as file:
            all_mega_obs = pickle.load(file)
        
        x_vals = np.arange(20)
        for run in range(len(all_mega_obs)):

            for timestep in range(len(all_mega_obs['Run-'+str(run+1)])):

                plt.scatter(np.arange(1,21,1), all_mega_obs['Run-'+str(run+1)][timestep][2],color='black', s=60)


        plt.xlabel('Number of tasks')
        plt.ylabel('Posterior H1 Value')

    
        plt.savefig(path2+'posterior_h1_freq.pdf')
        plt.clf()
        plt.close()


    


if __name__=='__main__':

    betas = [0.3,0.5,0.7,0.9]


    result_path = 'results/'

    for beta in betas:

        plot_decision_boundary(result_path, beta,False)




       

