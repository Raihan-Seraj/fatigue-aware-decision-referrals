import numpy as np 
import json 
import matplotlib.pyplot as plt
import pandas as pd 
import os 
from utils import Utils
import matplotlib
matplotlib.use('Agg')

## get the parameter for any value of beta since we are using the same parameters for all value of beta

def plot_decision_boundary(result_path, beta):
    plt.figure(figsize=(110, 110)) 
    

   

    num_tasks_per_batch = 20

    mu = 0.03
    lamda = 0.05

    H0 = 0

    H1 = 1

    prior = [0.8,0.2]

    d_0 = 4#params["d_0"]


    sigma_h = 1
    sigma_a = 2

    ctp = 1.5#params["ctp"]
    ctn = 1.5#params["ctn"]
    cfp = 8#params["cfp"]
    cfn = 8#params["cfn"]
    cm = 0
    num_bins_fatigue = 10
    T = 20
    num_expecation_samples = 100

    w_0 = 12#params["w_0"]


    ut = Utils(num_tasks_per_batch, mu, lamda, w_0, sigma_a, H0, H1, prior, d_0, beta, sigma_h, ctp, ctn, cfp, cfn, cm, num_bins_fatigue)


     # compute the auto_cost for different posterior of the tasks

    posteriors_h1 = np.linspace(0,1,21)

    auto_costs= [min(
        ctn + (cfn - ctn) * automation_posterior_h1,
        cfp + (ctp - cfp) * automation_posterior_h1,
    ) for automation_posterior_h1 in posteriors_h1]

    plt.plot(posteriors_h1,auto_costs,'*--',label='automation cost',color='red', linewidth=20, markersize=50)

    F_states = np.round(np.linspace(0, 1, num_bins_fatigue + 1), 2)

    automation_posteriors = [[1-post_h1,post_h1] for post_h1 in posteriors_h1]

    # we are considering taskload form {0,..,20}
    for w_t in range(21):

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

    


if __name__=='__main__':

    betas = [0.1,0.3,0.5,0.7,0.9]


    result_path = 'results/'

    for beta in betas:

        plot_decision_boundary(result_path, beta)




       

