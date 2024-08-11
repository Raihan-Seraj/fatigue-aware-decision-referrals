import matplotlib.pyplot as plt 
import os
import numpy as np 
from tqdm import tqdm
import json 
import pandas as pd
from utils import Utils 
import seaborn as sns
import matplotlib
matplotlib.use('Agg')


'''

Function that computes Kesav's algorithm

Args:
    batched_obs: A batch of observation for the automation
    batched_posterior_h0: A list consisting of posterior value of H0 given observation. 
    batched_posterior_h1: A list consisting of posterior value of H1 given observation. 
    F_t: The fatigue state at time t in the range [0.1]
    ut: a utility object from the class Utility

Returns:
    optimal workload for the human

'''

def compute_kesavs_algo(
    batched_posterior_h0,
    batched_posterior_h1,
    F_t,
    ut
):

    all_cost = []
    all_deferred_indices =[]
    for w_t in range(ut.num_tasks_per_batch+1):

        cstar, deferred_indices, gbar = ut.compute_cstar(
            F_t,
            w_t,
            batched_posterior_h0,
            batched_posterior_h1,
           
        )

        all_cost.append(cstar)

        all_deferred_indices.append(deferred_indices)

    min_idx = np.argmin(all_cost)

    min_cost = all_cost[min_idx]

    deferred_idx_dp = all_deferred_indices[min_idx]

    min_wl = min_idx

    return min_wl, deferred_idx_dp


'''
Function that computes the optimal workload using the adp solution
'''
def compute_adp_solution(
    batched_posterior_h0,
    batched_posterior_h1,
    F_t,
    V_bar,
    ut
):

    all_cost = []

    all_deferred_indices=[]

    F_t_idx = ut.discretize_fatigue_state(F_t)

    all_cost = []

    for w_t in range(ut.num_tasks_per_batch):

        F_tp1 = ut.get_fatigue(F_t, w_t)

        F_tp1_idx = ut.discretize_fatigue_state(F_tp1)

        cstar, deferred_indices, gbar = ut.compute_cstar(
            F_t,
            w_t,
            batched_posterior_h0,
            batched_posterior_h1
        )

        future_cost = V_bar[F_tp1_idx]

        total_cost = cstar + future_cost

        all_cost.append(total_cost)

        all_deferred_indices.append(deferred_indices)

    min_idx = np.argmin(all_cost)

    min_cost = all_cost[min_idx]

    wl_dp = min_idx
    
    defrred_idx_dp = all_deferred_indices[min_idx]

    return wl_dp, defrred_idx_dp


def compute_performance(betas, simulation_time=100, num_runs=100, result_path='results/'):

    result_dataset = pd.DataFrame(columns=['Algorithm Name','Beta Value','Avg Cost'])

    for beta in betas:

        param_path = result_path+'num_tasks 20'+'/beta '+str(beta)+'/params.json'

        with open(param_path,'r') as file:
            params = json.load(file)
        
        num_tasks_per_batch = params["num_tasks_per_batch"]

        mu = params["mu"]

        lamda = params["lamda"]

        H0 = params["H0"]

        H1 = params["H1"]

        prior = params["prior"]

        d_0 = params["d_0"]

        beta = params["beta"]

        sigma_h = params["sigma_h"]
        sigma_a = params["sigma_a"]

        ctp = params["ctp"]
        ctn = params["ctn"]
        cfp = params["cfp"]
        cfn = params["cfn"]
        cm = params["cm"]
        num_bins_fatigue = params["num_bins_fatigue"]
        T = params["T"]
        num_expecation_samples = params["num_expectation_samples"]

        w_0 = params["w_0"]

        ut = Utils(num_tasks_per_batch, mu, lamda, w_0, sigma_a, H0, H1, prior, d_0, beta, sigma_h, ctp, ctn, cfp, cfn, cm, num_bins_fatigue)


        V_bar = np.load(result_path+'num_tasks 20/beta '+str(beta)+'/V_bar.npy')

        all_avg_cost_k = np.zeros(num_runs)

        all_avg_cost_adp = np.zeros(num_runs)

        

        for run in tqdm(range(num_runs)):
        
            ## initial fatigue is for kesav 0
            F_k = 0

            #initial fatigue for adp
            F_adp = 0

            cost_adp = 0
            cost_k = 0
            for t in tqdm(range(simulation_time)):


                batched_obs, batched_posterior_h0, batched_posterior_h1=ut.get_auto_obs()

                wl_k, deferred_idx_k = compute_kesavs_algo(batched_posterior_h0, batched_posterior_h1, F_k, ut)

                wl_adp, deferred_idx_adp = compute_adp_solution(batched_posterior_h0,batched_posterior_h1,F_adp, V_bar,ut)

                cost_adp += ut.per_step_cost(F_adp,batched_posterior_h1,deferred_idx_adp)

                cost_k += ut.per_step_cost(F_k,batched_posterior_h1,deferred_idx_k)




                #get the next fatigue state for kesav
                F_k = ut.get_fatigue(F_k, wl_k)

                #get the next fatigue state for adp

                F_adp = ut.get_fatigue(F_adp,wl_adp)

            avg_cost_adp = cost_adp/simulation_time
            all_avg_cost_adp[run]=avg_cost_adp

            avg_cost_k = cost_k/simulation_time
            all_avg_cost_k[run]=avg_cost_k

            new_rows = pd.DataFrame([['ADP', beta, avg_cost_adp],['K-Algorithm',beta,avg_cost_k]], columns=result_dataset.columns)
            #new_row2 = pd.DataFrame([['K-Algorithm', beta, avg_cost_k]], columns=result_dataset.columns)

            result_dataset = pd.concat([result_dataset,new_rows],ignore_index=True)
            

        path1 = 'plot_analysis/'
        if not os.path.exists(path1):
            os.makedirs(path1)
        
        path2 = path1+'cost_comparison/'

        if not os.path.exists(path2):
            os.makedirs(path2)

        path3 = path2+'beta '+str(beta)+'/'

        if not os.path.exists(path3):
            os.makedirs(path3)

        np.save(path3+'all_avg_cost_adp.npy',all_avg_cost_adp)

        np.save(path3+'all_avg_cost_k.npy',all_avg_cost_k)
        


    sns.boxplot(data=result_dataset, x="Beta Value", y="Avg Cost", hue= "Algorithm Name",gap=0.1)
    plt.savefig('plot_analysis/cost_comparison/cost_comparison_beta.pdf')
    
    return 









def run_evalutation(beta, simulation_time=100, result_path='results/'):

    ## loading the parameters

    param_path = result_path+'num_tasks 20'+'/beta '+str(beta)+'/params.json'

    with open(param_path,'r') as file:
        params = json.load(file)
    
    num_tasks_per_batch = params["num_tasks_per_batch"]

    mu = params["mu"]

    lamda = params["lamda"]

    H0 = params["H0"]

    H1 = params["H1"]

    prior = params["prior"]

    d_0 = params["d_0"]

    beta = params["beta"]

    sigma_h = params["sigma_h"]
    sigma_a = params["sigma_a"]

    ctp = params["ctp"]
    ctn = params["ctn"]
    cfp = params["cfp"]
    cfn = params["cfn"]
    cm = params["cm"]
    num_bins_fatigue = params["num_bins_fatigue"]
    T = params["T"]
    num_expecation_samples = params["num_expectation_samples"]

    w_0 = params["w_0"]

    

    ut = Utils(num_tasks_per_batch, mu, lamda, w_0, sigma_a, H0, H1, prior, d_0, beta, sigma_h, ctp, ctn, cfp, cfn, cm, num_bins_fatigue)


    V_bar = np.load(result_path+'num_tasks 20/beta '+str(beta)+'/V_bar.npy')


    ## initial fatigue is for kesav 0
    F_k = 0

    #initial fatigue for adp
    F_adp = 0

    fatigue_evolution_kesav = []
    fatigue_evolution_adp = []

    taskload_evolution_kesav = []
    taskload_evolution_adp =[]

    for t in tqdm(range(simulation_time)):

        fatigue_evolution_kesav.append(F_k)
        fatigue_evolution_adp.append(F_adp)

        batched_obs, batched_posterior_h0, batched_posterior_h1=ut.get_auto_obs()

        wl_k, deferred_idx_k = compute_kesavs_algo(batched_posterior_h0, batched_posterior_h1, F_k, ut)

        wl_adp, deferred_idx_adp = compute_adp_solution(batched_posterior_h0,batched_posterior_h1,F_adp, V_bar,ut)




        #get the next fatigue state for kesav
        F_k = ut.get_fatigue(F_k, wl_k)

        #get the next fatigue state for adp

        F_adp = ut.get_fatigue(F_adp,wl_adp)

        taskload_evolution_adp.append(wl_adp)
        taskload_evolution_kesav.append(wl_k)
    
    return fatigue_evolution_kesav,fatigue_evolution_adp, taskload_evolution_kesav, taskload_evolution_adp





def main():

    betas = [0.1,0.3,0.5,0.8]
    #betas = [0.1]
    for beta in betas:

        fatigue_evolution_kesav,fatigue_evolution_adp, taskload_evolution_kesav, taskload_evolution_adp = run_evalutation(beta)

        ## plotting the level of fatigue 

        path1 = 'plot_analysis/'

        if not os.path.exists(path1):
            os.makedirs(path1)
        
       

        plt.plot(fatigue_evolution_adp,label='Approx dynamic program',color='orange')
        plt.plot(fatigue_evolution_kesav,label='Kesav', color='black')

        plt.xlabel('Time')
        plt.ylabel('Fatigue Level')
        plt.legend()
        plt.savefig(path1+'beta_'+str(beta)+'_fatigue.pdf')

        plt.clf()
        plt.close()


        plt.plot(taskload_evolution_adp,label='Approx dynamic program',color='orange')
        plt.plot(taskload_evolution_kesav,label='Kesav', color='black')

        plt.xlabel('Time')
        plt.ylabel('Workload level')
        plt.legend()
        plt.savefig(path1+'beta_'+str(beta)+'_workload.pdf')

        plt.clf()
        plt.close()
    
    
    compute_performance(betas)


if __name__=='__main__':

    main()

        


