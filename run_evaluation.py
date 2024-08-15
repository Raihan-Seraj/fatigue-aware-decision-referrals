import matplotlib.pyplot as plt 
import os
import numpy as np 
from tqdm import tqdm
import json 
import pandas as pd
from utils import Utils 
import seaborn as sns
import matplotlib
import multiprocessing
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


def compute_performance(betas,result_path,lamda_new, simulation_time=100, num_runs=10):

    result_dataset = pd.DataFrame(columns=['Algorithm Name','Beta Value','Avg Cost'])

    
    for beta in betas:


        print("Computing peformance with beta = "+str(beta)+'\n')

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
        
        #initializing the utility with a different value of lambda (different than the one used for training)
        ut_new = Utils(num_tasks_per_batch, mu, lamda_new, w_0, sigma_a, H0, H1, prior, d_0, beta, sigma_h, ctp, ctn, cfp, cfn, cm, num_bins_fatigue)


        V_bar = np.load(result_path+'num_tasks 20/beta '+str(beta)+'/V_bar.npy')

        all_perf_k = np.zeros(num_runs)

        all_perf_adp = np.zeros(num_runs)

        all_perf_adp_new = np.zeros(num_runs)

        

        for run in tqdm(range(num_runs)):
        
            ## initial fatigue is for kesav 0
            F_k = 0

            #initial fatigue for adp
            F_adp = 0

            F_adp_new=0

            cost_adp = 0
            cost_k = 0
            cost_adp_new=0
            for t in range(simulation_time):


                batched_obs, batched_posterior_h0, batched_posterior_h1=ut.get_auto_obs()

                wl_k, deferred_idx_k = compute_kesavs_algo(batched_posterior_h0, batched_posterior_h1, F_k, ut)

                wl_adp, deferred_idx_adp = compute_adp_solution(batched_posterior_h0,batched_posterior_h1,F_adp, V_bar,ut)

                wl_adp_new, deferred_idx_adp_new = compute_adp_solution(batched_posterior_h0,batched_posterior_h1,F_adp_new, V_bar,ut_new)

                
                cost_adp += ut.per_step_cost(F_adp,batched_posterior_h1,deferred_idx_adp)

                cost_k += ut.per_step_cost(F_k,batched_posterior_h1,deferred_idx_k)

                cost_adp_new += ut_new.per_step_cost(F_adp_new,batched_posterior_h1,deferred_idx_adp)



                #get the next fatigue state for kesav
                F_k = ut.get_fatigue(F_k, wl_k)

                #get the next fatigue state for adp

                F_adp = ut.get_fatigue(F_adp,wl_adp)

                F_adp_new = ut_new.get_fatigue(F_adp_new,wl_adp_new)

            perf_adp = cost_adp
            all_perf_adp[run]=perf_adp

            perf_k = cost_k
            all_perf_k[run]=perf_k

            perf_adp_new = cost_adp_new
            all_perf_adp_new[run] = perf_adp_new

            new_rows = pd.DataFrame([['ADP', beta, perf_adp],['ADP-corr', beta, perf_adp_new],['K-Algorithm',beta,perf_k]], columns=result_dataset.columns)
            #new_row2 = pd.DataFrame([['K-Algorithm', beta, avg_cost_k]], columns=result_dataset.columns)

            result_dataset = pd.concat([result_dataset,new_rows],ignore_index=True)
            

        path1 = result_path+'plot_analysis/'
        if not os.path.exists(path1):
            os.makedirs(path1)
        
        path2 = path1+'cost_comparison/'

        if not os.path.exists(path2):
            os.makedirs(path2)

        path3 = path2+'beta '+str(beta)+'/'

        if not os.path.exists(path3):
            os.makedirs(path3)

        np.save(path3+'all_perf_adp.npy',all_perf_adp)

        np.save(path3+'all_perf_adp_new.npy',all_perf_adp_new)

        np.save(path3+'all_perf_k.npy',all_perf_k)
        
        result_dataset.to_csv(result_path+'plot_analysis/plot_data.csv')

        sns.boxplot(data=result_dataset, x="Beta Value", y="Avg Cost", hue= "Algorithm Name",gap=0.1)
        plt.savefig(result_path+'plot_analysis/cost_comparison/cost_comparison_beta.pdf')
        plt.clf()
        plt.close()

    return
        
 









def run_evalutation(beta, result_path,lamda_new, simulation_time=100):

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

    ut_new = Utils(num_tasks_per_batch, mu, lamda_new, w_0, sigma_a, H0, H1, prior, d_0, beta, sigma_h, ctp, ctn, cfp, cfn, cm, num_bins_fatigue)


    V_bar = np.load(result_path+'num_tasks 20/beta '+str(beta)+'/V_bar.npy')


    ## initial fatigue is for kesav 0
    F_k = 0

    #initial fatigue for adp
    F_adp = 0

    #initial fatigue for the new model
    F_adp_new=0

    fatigue_evolution_kesav = []
    fatigue_evolution_adp = []
    fatigue_evolution_adp_new = []

    taskload_evolution_kesav = []
    taskload_evolution_adp =[]
    taskload_evolution_adp_new =[]

    for t in tqdm(range(simulation_time)):

        fatigue_evolution_kesav.append(F_k)
        fatigue_evolution_adp.append(F_adp)
        fatigue_evolution_adp_new.append(F_adp_new)

        batched_obs, batched_posterior_h0, batched_posterior_h1=ut.get_auto_obs()

        wl_k, deferred_idx_k = compute_kesavs_algo(batched_posterior_h0, batched_posterior_h1, F_k, ut)

        wl_adp, deferred_idx_adp = compute_adp_solution(batched_posterior_h0,batched_posterior_h1,F_adp, V_bar,ut)

        wl_adp_new, deferred_idx_adp_new = compute_adp_solution(batched_posterior_h0,batched_posterior_h1,F_adp_new, V_bar,ut_new)




        #get the next fatigue state for kesav
        F_k = ut.get_fatigue(F_k, wl_k)

        #get the next fatigue state for adp

        F_adp = ut.get_fatigue(F_adp,wl_adp)

        F_adp_new = ut_new.get_fatigue(F_adp_new,wl_adp_new)

        taskload_evolution_adp.append(wl_adp)
        taskload_evolution_adp_new.append(wl_adp_new)
        taskload_evolution_kesav.append(wl_k)
    
    return fatigue_evolution_kesav,fatigue_evolution_adp,fatigue_evolution_adp_new, taskload_evolution_kesav, taskload_evolution_adp, taskload_evolution_adp_new




def run_perf_eval(beta, result_path,lamda_new):
    
    fatigue_evolution_kesav,fatigue_evolution_adp,fatigue_evolution_adp_new, taskload_evolution_kesav, taskload_evolution_adp, taskload_evolution_adp_new = run_evalutation(beta,result_path,lamda_new)

        ## plotting the level of fatigue 

    path1 = result_path+'plot_analysis/'

    if not os.path.exists(path1):
        try:
            os.makedirs(path1,exist_ok=True)
        except FileExistsError:
            pass
    
    

    plt.plot(fatigue_evolution_adp,label='Approx dynamic program',color='orange')
    plt.plot(fatigue_evolution_adp_new,label='Approx dynamic program - corr',color='red')
    plt.plot(fatigue_evolution_kesav,label='Kesav', color='black')

    plt.xlabel('Time')
    plt.ylabel('Fatigue Level')
    plt.legend()
    plt.savefig(path1+'beta_'+str(beta)+'_fatigue.pdf')

    plt.clf()
    plt.close()


    plt.plot(taskload_evolution_adp,label='Approx dynamic program',color='orange')
    plt.plot(taskload_evolution_adp_new,label='Approx dynamic program-Corr',color='red')
    plt.plot(taskload_evolution_kesav,label='Kesav', color='black')

    plt.xlabel('Time')
    plt.ylabel('Workload level')
    plt.legend()
    plt.savefig(path1+'beta_'+str(beta)+'_workload.pdf')

    plt.clf()
    plt.close()

    return

   


def main():

    #betas = np.round(np.linspace(0.1,0.9,9),1)
    
    betas = [0.2,0.4,0.6,0.8]

    result_path = "test/"
    lamda_new=0.01

    inputs = [(beta,result_path,lamda_new) for beta in betas]
    
    
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.starmap(run_perf_eval, inputs)

   
    
    compute_performance(betas, result_path,lamda_new)

        


if __name__=='__main__':

    main()

        


