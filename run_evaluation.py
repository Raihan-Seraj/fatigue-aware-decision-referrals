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
import pickle
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

    for w_t in range(ut.num_tasks_per_batch+1):

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


def compute_perf_multiprocess(beta,result_path, lamda_new, simulation_time, num_runs,result_dataset,performance_data):

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

    ut_k = Utils(num_tasks_per_batch, mu, lamda, w_0, sigma_a, H0, H1, prior, d_0, beta, sigma_h, ctp, ctn, cfp, cfn,0, num_bins_fatigue)


    V_bar = np.load(result_path+'num_tasks 20/beta '+str(beta)+'/V_bar.npy')

    all_auto_cost_k = np.zeros(num_runs)
    all_human_cost_k = np.zeros(num_runs)
    all_deferred_cost_k = np.zeros(num_runs)

    all_auto_cost_adp = np.zeros(num_runs)
    all_human_cost_adp = np.zeros(num_runs)
    all_deferred_cost_adp = np.zeros(num_runs)
    
    
    all_auto_cost_adp_new = np.zeros(num_runs)
    all_human_cost_adp_new = np.zeros(num_runs)
    all_deferred_cost_adp_new = np.zeros(num_runs)

    all_human_wl_adp = {}
    all_human_wl_k={}

    all_mega_batch = {}

    for run in tqdm(range(num_runs)):
    
        ## initial fatigue is for kesav 0
        F_k = 0

        #initial fatigue for adp
        F_adp = 0

        F_adp_new=0

        auto_cost_adp = 0
        human_cost_adp = 0
        deferred_cost_adp = 0

        auto_cost_k = 0
        human_cost_k=0
        deferred_cost_k=0

        auto_cost_adp_new=0
        human_cost_adp_new=0
        deferred_cost_adp_new = 0

        mega_batch = [ut.get_auto_obs() for _ in range(simulation_time)]
        all_mega_batch['Run-'+str(run+1)]=mega_batch
        hum_wl_adp = np.zeros(simulation_time)
        hum_wl_k = np.zeros(simulation_time)
        for t in range(simulation_time):


            batched_obs, batched_posterior_h0, batched_posterior_h1=mega_batch[t]

            wl_k, deferred_idx_k = compute_kesavs_algo(batched_posterior_h0, batched_posterior_h1, F_k, ut_k)

            wl_adp, deferred_idx_adp = compute_adp_solution(batched_posterior_h0,batched_posterior_h1,F_adp, V_bar,ut)

            wl_adp_new, deferred_idx_adp_new = compute_adp_solution(batched_posterior_h0,batched_posterior_h1,F_adp_new, V_bar,ut_new)

            hum_wl_adp[t]=wl_adp
            hum_wl_k[t]=wl_k
            
            
            a_cost_adp, h_cost_adp,def_cost_adp = ut.per_step_cost(F_adp,batched_posterior_h1,deferred_idx_adp)

            a_cost_k, h_cost_k, def_cost_k = ut_k.per_step_cost(F_k,batched_posterior_h1,deferred_idx_k)

            a_cost_adp_new, h_cost_adp_new, def_cost_adp_new = ut_new.per_step_cost(F_adp_new,batched_posterior_h1,deferred_idx_adp_new)
            

            auto_cost_adp+=a_cost_adp
            human_cost_adp+=h_cost_adp
            deferred_cost_adp += def_cost_adp

            auto_cost_k += a_cost_k
            human_cost_k += h_cost_k
            deferred_cost_k += def_cost_k

            auto_cost_adp_new+=a_cost_adp_new
            human_cost_adp_new += h_cost_adp_new
            deferred_cost_adp_new += def_cost_adp_new


            #get the next fatigue state for kesav
            F_k = ut_k.get_fatigue(F_k, wl_k)

            #get the next fatigue state for adp

            F_adp = ut.get_fatigue(F_adp,wl_adp)

            F_adp_new = ut_new.get_fatigue(F_adp_new,wl_adp_new)

        all_human_wl_adp['Run-'+str(run+1)]=hum_wl_adp
        all_human_wl_k['Run-'+str(run+1)]=hum_wl_k


        all_auto_cost_adp[run]= auto_cost_adp
        all_human_cost_adp[run] = human_cost_adp
        all_deferred_cost_adp[run] = deferred_cost_adp

        all_auto_cost_k[run]= auto_cost_k
        all_human_cost_k[run] = human_cost_k
        all_deferred_cost_k[run] = deferred_cost_k


        all_auto_cost_adp_new[run]= auto_cost_adp_new
        all_human_cost_adp_new[run] = human_cost_adp_new
        all_deferred_cost_adp_new[run] = deferred_cost_adp_new
        
        #result_dataset = pd.DataFrame(columns=['Algorithm Name','Beta Value','Automation Cost', 'Human Cost','Deferred Cost'])
        
        new_rows = pd.DataFrame([['ADP', beta, auto_cost_adp, human_cost_adp, deferred_cost_adp],['ADP-Lambda=0.01', beta, auto_cost_adp_new, human_cost_adp_new, deferred_cost_adp_new],['K-Algorithm',beta, auto_cost_k, human_cost_k, deferred_cost_k]], columns=result_dataset.columns)
        #new_row2 = pd.DataFrame([['K-Algorithm', beta, avg_cost_k]], columns=result_dataset.columns)

        result_dataset = pd.concat([result_dataset,new_rows],ignore_index=True)
        

    path1 = result_path+'plot_analysis/'
    if not os.path.exists(path1):
        try:
            os.makedirs(path1,exist_ok=True)
        except FileExistsError:
            pass
    
    path2 = path1+'cost_comparison/'

    if not os.path.exists(path2):
        try:
            os.makedirs(path2,exist_ok=True)
        except FileExistsError:
            pass

    path3 = path2+'beta '+str(beta)+'/'

    if not os.path.exists(path3):
        try:
            os.makedirs(path3,exist_ok=True)
        except FileExistsError:
            pass


    
    with open(path3 + 'all_human_wl_adp.pkl','wb') as file:
        pickle.dump(all_human_wl_adp,file)

    
    with open(path3 + 'all_human_wl_k.pkl','wb') as file2:
        pickle.dump(all_human_wl_k,file2)
    

    with open(path3 + 'all_mega_batch.pkl','wb') as file3:
        pickle.dump(all_mega_batch,file3)

    
    np.save(path3+'all_auto_cost_adp.npy',all_auto_cost_adp)

    np.save(path3+'all_human_cost_adp.npy',all_human_cost_adp)

    np.save(path3+'all_deferred_cost_adp.npy',all_deferred_cost_adp)

    
    np.save(path3+'all_auto_cost_adp_new.npy',all_auto_cost_adp_new)

    np.save(path3+'all_human_cost_adp_new.npy',all_human_cost_adp_new)

    np.save(path3+'all_deferred_cost_adp_new.npy',all_deferred_cost_adp_new)


    np.save(path3+'all_auto_cost_k.npy',all_auto_cost_k)

    np.save(path3+'all_human_cost_k.npy',all_human_cost_k)

    np.save(path3+'all_deferred_cost_k.npy',all_deferred_cost_k)


    
    
    return




def compute_performance(betas,result_path,lamda_new, simulation_time, num_runs=100):

    result_dataset = pd.DataFrame(columns=['Algorithm Name','Beta Value','Automation Cost', 'Human Cost','Deferred Cost'])

    performance_data = pd.DataFrame(columns=['Beta', 'ADP Human Cost','ADP Automation Cost','ADP Deferred Cost','ADP Total Cost',
                                             'K Algorithm Human Cost', 'K Algorithm Automation Cost','K Deferred Cost','K Algorithm Total Cost'])
                                           

    inputs = [(beta,result_path,lamda_new,simulation_time, num_runs,result_dataset,performance_data) for beta in betas]                 
    
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.starmap(compute_perf_multiprocess, inputs)
    
    return
        
 









def run_evalutation(beta, result_path,lamda_new, simulation_time):

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

    ut_k = Utils(num_tasks_per_batch, mu, lamda, w_0, sigma_a, H0, H1, prior, d_0, beta, sigma_h, ctp, ctn, cfp, cfn, 0, num_bins_fatigue)

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

    mega_obs = [ut.get_auto_obs() for _ in range(simulation_time)]
    
    for t in tqdm(range(simulation_time)):

        fatigue_evolution_kesav.append(F_k)
        fatigue_evolution_adp.append(F_adp)
        fatigue_evolution_adp_new.append(F_adp_new)

        batched_obs, batched_posterior_h0, batched_posterior_h1= mega_obs[t]

        wl_k, deferred_idx_k = compute_kesavs_algo(batched_posterior_h0, batched_posterior_h1, F_k, ut_k)

        wl_adp, deferred_idx_adp = compute_adp_solution(batched_posterior_h0,batched_posterior_h1,F_adp, V_bar,ut)

        wl_adp_new, deferred_idx_adp_new = compute_adp_solution(batched_posterior_h0,batched_posterior_h1,F_adp_new, V_bar,ut_new)




        #get the next fatigue state for kesav
        F_k = ut_k.get_fatigue(F_k, wl_k)

        #get the next fatigue state for adp

        F_adp = ut.get_fatigue(F_adp,wl_adp)

        F_adp_new = ut_new.get_fatigue(F_adp_new,wl_adp_new)

        taskload_evolution_adp.append(wl_adp)
        taskload_evolution_adp_new.append(wl_adp_new)
        taskload_evolution_kesav.append(wl_k)
    
    return fatigue_evolution_kesav,fatigue_evolution_adp,fatigue_evolution_adp_new, taskload_evolution_kesav, taskload_evolution_adp, taskload_evolution_adp_new




def run_perf_eval(beta, result_path,lamda_new,simulation_time):
    
    fatigue_evolution_kesav,fatigue_evolution_adp,fatigue_evolution_adp_new, taskload_evolution_kesav, taskload_evolution_adp, taskload_evolution_adp_new = run_evalutation(beta,result_path,lamda_new,simulation_time)

        ## plotting the level of fatigue 

    path1 = result_path+'plot_analysis/'

    if not os.path.exists(path1):
        try:
            os.makedirs(path1,exist_ok=True)
        except FileExistsError:
            pass
    
    path2 = path1+'beta '+str(beta)+'/'

    if not os.path.exists(path2):
        try:
            os.makedirs(path2,exist_ok=True)
        
        except FileExistsError:
            pass
    

    np.save(path2+'fatigue_k.npy',fatigue_evolution_kesav)
    np.save(path2 +'fatigue_adp.npy',fatigue_evolution_adp)
    np.save(path2 +'fatigue_adp_new.npy',fatigue_evolution_adp_new)
    
    np.save(path2 +'taskload_k.npy',taskload_evolution_kesav)
    np.save(path2 +'taskload_adp.npy',taskload_evolution_adp)
    np.save(path2 +'taskload_adp_new.npy',taskload_evolution_adp_new)

    plt.plot(fatigue_evolution_adp,label='Approx dynamic program',color='orange')
    #plt.plot(fatigue_evolution_adp_new,label='Approx dynamic program - corr',color='red')
    plt.plot(fatigue_evolution_kesav,label='Kesav', color='black')

    plt.xlabel('Time')
    plt.ylabel('Fatigue Level')
    plt.legend()
    plt.savefig(path2+'beta_'+str(beta)+'_fatigue.pdf')

    plt.clf()
    plt.close()


    plt.step(np.arange(1,simulation_time+1,1),taskload_evolution_adp,label='Approx dynamic program',color='orange',where='post')
    #plt.step(np.arange(1,simulation_time+1,1),taskload_evolution_adp_new,label='Approx dynamic program-Corr',color='red',where='post')
    plt.step(np.arange(1,simulation_time+1,1),taskload_evolution_kesav,label='Kesav', color='black',where='post')

    plt.xlabel('Time')
    plt.ylabel('Workload level')
    plt.legend()
    plt.savefig(path2+'beta_'+str(beta)+'_workload.pdf')

    plt.clf()
    plt.close()

    return

   


def main():

  
    
    betas = [0.3,0.5,0.7,0.9]

    result_path = "results/"
    simulation_time = 20
    lamda_new=0.01

    inputs = [(beta,result_path,lamda_new,simulation_time) for beta in betas]
    
    # for beta in betas:
    #     run_perf_eval(beta, result_path,lamda_new,simulation_time)

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.starmap(run_perf_eval, inputs)

   
    
    compute_performance(betas, result_path,lamda_new,simulation_time)

        


if __name__=='__main__':

    main()

        


