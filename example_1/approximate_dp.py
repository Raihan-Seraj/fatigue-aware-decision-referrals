import numpy as np 

from utils import Utils
from run_evaluation import Evaluations
from tqdm import tqdm
import json
import os
import multiprocessing
import pickle
import wandb
import pandas as pd
import argparse
wandb.require("legacy-service")

'''
function that computes the approximate dynamic programming algorithm

Args:
    T: The total number of time to perform the dynamic program

    num_expectation_samples: The number of samples of automation observation vector to take for approximating the expectation --> input type integer

    ut: a utility object from the class Utility
'''


def approximate_dynamic_program(T, num_expectation_samples,ut):

    

    # number of possible fatigue states
    F_states = np.round(np.linspace(0, 1, ut.num_bins_fatigue + 1), 2)

    # initializing the value of V_bar
    V_bar = {t: np.zeros((ut.num_bins_fatigue + 1)) for t in range(T + 2)}
    
    V_bar_k = {t: np.zeros((ut.num_bins_fatigue + 1)) for t in range(T + 2)}

    
    
    # looping over time
    for t in tqdm(range(T, -1, -1)):

        # looping over all fatigue states
        for F_idx, F_t in enumerate(F_states):

            sum_y = 0
            sum_y_k=0
            # number of expectation samples of Y
            
            for y in range(num_expectation_samples):
                
                min_cost = float('inf')
                batched_obs, batched_posterior_h0, batched_posterior_h1 = ut.get_auto_obs()
                w_t_k, deferred_k = ut.compute_kesav_policy(F_t,batched_posterior_h0,batched_posterior_h1)
                for w_t in range(ut.num_tasks_per_batch + 1):

                    ## computing the expectation

                    F_next = ut.get_fatigue(F_t, w_t)

                    F_next_idx = ut.discretize_fatigue_state(F_next)

                    

                    cstar, deferred_indices, _ = ut.compute_cstar(
                        F_t,
                        w_t,
                        batched_posterior_h0,
                        batched_posterior_h1,
                    )

                    auto_cost, hum_cost, deff_cost = ut.per_step_cost(F_t, batched_posterior_h1,deferred_indices)
                    
                    per_step_cost = auto_cost + hum_cost + deff_cost
                    total_cost = per_step_cost + V_bar[t + 1][F_next_idx]

                    

                    if total_cost < min_cost:
                        
                        min_cost = total_cost

                V_t = min_cost

                auto_cost_k,human_cost_k,deferred_cost_k = ut.per_step_cost(F_t,batched_posterior_h1,deferred_k)

                total_cost_k = auto_cost_k + human_cost_k + deferred_cost_k 
                
                F_next_k = ut.get_fatigue(F_t,w_t_k)

                F_next_idx_k = ut.discretize_fatigue_state(F_next_k)
                
                V_t_k = total_cost_k + V_bar_k[t+1][F_next_idx_k] 

                sum_y += V_t
                sum_y_k+=V_t_k

            expected_value = sum_y / num_expectation_samples
            expected_value_k = sum_y_k/ num_expectation_samples

            V_bar[t][F_idx] = expected_value
            V_bar_k[t][F_idx]= expected_value_k

            


    return V_bar, V_bar_k





'''
Function that runs dynamic program for different values of beta
'''
def run_dp_parallel_beta(num_tasks_per_batch, mu, lamda, w_0, sigma_a, H0, H1, prior, d_0, beta, sigma_h, ctp, ctn, cfp, cfn, cm, num_bins_fatigue, T, num_expectation_samples,result_path):
    
    ut = Utils(num_tasks_per_batch, mu, lamda, w_0, sigma_a, H0, H1, prior, d_0, beta, sigma_h, ctp, ctn, cfp, cfn, cm, num_bins_fatigue)

    run_info = wandb.init(project="Example 1",name="beta "+str(beta)+' mu '+str(mu)+' lambda '+str(lamda),settings=wandb.Settings(start_method="fork"))
    

    param_values = {
       "num_tasks_per_batch": num_tasks_per_batch, 

       "mu": mu, 

        "lamda": lamda,

        "w_0": w_0,
        
        "H0": H0,

        "H1": H1, 

        "prior": prior, 

        "d_0": d_0,

        "beta": beta, 

        "sigma_h": sigma_h, 

        "sigma_a": sigma_a, 

        "ctp": ctp, 
        
        "ctn": ctn,

        "cfp": cfp, 

        "cfn": cfn, 

        "cm": cm,

        "num_bins_fatigue": num_bins_fatigue, 

        "T": T, 

        "num_expectation_samples": num_expectation_samples  
    

    }

    run_info.config.update(param_values)


    path_name = result_path + 'num_tasks '+str(num_tasks_per_batch)+'/beta '+str(beta)+'/mu_'+str(mu)+'_lambda_'+str(lamda)+'/'

    if not os.path.exists(path_name):
        try:
            os.makedirs(path_name, exist_ok=True)
        except FileExistsError:
            pass


    
    with open(path_name + "params.json", "w") as json_file:
        json.dump(param_values, json_file, indent=4)

    V_func, V_func_k_pol = approximate_dynamic_program(T, num_expectation_samples,ut
        
    )
    
   

    V_final = V_func[0]

    V_final_k_pol = V_func_k_pol[0]

    with open(path_name + 'V_func.pkl','wb') as file:
        pickle.dump(V_func,file)
    np.save(path_name + "V_bar.npy", V_final)

    with open(path_name + 'V_func_k_pol.pkl','wb') as file1:
        pickle.dump(V_func_k_pol,file1)
    np.save(path_name + "V_bar_k_pol.npy", V_final_k_pol)

    run_info.finish()


    return







'''
The main function 
'''
def main():

    parser = argparse.ArgumentParser(description="Approximate Dynamic Program parameters")

    parser.add_argument('--beta', type=int, default= 2, help='The effect of fatigue on the human observation channel')
    parser.add_argument('--mu', type=float, default=0.05,  help='Decay rate of fatigue')
    parser.add_argument('--lamda', type=float, default=0.07,  help='Growth rate of fatigue')
    parser.add_argument('--num_expectation_samples', type=int, default=10, help='Number of expectation samples to take for the approximate Dynamic Program')
    parser.add_argument('--horizon', type=int, default=20, help='The length of the horizon')

    args = parser.parse_args()

    
    beta= args.beta

    # defining the value of d_0
    d_0 = 5
    # defining the prior distribution of H_0 and H_1 respectively
    prior = [0.8, 0.2]

    # defining the value of H_0 and H_1
    H0 = 0
    H1 = d_0

    # the number of tasks per batch
    num_tasks_per_batch=20
    # parameters used
    sigma_a = 2.5
    sigma_h = 1.0

    # total time for which the system will run 
    T = args.horizon

    # The threshold value 
    w_0 = 15

    # number of bins used for the discretization of fatigue 
    num_bins_fatigue = 10
    num_expectation_samples = args.num_expectation_samples

    cfp = 1#np.random.uniform(8, 12)
    cfn = 1#np.random.uniform(8, 12)
    ctp = 0#np.random.uniform(0, 2)
    ctn = 0#np.random.uniform(0, 2)
    cm = 0

    # fatigue recovery rate
    mu = args.mu

    # fatigue growth rate
    lamda = args.lamda

    result_path = "results/"

    
    
   
    run_dp_parallel_beta(num_tasks_per_batch, mu, lamda, w_0, sigma_a, H0, H1, prior, d_0, beta, sigma_h, ctp, ctn, cfp, cfn, cm, num_bins_fatigue, T, num_expectation_samples,result_path)
    
    # with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
    
    #     pool.starmap(run_dp_parallel_beta, inputs)
    

    #Running evaluation 
    EVAL = Evaluations(args)

    lamda_new = 0.01



    print("Running evaluation for the computed value function ")

 
    EVAL.run_perf_eval(beta, result_path, lamda_new, T, num_tasks_per_batch)

    # with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
    #     pool.starmap(EVAL.run_perf_eval, inputs_eval)

    
    ## Compute the performance now 

    print("Computing the performance")

    EVAL.compute_performance(beta, result_path, lamda_new, T, num_tasks_per_batch)
    






if __name__ == "__main__":
    main()




