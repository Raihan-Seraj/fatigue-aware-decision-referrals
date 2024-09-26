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
    F_states = np.round(np.linspace(0, 20, ut.num_bins_fatigue + 1), 2)

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
def run_dp_parallel_beta(args, H0, H1):

    num_tasks_per_batch = args.num_tasks_per_batch
    alpha = args.alpha
    gamma = args.gamma
    sigma_a = args.sigma_a
    prior = args.prior
    d_0 = args.d_0
    beta = args.beta
    sigma_h = args.sigma_h
    ctp = args.ctp
    ctn = args.ctn
    cfp = args.cfp
    cfn = args.cfn
    cm = args.cm
    num_bins_fatigue=args.num_bins_fatigue
    num_expectation_samples = args.num_expectation_samples
    T = args.horizon

    
    ut = Utils(args, H0, H1)

    if args.use_wandb:
        run_info = wandb.init(project="Example 3",name="beta "+str(beta)+'alpha '+str(alpha)+' gamma '+str(gamma),settings=wandb.Settings(start_method="fork"), mode=args.wandb_sync)
    

    param_values = {
       "num_tasks_per_batch": num_tasks_per_batch, 
        
        "H0": H0,

        "H1": H1, 

        "prior": prior, 

        "d_0": d_0,

        "beta": beta,

        "alpha": alpha,  

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

    if args.use_wandb:
        run_info.config.update(param_values)


    path_name = args.results_path + 'num_tasks '+str(num_tasks_per_batch)+'/beta '+str(beta)+'/alpha '+str(alpha)+'/gamma_'+str(gamma)+'/'

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
    
    if args.use_wandb:
        run_info.finish()


    return







'''
The main function 
'''
def main():

    
    parser = argparse.ArgumentParser(description="Approximate Dynamic Program parameters.")

    parser.add_argument('--beta', type=float, default= 0.5, help='Exponent that influcences the extent to which workload affects the observation channel')
    parser.add_argument('--alpha', type=float, default= 0.5, help='Exponent that influcences the extent to which fatigue affects the observation channel')
    
    parser.add_argument('--num_expectation_samples', type=int, default=10, help='Number of expectation samples to take for the approximate Dynamic Program.')
    parser.add_argument('--horizon', type=int, default=20, help='The length of the horizon.')
    parser.add_argument('--d_0',type=float, default= 5, help='The value of d0 in the experiment.')
    parser.add_argument('--prior',default=[0.8,0.2], nargs=2, type=float, help='A list containing the prior of [H0, H1].' )
    parser.add_argument('--num_tasks_per_batch', type=int, default=20, help='The total number of tasks in a batch.')
    parser.add_argument('--sigma_a',type=float, default=2.5, help='Automation observation channel variance.')
    parser.add_argument('--sigma_h', type=float, default=1.0, help='Human observation channel variance.')
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
    parser.add_argument('--Fmax', type=int, default=20,help='Max value for the fatigue level')
    args = parser.parse_args()


    H0=0
    H1= args.d_0

    eval_flag = args.run_eval_only

    
    
    if not eval_flag:
   
        run_dp_parallel_beta(args, H0, H1)
    
    

    #Running evaluation 
    EVAL = Evaluations(args)

    



    print("Running evaluation for the computed value function ")

 
    EVAL.run_perf_eval()

    
    
    ## Compute the performance now 

    print("Computing the performance")

    EVAL.compute_performance()
    






if __name__ == "__main__":
    main()




