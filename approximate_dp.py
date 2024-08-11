import numpy as np 

from utils import Utils
from tqdm import tqdm
import json
import os
import multiprocessing


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

    # looping over time
    for t in tqdm(range(T, -1, -1)):

        # looping over all fatigue states
        for F_idx, F_t in enumerate(F_states):

            sum_y = 0

            # number of expectation samples of Y
            for y in range(num_expectation_samples):

                all_cost = []
                for w_t in range(ut.num_tasks_per_batch + 1):

                    ## computing the expectation

                    F_next = ut.get_fatigue(F_t, w_t)

                    F_next_idx = ut.discretize_fatigue_state(F_next)

                    batched_obs, batched_posterior_h0, batched_posterior_h1 = (
                        ut.get_auto_obs()
                    )

                    cstar, _, _ = ut.compute_cstar(
                        F_t,
                        w_t,
                        batched_posterior_h0,
                        batched_posterior_h1,
                    )

                    total_cost = cstar + V_bar[t + 1][F_next_idx]

                    all_cost.append(total_cost)

                sum_y += min(all_cost)

            expected_value = sum_y / num_expectation_samples

            V_bar[t][F_idx] = expected_value

    return V_bar



'''
Function that runs dynamic program for different values of beta
'''
def run_dp_parallel_beta(num_tasks_per_batch, mu, lamda, w_0, sigma_a, H0, H1, prior, d_0, beta, sigma_h, ctp, ctn, cfp, cfn, cm, num_bins_fatigue, T, num_expectation_samples):
    
    ut = Utils(num_tasks_per_batch, mu, lamda, w_0, sigma_a, H0, H1, prior, d_0, beta, sigma_h, ctp, ctn, cfp, cfn, cm, num_bins_fatigue)


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

    path1 = "results/"

    if not os.path.exists(path1):
        os.makedirs(path1)

    path2 = path1 + "num_tasks " + str(num_tasks_per_batch)

    if not os.path.exists(path2):
        os.makedirs(path2)

    path3 = path2 + "/beta " + str(beta) + "/"

    if not os.path.exists(path3):
        os.makedirs(path3)

    path_name = path3

    with open(path_name + "params.json", "w") as json_file:
        json.dump(param_values, json_file, indent=4)

    V_func = approximate_dynamic_program(T, num_expectation_samples,ut
        
    )

    V_final = V_func[0]
    np.save(path_name + "V_bar.npy", V_final)







'''
The main function 
'''
def main():

    # defining the values of beta
    betas = np.round(np.linspace(0,1,21),2)

     # defining the value of d_0
    d_0 = 3
    # defining the prior distribution of H_0 and H_1 respectively
    prior = [0.8, 0.2]

    # defining the value of H_0 and H_1
    H0 = 0
    H1 = 1

    # the number of tasks per batch
    num_tasks_per_batch=20
    # parameters used
    sigma_a = np.random.uniform(1.5, 2)
    sigma_h = np.random.uniform(1, 1.5)

    # total time for which the system will run 
    T = 20

    # The threshold value 
    w_0 = 10

    # number of bins used for the discretization of fatigue 
    num_bins_fatigue = 10
    num_expectation_samples = 15

    cfp = np.random.uniform(8, 12)
    cfn = np.random.uniform(8, 12)
    ctp = np.random.uniform(0, 2)
    ctn = np.random.uniform(0, 2)
    cm = np.random.uniform(0, 0.5)

    # fatigue recovery rate
    mu = 0.03

    # fatigue growth rate
    lamda = 0.05

    inputs = [
        (
            num_tasks_per_batch, mu, lamda, w_0, sigma_a, H0, H1, prior, d_0, beta, sigma_h, ctp, ctn, cfp, cfn, cm, num_bins_fatigue, T, num_expectation_samples
        )
        for beta in betas
    ]

    # for inpt in inputs:
        
    #     run_dp_parallel_beta(num_tasks_per_batch, mu, lamda, w_0, sigma_a, H0, H1, prior, d_0, 0.5, sigma_h, ctp, ctn, cfp, cfn, cm, num_bins_fatigue, T, num_expectation_samples)

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.starmap(run_dp_parallel_beta, inputs)


if __name__ == "__main__":
    main()




