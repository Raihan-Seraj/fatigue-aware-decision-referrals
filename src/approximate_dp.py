import numpy as np 
from utils import Utils
from run_evaluation import Evaluations
from tqdm import tqdm
import json
import os
import pickle
import argparse



'''
function that computes the approximate dynamic programming algorithm

Args:
    T: The total number of time to perform the dynamic program

    num_expectation_samples: The number of samples of automation observation vector to take for approximating the expectation --> input type integer

    ut: a utility object from the class Utility
'''


def approximate_dynamic_program(T, num_expectation_samples,ut,fatigue_model='model_1'):

    
    if fatigue_model.lower()=='model_1':
    # number of possible fatigue states
        F_states = np.round(np.linspace(0, 1, ut.num_bins_fatigue + 1), 2)
    else:
        raise ValueError("Invalid fatigue model")

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
def run_approximate_dynamic_program(args):

    fatigue_model = args.fatigue_model
    
    num_tasks_per_batch = args.num_tasks_per_batch
    # params for the fatigue model 
    if fatigue_model.lower()=='model_1':
        mu = args.mu
        lamda = args.lamda
        w_0 = args.w_0
    
    else:
        raise ValueError("Incorrect fatigue model")

    # auto observation
    sigma_a = args.sigma_a
    d_0 = args.d_0
    
    prior = args.prior
    

    #error probability params for model_1
    alpha = args.alpha
    beta = args.beta
    gamma = args.gamma

    # cost
    ctp = args.ctp
    ctn = args.ctn
    cfp = args.cfp
    cfn = args.cfn
    cm = args.cm

    #ADP params
    num_bins_fatigue=args.num_bins_fatigue
    num_expectation_samples = args.num_expectation_samples
    T = args.horizon

    

    
    ut = Utils(args)
    
    
    
    
    

    path_name = args.results_path + 'num_tasks '+str(num_tasks_per_batch)+'/alpha '+str(alpha)+'/beta '+str(beta)+'/gamma '+str(gamma)+'/mu_'+str(mu)+'_lambda_'+str(lamda)+'/'

    if not os.path.exists(path_name):
        try:
            os.makedirs(path_name, exist_ok=True)
        except FileExistsError:
            pass


    args_dict = vars(args)

    with open(path_name+'params.json','w') as json_file:
        json.dump(args_dict,json_file,indent=4)

    
    
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
    
    

    return







'''
The main function 
'''
def main():

    
    parser = argparse.ArgumentParser(description="Approximate Dynamic Program parameters.")

    parser.add_argument('--alpha', type=float, default= 1, help='The influence of fatigue on false positive probability')
    parser.add_argument('--beta',type=float, default=0.05, help='The influence of taskload on false positive probability' )
    parser.add_argument('--gamma',type=float, default=0.1, help='The exponent of the true positive probability model' )

    parser.add_argument('--mu', type=float, default=0.05,  help='Decay rate of fatigue.')
    parser.add_argument('--lamda', type=float, default=0.07,  help='Growth rate of fatigue.')
    parser.add_argument('--num_expectation_samples', type=int, default=500, help='Number of expectation samples to take for the approximate Dynamic Program.')
    parser.add_argument('--horizon', type=int, default=15, help='The length of the horizon.')
    parser.add_argument('--d_0',type=float, default= 3, help='The value of d0 in the experiment.')
    parser.add_argument('--prior',default=[0.8,0.2], nargs=2, type=float, help='A list containing the prior of [H0, H1].' )
    parser.add_argument('--num_tasks_per_batch', type=int, default=20, help='The total number of tasks in a batch.')
    parser.add_argument('--sigma_a',type=float, default=2.5, help='Automation observation channel variance.')
    parser.add_argument('--w_0', type=int, default=15, help='The workload threshold beyond which fatigue recovery does not occur')
    parser.add_argument('--num_bins_fatigue', type=int, default=10, help='The number of bins to be used to discretize fatigue')

    #hypothesis value 
    parser.add_argument('--H0', type=int, default=0, help='The value of the null hypothesis')
    parser.add_argument('--H1',type=int, default=3,help='The value of alternate hypothesis'  )
    
    # Arguments associated with costs
    parser.add_argument('--ctp',type=float, default=0, help='The cost associated with true positive rate')
    parser.add_argument('--ctn',type=float, default=0, help='The cost associated with true negative value.')
    parser.add_argument('--cfp', type=float, default=1.0, help='The cost associated with false positive rates')
    parser.add_argument('--cfn', type=float, default=1.0, help='The cost associated with false negative rates.')
    parser.add_argument('--cm', type=float, default=0.0, help='The cost associated with deferrals.')

    parser.add_argument('--fatigue_model', type=str, default='model_1',help='The fatigue model to use. Choices are ["model_1", "model_2"]')
    parser.add_argument('--results_path', type=str, default='results/', help='The name of the directory to save the results')

    parser.add_argument('--run_eval_only', type=bool, default=False)
    parser.add_argument('--num_eval_runs', type=int, default=10, help="Number of independent runs for monte carlo performance evaluation")

    args = parser.parse_args()

    



    eval_flag = args.run_eval_only

    
    
    if not eval_flag:
   
        run_approximate_dynamic_program(args)
    
    

    #Running evaluation 
    EVAL = Evaluations(args)

    



    print("Running evaluation for the computed value function ")

 
    EVAL.run_perf_eval()

    
    
    ## Compute the performance now 

    print("Computing the performance")

    EVAL.compute_performance()
    






if __name__ == "__main__":
    main()




