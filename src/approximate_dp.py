import numpy as np 
from utils import Utils
from run_evaluation import Evaluations
from envs.fatigue_model_1 import FatigueMDP
from envs.fatigue_model_2 import FatigueMDP2
from tqdm import tqdm
import json
import os
import pickle
import argparse
import contextlib



'''
function that computes the approximate dynamic programming algorithm

Args:
    T: The total number of time to perform the dynamic program

    num_expectation_samples: The number of samples of automation observation vector to take for approximating the expectation --> input type integer

    ut: a utility object from the class Utility
'''


def approximate_dynamic_program(T, num_expectation_samples,ut,model_name):

    
    if model_name.lower()=='fatigue_model_1':
        env = FatigueMDP()

    elif model_name.lower()=='fatigue_model_2':
        env = FatigueMDP2()


    num_fatigue_states = env.num_fatigue_states
    
    # initializing the value of V_bar
    V_bar = {t: np.zeros((num_fatigue_states + 1)) for t in range(T + 2)}
    
    #V_bar_k = {t: np.zeros((num_fatigue_states + 1)) for t in range(T + 2)}

    
    
    # looping over time
    for t in tqdm(range(T, -1, -1)):

        # looping over all fatigue states
        for  F_t in range(num_fatigue_states):

            sum_y = 0
            #sum_y_k=0
            # number of expectation samples of Y
            
            for y in range(num_expectation_samples):
                
                #expectation_f = 0
                
                min_cost = float('inf')
                _, batched_posterior_h0, batched_posterior_h1 = ut.get_auto_obs()
                
                #w_t_k, deferred_k = ut.compute_kesav_policy(F_t,batched_posterior_h0,batched_posterior_h1)
                
                #w_t_k_discretized = ut.discretize_taskload(w_t_k)

                all_costs_per_w = []
                for w_t in range(ut.num_tasks_per_batch + 1):

                    ## computing the expectation
                    cstar, _, _ = ut.compute_cstar(
                        F_t,
                        w_t,
                        batched_posterior_h0,
                        batched_posterior_h1,
                    )


                    if model_name.lower()=='fatigue_model_1':
                        w_t_discrete = ut.discretize_taskload(w_t)

                        F_next_t = env.next_state(F_t,w_t_discrete)
                    elif model_name.lower()=='fatigue_model_2':

                        _, F_next_t = env.next_state(F_t,w_t)
                    else:
                        raise ValueError("Invalid fatigue model name")

                    total_cost = cstar + V_bar[t + 1][F_next_t] 

                    all_costs_per_w.append(total_cost)

                
                w_minimum_cost = np.argmin(all_costs_per_w)

                min_cost = min(all_costs_per_w)

               
          
                V_t = min_cost

                
                #cstar_k,_,_ = ut.compute_cstar(F_t, w_t_k, batched_posterior_h0,batched_posterior_h1)
                
                #total_cost_k = cstar_k
                
                #F_next_k = env.next_state(F_t,w_t_k_discretized)

               
                
                #V_t_k = total_cost_k + V_bar_k[t+1][F_next_k] 

                sum_y += V_t
                #sum_y_k+=V_t_k

            expected_value = sum_y / num_expectation_samples
            #expected_value_k = sum_y_k/ num_expectation_samples

            V_bar[t][F_t] = expected_value
            #V_bar_k[t][F_t]= expected_value_k

            


    return V_bar #V_bar_k





'''
Function that runs dynamic program for different values of beta
'''
def run_approximate_dynamic_program(args):

    
    
    num_tasks_per_batch = args.num_tasks_per_batch
   

    #error probability params for model_1
    alpha = args.alpha
    beta = args.beta
    gamma = args.gamma

    #ADP params

    num_expectation_samples = args.num_expectation_samples
    T = args.horizon

    

    
    ut = Utils(args)
    
    model_name = args.model_name

    # if model_name.lower()=='fatigue_model_1':

    #     env = FatigueMDP()
    # elif model_name.lower()=='fatigue_model_2':
    #     env = FatigueMDP2()
    # else:
    #     raise ValueError("Invalid fatigue model")
    
    
    

    path_name = args.results_path + 'num_tasks '+str(num_tasks_per_batch)+'/alpha '+str(alpha)+'/beta '+str(beta)+'/gamma '+str(gamma)+'/'

    if not os.path.exists(path_name):
        with contextlib.suppress(FileExistsError):
            os.makedirs(path_name, exist_ok=True)


    args_dict = vars(args)

    with open(path_name+'params.json','w') as json_file:
        json.dump(args_dict,json_file,indent=4)

    
    
    V_func  = approximate_dynamic_program(T, num_expectation_samples,ut,model_name
        
    )
    
   

    V_final = V_func[0]

   # V_final_k_pol = V_func_k_pol[0]

    with open(path_name + 'V_func.pkl','wb') as file:
        pickle.dump(V_func,file)
    np.save(path_name + "V_bar.npy", V_final)

    # with open(path_name + 'V_func_k_pol.pkl','wb') as file1:
    #     pickle.dump(V_func_k_pol,file1)
    # np.save(path_name + "V_bar_k_pol.npy", V_final_k_pol)
    
    

    return







'''
The main function 
'''
def main():

    
    parser = argparse.ArgumentParser(description="Approximate Dynamic Program parameters.")

    parser.add_argument('--alpha', type=float, default= 0.1, help='The influence of fatigue on false positive probability')
    parser.add_argument('--beta',type=float, default=0.1, help='The influence of taskload on false positive probability' )
    parser.add_argument('--gamma',type=float, default=0.1, help='The exponent of the true positive probability model' )

    parser.add_argument('--num_expectation_samples', type=int, default=500, help='Number of expectation samples to take for the approximate Dynamic Program.')
    parser.add_argument('--horizon', type=int, default=20, help='The length of the horizon.')
    parser.add_argument('--d_0',type=float, default= 3, help='The value of d0 in the experiment.')
    parser.add_argument('--prior',default=[0.8,0.2], nargs=2, type=float, help='A list containing the prior of [H0, H1].' )
    parser.add_argument('--num_tasks_per_batch', type=int, default=20, help='The total number of tasks in a batch.')
    parser.add_argument('--sigma_a',type=float, default=2, help='Automation observation channel variance.')
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
    parser.add_argument('--num_eval_runs', type=int, default=500, help="Number of independent runs for monte carlo performance evaluation")
    parser.add_argument('--model_name', type=str,default='fatigue_model_1',  help='The fatigue model to choose options are [fatigue_model_1, fatigue_model_2]')
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




