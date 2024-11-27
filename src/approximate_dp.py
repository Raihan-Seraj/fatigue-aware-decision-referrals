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

    V_post = [{} for _ in range(T+1)]  # V_t^a(F_t^a)
    #V_post_kesav = [{} for _ in range(T+1)]

    # Terminal value
    for F_T in range(num_fatigue_states):
        
        expected_terminal_value = 0

        for _ in range(num_expectation_samples):
            Y_T_vec, batched_posterior_h0_T, batched_posterior_h1_T = ut.get_auto_obs() 

            best_action_value = float('inf')
            for w_t in range(ut.num_tasks_per_batch):

                cstar , _ , _ = ut.compute_cstar(F_T, w_t, batched_posterior_h0_T, batched_posterior_h1_T)

                best_action_value = min(best_action_value,cstar)

            expected_terminal_value += best_action_value/num_expectation_samples
            
        V_post[T][F_T] = expected_terminal_value

    ###############done #################################

    #performing backward iteration 

    for t in tqdm(range(T-1,-1,-1)):

        for F_t_a in range(num_fatigue_states):

          
        

            expected_value = 0
            
            for _ in range(num_expectation_samples):

                y_t, batched_posterior_h0_t, batched_posterior_h1_t = ut.get_auto_obs()

                
                best_action_value = float('inf')

               

                #w_t_k = ut.compute_kesav_policy(F_t_a, batched_posterior_h0_t, batched_posterior_h1_t)

             

                for w_t in range(ut.num_tasks_per_batch):

                    w_t_discrete = ut.discretize_taskload(w_t)

                    F_next  = env.next_state(F_t_a,w_t_discrete)

                    cstar, _, _ = ut.compute_cstar(F_t_a,w_t,batched_posterior_h0_t, batched_posterior_h1_t)
                    
                    V_expectations_epsilon = 0
                    for F_next in range(num_fatigue_states):

                        V_expectations_epsilon+= env.P[w_t_discrete][F_t_a,F_next] * V_post[t+1][F_next]

                    best_action_value = min(cstar + V_expectations_epsilon, best_action_value)
                
                

                
                
                expected_value += best_action_value/num_expectation_samples
                

                
                
            V_post[t][F_t_a]=expected_value

            #V_post_kesav[t][F_t_a] = expected_value_kesav


   
    
    return V_post





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

    # with open(path_name+'policy_func,pkl','wb') as file1:
    #     pickle.dump(policy,file1)

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




