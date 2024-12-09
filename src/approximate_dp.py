import numpy as np 
from utils import Utils
from run_evaluation import Evaluations
from envs.fatigue_model_1 import FatigueMDP
from envs.fatigue_model_2 import FatigueMDP2
from envs.fatigue_model_3 import FatigueMDP3
from tqdm import tqdm
import json
import os
import pickle
import argparse
import contextlib



'''
function that computes the approximate dynamic programming algorithm

Args:
    T (int): The total time horizon in integer

    num_expectation_samples (int): The number of expectation samples of use to approximate the expectation with respect fo Y^a_{t+1}

    ut: a utility object from the class Utility
'''


def approximate_dynamic_program(T, num_expectation_samples,ut,model_name):

    
    if model_name.lower()=='fatigue_model_1':
        env = FatigueMDP()
        
        fatigue_states = env.fatigue_states

    elif model_name.lower()=='fatigue_model_3':
        env = FatigueMDP3()
        
        fatigue_states = env.fatigue_states
    else:
        raise ValueError("Invalid Fatigue Model Selected")


    

    V_post = [{} for _ in range(T+1)]  # V_t^a(F_t^a)
    #V_post_kesav = [{} for _ in range(T+1)]

    # Terminal value
    for idx_F_T, F_T in enumerate(fatigue_states):
        
        expected_terminal_value = 0

        for _ in range(num_expectation_samples):
            Y_T_vec, batched_posterior_h0_T, batched_posterior_h1_T = ut.get_auto_obs() 

            best_action_value = float('inf')
            for w_t in range(ut.num_tasks_per_batch):

                cstar , _ , _ = ut.compute_cstar(F_T, w_t, batched_posterior_h0_T, batched_posterior_h1_T)

                best_action_value = min(best_action_value,cstar)

            expected_terminal_value += best_action_value/num_expectation_samples
            
        V_post[T][idx_F_T] = expected_terminal_value

    ###############done #################################

    #performing backward iteration 

    for t in tqdm(range(T-1,-1,-1)):

        for idx_F_t, F_t in enumerate(fatigue_states):

            expected_value = 0
            
            for _ in range(num_expectation_samples):

                y_t, batched_posterior_h0_t, batched_posterior_h1_t = ut.get_auto_obs()

                
                best_action_value = float('inf')

               

                #w_t_k = ut.compute_kesav_policy(F_t_a, batched_posterior_h0_t, batched_posterior_h1_t)

             

                for w_t in range(ut.num_tasks_per_batch):

                   

                    # F_next  = env.next_state(F_t,w_t_discrete)

                    cstar, _, _ = ut.compute_cstar(F_t,w_t,batched_posterior_h0_t, batched_posterior_h1_t)
                    
                    V_expectations_epsilon = 0

                    
                    if model_name.lower()=='fatigue_model_1':

                        w_t_discrete = ut.discretize_taskload(w_t)
                
                        for idx_F_next, F_next in enumerate(fatigue_states):
                                
                            # perform expectation with respect to the next state since fatigue is stochastic
                            V_expectations_epsilon+= env.P[w_t_discrete][idx_F_t,idx_F_next] * V_post[t+1][idx_F_next]
                    
                    elif model_name.lower()=='fatigue_model_3':

                        F_next, idx_F_next = env.next_state(F_t, w_t)
                        V_expectations_epsilon = V_post[t+1][idx_F_next]
                    
                    else:
                        raise ValueError("Invalid Fatigue Model Selected")
                            
                       
                            
                        
                        
                        

                    best_action_value = min(cstar + V_expectations_epsilon, best_action_value)
                
                

                
                
                expected_value += best_action_value/num_expectation_samples
                

                
                
            V_post[t][idx_F_t]=expected_value

            #V_post_kesav[t][F_t_a] = expected_value_kesav


   
    
    return V_post





'''
Function that runs dynamic program for different values of beta
'''
def run_approximate_dynamic_program(args):

    
    
    num_tasks_per_batch = args.num_tasks_per_batch
   
    model_name = args.fatigue_model

    if model_name.lower()=='fatigue_model_1':
        #error probability params for model_1
        alpha_tp = args.alpha_tp
        alpha_fp = args.alpha_fp
        beta_tp = args.beta_tp
        beta_fp = args.beta_fp
        gamma_tp = args.gamma_tp
        gamma_fp = args.gamma_fp


        path_name = args.results_path + 'fatigue_model_1/num_tasks '+str(num_tasks_per_batch)+'/alpha_tp '+str(alpha_tp)+'/beta_tp '+str(beta_tp)+'/gamma_tp '+str(gamma_tp)+'/' \
        +'alpha_fp '+str(alpha_fp) +'/beta_fp ' +str(beta_fp) + '/gamma_fp '+str(gamma_fp) + '/'
    

    elif model_name.lower()=='fatigue_model_3':

        path_name = args.results_path + 'fatigue_model_3/num_tasks '+str(num_tasks_per_batch)+'/'
    
    else:
        raise ValueError("Invalid Fatigue Model Selected")

    #ADP params

    num_expectation_samples = args.num_expectation_samples
    T = args.horizon

    

    
    ut = Utils(args)
    
    if not os.path.exists(path_name):
        with contextlib.suppress(FileExistsError):
            os.makedirs(path_name, exist_ok=True)


    args_dict = vars(args)

    with open(path_name+'params.json','w') as json_file:
        json.dump(args_dict,json_file,indent=4)

    
    
    V_func  = approximate_dynamic_program(T, num_expectation_samples,ut,model_name
        
    )
    
   

    V_final = V_func[0]


    with open(path_name + 'V_func.pkl','wb') as file:
        pickle.dump(V_func,file)
    np.save(path_name + "V_bar.npy", V_final)

   
    
    

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

    parser.add_argument('--fatigue_model', type=str, default='fatigue_model_1',help='The fatigue model to use. Choices are ["fatigue_model_1", "fatigue_model_3"]')
    parser.add_argument('--results_path', type=str, default='results/', help='The name of the directory to save the results')

    parser.add_argument('--run_eval_only', type=bool, default=False)
    parser.add_argument('--num_eval_runs', type=int, default=500, help="Number of independent runs for monte carlo performance evaluation")
    # parser.add_argument('--fatigue_model', type=str,default='fatigue_model_1',  help='The fatigue model to choose options are [fatigue_model_1, fatigue_model_2]')
    parser.add_argument('--alpha_tp', type=float, default=0.2, help='The value of alpha for Ptp in range [0,1]')
    parser.add_argument('--beta_tp',type=float, default=0.1, help='Value of beta for Ptp in range [0,1]')
    parser.add_argument('--gamma_tp', type=float, default=2.5, help='normalizing term for Ptp should be positive real number')

    parser.add_argument('--alpha_fp', type=float, default=0.3, help='The value of alpha for Pfp in range [0,1]')
    parser.add_argument('--beta_fp',type=float, default=0.1, help='Value of beta for Pfp in range [0,1]')
    parser.add_argument('--gamma_fp', type=float, default=3, help='normalizing term for Pfp should be positive real number')
    
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




