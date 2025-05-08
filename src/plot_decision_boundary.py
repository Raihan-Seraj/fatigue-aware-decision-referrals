import numpy as np 
import matplotlib.pyplot as plt
import os 
from utils import Utils
import matplotlib
import argparse
from envs.fatigue_model_1 import FatigueMDP
from envs.fatigue_model_3 import FatigueMDP3
matplotlib.use('Agg')

## get the parameter for any value of beta since we are using the same parameters for all value of beta

def plot_decision_boundary(args):
    #plt.figure(figsize=(110, 110)) 
    

   

    ctp = args.ctp
    ctn = args.ctn
    cfp = args.cfp
    cfn = args.cfn

    if args.fatigue_model.lower()=='fatigue_model_1':

        env = FatigueMDP()

        F_states = env.fatigue_states
    
    elif args.fatigue_model.lower()=='fatigue_model_3':

        env = FatigueMDP3()
        F_states = env.fatigue_states
   


    ut = Utils(args)


     # compute the auto_cost for different posterior of the tasks

    _, batched_posteriors_h0, batched_posterior_h1 = ut.get_auto_obs()
    posteriors_h1 = np.sort(batched_posterior_h1)

    auto_costs= [min(
        ctn + (cfn - ctn) * automation_posterior_h1,
        cfp + (ctp - cfp) * automation_posterior_h1,
    ) for automation_posterior_h1 in posteriors_h1]
    
    plt.plot(posteriors_h1,auto_costs,'*--',label='automation cost',color='red',linewidth=2)
    
    colors = ['blue','green','orange','magenta','cyan','teal','purple','brown','black']
    
    automation_posteriors = [[1-post_h1,post_h1] for post_h1 in posteriors_h1]
    Ftwt = []
    

    # we are considering taskload form {0,..,20}
    
    test_counter = 0
    for w_t in range(20):
        color_counter=0
        
    #for w_t in all_workloads:
        #computing the human cost for each taskload
       
        for F_t_idx, F_t in enumerate(F_states):
            
            #mport ipdb;ipdb.set_trace()
       
            hum_costs=[ut.compute_gamma(automation_posterior,F_t,w_t) for automation_posterior in automation_posteriors]

            if args.model_name.lower()=='fatigue_model_1':

                #w_t_d = ut.discretize_taskload(w_t)
                #print(w_t_d)
                w_t_d = w_t

                #if (F_t,w_t_d) not in Ftwt:
                
                plt.plot(posteriors_h1,hum_costs, label='Human (F_t,w_t)=('+str(F_t)+','+str(w_t_d)+')',color='blue')
                color_counter+=1
                test_counter+=1
                #print(test_counter)
                Ftwt.append((F_t,w_t_d))
            elif args.model_name.lower()=='fatigue_model_3':

                plt.plot(posteriors_h1,hum_costs, label='Human (F_t,w_t)=('+str(F_t)+','+str(w_t)+')')
            
            else:
                raise ValueError("Invalid fatigue model name")

           

    
                


    plt.grid(True)  # Show grid
    plt.legend()
    plt.tick_params(axis='both', which='major') 
    plt.xlabel('Posterior_H1')
    plt.ylabel('cost')
    plt.xlim(left=0)
    
    

    path1 = 'decision_boundary/'

    if not os.path.exists(path1):

        os.makedirs(path1)
    
    path2 = path1+'beta '+str(args.beta)+'/' +'alpha/'+str(args.alpha)+'/'

    if not os.path.exists(path2):
        os.makedirs(path2)
    

    plt.savefig(path2+'decision_boundary.pdf')

    plt.clf()
    plt.close()

    


    


if __name__=='__main__':

    parser = argparse.ArgumentParser(description="Approximate Dynamic Program parameters.")

    parser.add_argument('--alpha', type=float, default= 0.1, help='The influence of fatigue on false positive probability')
    parser.add_argument('--beta',type=float, default=0.1, help='The influence of taskload on false positive probability' )
    parser.add_argument('--gamma',type=float, default=0.1, help='The exponent of the true positive probability model' )

    parser.add_argument('--num_expectation_samples', type=int, default=500, help='Number of expectation samples to take for the approximate Dynamic Program.')
    parser.add_argument('--horizon', type=int, default=15, help='The length of the horizon.')
    parser.add_argument('--d_0',type=float, default= 3, help='The value of d0 in the experiment.')
    parser.add_argument('--prior',default=[0.8,0.2], nargs=2, type=float, help='A list containing the prior of [H0, H1].' )
    parser.add_argument('--num_tasks_per_batch', type=int, default=20, help='The total number of tasks in a batch.')
    parser.add_argument('--sigma_a',type=float, default=2, help='Automation observation channel variance.')
    
    #hypothesis value 
    parser.add_argument('--H0', type=int, default=0, help='The value of the null hypothesis')
    parser.add_argument('--H1',type=int, default=3,help='The value of alternate hypothesis'  )
    
    # Arguments associated with costs
    parser.add_argument('--ctp',type=float, default=0, help='The cost associated with true positive rate')
    parser.add_argument('--ctn',type=float, default=0, help='The cost associated with true negative value.')
    parser.add_argument('--cfp', type=float, default=1.0, help='The cost associated with false positive rates')
    parser.add_argument('--cfn', type=float, default=1.0, help='The cost associated with false negative rates.')
    parser.add_argument('--cm', type=float, default=0.0, help='The cost associated with deferrals.')

    parser.add_argument('--fatigue_model', type=str,default='fatigue_model_1',  help='The fatigue model to choose options are [fatigue_model_1, fatigue_model_2]')
    parser.add_argument('--results_path', type=str, default='results/', help='The name of the directory to save the results')
    parser.add_argument('--model_name', type=str,default='fatigue_model_1',  help='The fatigue model to choose options are [fatigue_model_1, fatigue_model_2]')
    #parser.add_argument('--plot_posterior', type=bool, default=False, help='Flag whether to plot the posteriors or not')
    args = parser.parse_args()

    plot_decision_boundary(args)




       

