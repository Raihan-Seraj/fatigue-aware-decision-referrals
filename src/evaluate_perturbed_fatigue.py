from envs.fatigue_model_1 import FatigueMDP
from envs.fatigue_model_2 import FatigueMDP2
from envs.fatigue_model_1_perturbed import FatigueMDPPerturbed
from utils import Utils
from run_evaluation import Evaluations
from tqdm import tqdm
import json
import os
import pickle
import argparse
import contextlib


parser = argparse.ArgumentParser(description="Approximate Dynamic Program parameters.")

parser.add_argument('--alpha', type=float, default= 0.1, help='The influence of fatigue on false positive probability')
parser.add_argument('--beta',type=float, default=0.1, help='The influence of taskload on false positive probability' )
parser.add_argument('--gamma',type=float, default=0.1, help='The exponent of the true positive probability model' )

parser.add_argument('--num_expectation_samples', type=int, default=500, help='Number of expectation samples to take for the approximate Dynamic Program.')
parser.add_argument('--horizon', type=int, default=10, help='The length of the horizon.')
parser.add_argument('--d_0',type=float, default= 3, help='The value of d0 in the experiment.')
parser.add_argument('--prior',default=[0.5,0.5], nargs=2, type=float, help='A list containing the prior of [H0, H1].' )
parser.add_argument('--num_tasks_per_batch', type=int, default=20, help='The total number of tasks in a batch.')
parser.add_argument('--sigma_a',type=float, default=2.3, help='Automation observation channel variance.')
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
parser.add_argument('--num_eval_runs', type=int, default=1000, help="Number of independent runs for monte carlo performance evaluation")
# parser.add_argument('--fatigue_model', type=str,default='fatigue_model_1',  help='The fatigue model to choose options are [fatigue_model_1, fatigue_model_2]')
parser.add_argument('--alpha_tp', type=float, default=0.2, help='The value of alpha for Ptp in range [0,1]')
parser.add_argument('--beta_tp',type=float, default=0.1, help='Value of beta for Ptp in range [0,1]')
parser.add_argument('--gamma_tp', type=float, default=2.3, help='normalizing term for Ptp should be positive real number')

parser.add_argument('--alpha_fp', type=float, default=0.3, help='The value of alpha for Pfp in range [0,1]')
parser.add_argument('--beta_fp',type=float, default=0.1, help='Value of beta for Pfp in range [0,1]')
parser.add_argument('--gamma_fp', type=float, default=3, help='normalizing term for Pfp should be positive real number')

args = parser.parse_args()


##Running the evaluation pipeline 

EVAL = Evaluations(args)

# defining the path to save the results

save_path = 'results_perturbed_fatigue/'

if not os.path.exists(save_path):

    os.makedirs(save_path,exist_ok=True)



# number of runs 

num_models = 10





for model in tqdm(range(1,num_models+1)):

    # each time this is initialized, the transition probability is perturbed
    env_perturbed = FatigueMDPPerturbed()

    

    new_path = os.path.join(save_path, 'model_'+str(model))

    if not os.path.exists(new_path):

        os.makedirs(new_path,exist_ok = True)
    

    EVAL.compute_performance_perturbed_fatigue(env_perturbed, new_path)