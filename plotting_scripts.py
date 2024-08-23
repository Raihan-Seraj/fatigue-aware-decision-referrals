import numpy as np 
import pandas as pd 
import pickle 
import matplotlib.pyplot as plt 
import seaborn as sns 




def create_performance_table(betas, result_path):
      
      final_data = pd.DataFrame(columns=['Beta','Total Human Cost-ADP', 'Total Human Cost-K','Human Cost Per Taskload-ADP','Human Cost Per Taskload-K',
                                         'Total Automation Cost-ADP','Total Automation Cost-K', 'Automation Cost Per Taskload-ADP', 'Automation Cost Per Taskload-K','Total Cost-ADP','Total Cost-K'])
      

      
      for beta in betas:
            
            all_run_human_cost_adp_path = result_path + 'plot_analysis/cost_comparison/beta '+str(beta)+'/all_human_cost_adp.npy'

            all_run_human_cost_k_path = result_path  + 'plot_analysis/cost_comparison/beta '+str(beta)+'/all_human_cost_k.npy'

            all_run_automation_cost_adp_path = result_path + 'plot_analysis/cost_comparison/beta '+str(beta)+'/all_auto_cost_adp.npy'

            all_run_automation_cost_k_path = result_path + 'plot_analysis/cost_comparison/beta '+str(beta)+'/all_auto_cost_k.npy'


            all_run_human_cost_adp = np.load(all_run_human_cost_adp_path)
            all_run_human_cost_k = np.load(all_run_human_cost_k_path)

            all_run_auto_cost_adp = np.load(all_run_automation_cost_adp_path)
            all_run_auto_cost_k =  np.load(all_run_automation_cost_k_path)

            # now loading the taskload 
            all_run_human_tl_adp_path = result_path + 'plot_analysis/cost_comparison/beta '+str(beta)+'/all_human_wl_adp.pkl'
            
            all_run_human_tl_k_path = result_path + 'plot_analysis/cost_comparison/beta '+str(beta)+'/all_human_wl_k.pkl'

            with open(all_run_human_tl_adp_path,'rb') as file1:
                  all_human_wl_adp = pickle.load(file1)


      
      



def plot_human_perf_vs_taskload(beta, result_path):

    final_path = result_path +'plot_analysis/cost_comparison/beta '+str(beta)+'/'


    all_run_hum_wl_adp_path = final_path +'all_human_wl_adp.pkl'

    all_run_hum_wl_k_path = final_path  + 'all_human_wl_k.pkl'

    all_hum_cost_adp_path = final_path +'all_human_cost_adp.npy'

    all_hum_cost_k_path = final_path+'all_human_cost_k.npy'



    with open(all_run_hum_wl_adp_path,'rb') as file1:
           all_human_wl_adp = pickle.load(file1)
        
    
    with open(all_run_hum_wl_k_path,'rb') as file2:
           all_human_wl_k = pickle.load(file2)

    
    all_human_cost_adp = np.load(all_hum_cost_adp_path)

    all_human_cost_k = np.load(all_hum_cost_k_path)

    import ipdb;ipdb.set_trace()

    return 



if __name__=='__main__':
      
    result_path = 'results_v2/'
    beta = 0.5 #
    plot_human_perf_vs_taskload(beta, result_path)