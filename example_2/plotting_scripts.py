import numpy as np 
import pandas as pd 
import pickle 
import matplotlib.pyplot as plt 
import seaborn as sns 




def create_performance_table(beta,alphas,mus,lamdas,num_tasks_per_batch, result_path):

	result_path = result_path + 'num_tasks '+str(num_tasks_per_batch)+'/'

	total_num_tasks=num_tasks_per_batch

	round_decimal_places=3
      
	final_data = pd.DataFrame(columns=['Beta','Alpha','Mu','Lambda', 'Expected Total Human Cost-ADP', ' Expected Total Human Cost-K','Expected Human Cost Per Taskload-ADP','Expected Human Cost Per Taskload-K',
										'Expected Total Automation Cost-ADP','Expected Total Automation Cost-K', 'Expected Automation Cost Per Taskload-ADP', 'Expected Automation Cost Per Taskload-K',
										'Expected Total Deferred Cost-ADP','Expected Total Deferred Cost-K',
										'Expected Total Cost-ADP','Expected Total Cost-K', 'Expected Taskload of the Human-ADP','Expected Taskload of the Human-K'])



	for alpha in alphas:

		for mu in mus:

			for lamda in lamdas:

				all_run_human_cost_adp_path = result_path + 'beta '+str(beta)+'/alpha '+str(alpha)+'/mu_'+str(mu)+'_lambda_'+str(lamda)+'/all_human_cost_adp.npy'

				all_run_human_cost_k_path = result_path  + 'beta '+str(beta)+'/alpha '+str(alpha)+'/mu_'+str(mu)+'_lambda_'+str(lamda)+'/all_human_cost_k.npy'

				all_run_automation_cost_adp_path = result_path + 'beta '+str(beta)+'/alpha '+str(alpha)+'/mu_'+str(mu)+'_lambda_'+str(lamda)+'/all_auto_cost_adp.npy'

				all_run_automation_cost_k_path = result_path + 'beta '+str(beta)+'/alpha '+str(alpha)+'/mu_'+str(mu)+'_lambda_'+str(lamda)+'/all_auto_cost_k.npy'


				all_run_deferred_cost_k_path = result_path + 'beta '+str(beta)+'/alpha '+str(alpha)+'/mu_'+str(mu)+'_lambda_'+str(lamda)+'/all_deferred_cost_k.npy'
				all_run_deferred_cost_adp_path = result_path + 'beta '+str(beta)+'/alpha '+str(alpha)+'/mu_'+str(mu)+'_lambda_'+str(lamda)+'/all_deferred_cost_adp.npy'



				all_run_human_cost_adp = np.load(all_run_human_cost_adp_path)
				all_run_human_cost_k = np.load(all_run_human_cost_k_path)

				all_run_auto_cost_adp = np.load(all_run_automation_cost_adp_path)
				all_run_auto_cost_k =  np.load(all_run_automation_cost_k_path)

				all_run_deferred_cost_k = np.load(all_run_deferred_cost_k_path)
				all_run_deferred_cost_adp = np.load(all_run_deferred_cost_adp_path)


				# now loading the taskload 
				all_run_human_tl_adp_path = result_path + 'beta '+str(beta)+'/alpha '+str(alpha)+'/mu_'+str(mu)+'_lambda_'+str(lamda)+'/all_human_wl_adp.pkl'

				all_run_human_tl_k_path = result_path + 'beta '+str(beta)+'/alpha '+str(alpha)+'/mu_'+str(mu)+'_lambda_'+str(lamda)+'/all_human_wl_k.pkl'

				with open(all_run_human_tl_adp_path,'rb') as file1:

					all_run_human_wl_adp = pickle.load(file1)

				with open(all_run_human_tl_k_path, 'rb') as file2:
					
					all_run_human_wl_k = pickle.load(file2)

				
				human_cost_per_wl_per_run_adp = [all_run_human_cost_adp[i]/(sum(all_run_human_wl_adp['Run-'+str(i+1)])) for i in range(len(all_run_human_cost_adp))]

				human_cost_per_wl_per_run_k = [all_run_human_cost_k[i]/(sum(all_run_human_wl_k['Run-'+str(i+1)])) for i in range(len(all_run_human_cost_k))]

				automation_cost_per_wl_per_run_adp = [all_run_auto_cost_adp[i]/(sum(total_num_tasks-all_run_human_wl_adp['Run-'+str(i+1)])) for i in range(len(all_run_auto_cost_adp))]

				automation_cost_per_wl_per_run_k = [all_run_auto_cost_k[i]/(sum(total_num_tasks-all_run_human_wl_k['Run-'+str(i+1)])) for i in range(len(all_run_auto_cost_k))]
			
				

				expected_total_human_cost_adp = np.round(np.mean(all_run_human_cost_adp),round_decimal_places)
				std_total_human_cost_adp = np.round(np.std(all_run_human_cost_adp),round_decimal_places)

				expected_total_human_cost_k = np.round(np.mean(all_run_human_cost_k),round_decimal_places)
				std_total_human_cost_k = np.round(np.std(all_run_human_cost_k),round_decimal_places)

				expected_human_cost_per_wl_adp = np.round(np.mean(human_cost_per_wl_per_run_adp),round_decimal_places)
				std_human_cost_per_wl_adp = np.round(np.std(human_cost_per_wl_per_run_adp),round_decimal_places)

				expected_human_cost_per_wl_k = np.round(np.mean(human_cost_per_wl_per_run_k),round_decimal_places)
				std_human_cost_per_wl_k = np.round(np.std(human_cost_per_wl_per_run_k),round_decimal_places)

				##

				expected_total_automation_cost_adp = np.round(np.mean(all_run_auto_cost_adp),round_decimal_places)
				std_total_automation_cost_adp = np.round(np.std(all_run_auto_cost_adp),round_decimal_places)

				expected_total_automation_cost_k = np.round(np.mean(all_run_auto_cost_k),round_decimal_places)
				std_total_automation_cost_k = np.round(np.std(all_run_auto_cost_k),round_decimal_places)

				expected_automation_cost_per_wl_adp = np.round(np.mean(automation_cost_per_wl_per_run_adp),round_decimal_places)
				std_automation_cost_per_wl_adp = np.round(np.std(automation_cost_per_wl_per_run_adp),round_decimal_places)

				expected_automation_cost_per_wl_k = np.round(np.mean(automation_cost_per_wl_per_run_k),round_decimal_places)
				std_automation_cost_per_wl_k = np.round(np.std(automation_cost_per_wl_per_run_k),round_decimal_places)

				expected_total_deferred_cost_k = np.round(np.mean(all_run_deferred_cost_k),round_decimal_places)
				std_total_deferred_cost_k = np.round(np.std(all_run_deferred_cost_k),round_decimal_places)

				expected_total_deferred_cost_adp = np.round(np.mean(all_run_deferred_cost_adp),round_decimal_places)
				std_total_deferred_cost_adp = np.round(np.std(all_run_deferred_cost_adp),round_decimal_places)



				expected_total_cost_adp = np.round(np.mean(all_run_human_cost_adp + all_run_auto_cost_adp+all_run_deferred_cost_adp),round_decimal_places)
				std_total_cost_adp = np.round(np.std(all_run_human_cost_adp + all_run_auto_cost_adp+ all_run_deferred_cost_adp),round_decimal_places)
			
				expected_total_cost_k = np.round(np.mean(all_run_human_cost_k + all_run_auto_cost_k + all_run_deferred_cost_k),round_decimal_places)
				std_total_cost_k = np.round(np.std(all_run_human_cost_k + all_run_auto_cost_k + all_run_deferred_cost_k),round_decimal_places)

				expected_taskload_human_adp = np.round(np.mean([np.mean(all_run_human_wl_adp['Run-'+str(i+1)]) for i in range(len(all_run_human_wl_adp))]),round_decimal_places)
				std_taskload_human_adp = np.round(np.std([sum(all_run_human_wl_adp['Run-'+str(i+1)]) for i in range(len(all_run_human_wl_adp))]),round_decimal_places)
				expected_taskload_human_k = np.round(np.mean([np.mean(all_run_human_wl_k['Run-'+str(i+1)]) for i in range(len(all_run_human_wl_k))]),round_decimal_places)
				std_taskload_human_k = np.round(np.std([np.mean(all_run_human_wl_k['Run-'+str(i+1)]) for i in range(len(all_run_human_wl_k))]),round_decimal_places)


				

				new_row  = pd.DataFrame([[beta, alpha, mu, lamda, 
							  	str(expected_total_human_cost_adp)+'$\pm$'+str(std_total_human_cost_adp),
								str(expected_total_human_cost_k)+'$\pm$'+str(std_total_human_cost_k),
								str(expected_human_cost_per_wl_adp)+'$\pm$'+str(std_human_cost_per_wl_adp),
								str(expected_human_cost_per_wl_k)+'$\pm$'+str(std_human_cost_per_wl_k),
								str(expected_total_automation_cost_adp)+'$\pm$'+str(std_total_automation_cost_adp),
								str(expected_total_automation_cost_k)+'$\pm$'+str(std_total_automation_cost_k),
								str(expected_automation_cost_per_wl_adp)+'$\pm$'+str(std_automation_cost_per_wl_adp),
								str(expected_automation_cost_per_wl_k)+'$\pm$'+str(std_automation_cost_per_wl_k),
								str(expected_total_deferred_cost_adp)+'$\pm$'+str(std_total_deferred_cost_adp),
								str(expected_total_deferred_cost_k)+'$\pm$'+str(std_total_deferred_cost_k),
								str(expected_total_cost_adp)+'$\pm$'+str(std_total_cost_adp),
								str(expected_total_cost_k)+'$\pm$'+str(std_total_cost_k),
								str(expected_taskload_human_adp)+'$\pm$'+str(std_taskload_human_adp),
								str(expected_taskload_human_k)+'$\pm$'+str(std_taskload_human_k)]],columns=final_data.columns)
				

				final_data = pd.concat([final_data,new_row],ignore_index=True)

	final_data.to_csv(result_path+'beta '+str(beta)+'/refined_performance_table.csv')
	
	return


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
      
	result_path = 'results/'
	num_tasks_per_batch=20
	
	beta = 0.5

	alphas = [1.0, 2.0, 4.0, 7.0, 9.0]

	mus = [0.1, 0.003, 0.05, 0.07]

	lamdas = [0.1,0.03,0.07, 0.007]

	
	create_performance_table(beta,alphas, mus, lamdas, num_tasks_per_batch, result_path)