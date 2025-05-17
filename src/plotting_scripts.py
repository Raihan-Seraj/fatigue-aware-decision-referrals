import numpy as np 
import pandas as pd 
import pickle 
import matplotlib.pyplot as plt 
import seaborn as sns 
import argparse
import os

plt.rcParams["figure.figsize"] = (6.4, 5.3)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def create_performance_table(alphas_tp,betas_tp, gammas_tp,alphas_fp,betas_fp, gammas_fp,num_tasks_per_batch, result_path, fatigue_model):
	
	if fatigue_model.lower()=='fatigue_model_1':
		result_path = result_path + fatigue_model+'/num_tasks '+str(num_tasks_per_batch)+'/'

		total_num_tasks=num_tasks_per_batch

		round_decimal_places=5
		
		final_data = pd.DataFrame(columns=['Alpha_a','Beta_a','Gamma_a','Alpha_b','Beta_b','Gamma_b','Expected Total Human Cost-ADP', ' Expected Total Human Cost-K','Expected Human Cost Per Taskload-ADP','Expected Human Cost Per Taskload-K',
											'Expected Total Automation Cost-ADP','Expected Total Automation Cost-K', 'Expected Automation Cost Per Taskload-ADP', 'Expected Automation Cost Per Taskload-K',
											'Expected Total Deferred Cost-ADP','Expected Total Deferred Cost-K',
											'Expected Total Cost-ADP','Expected Total Cost-K', 'Expected Taskload of the Human-ADP','Expected Taskload of the Human-K'])


		
		for alpha_tp in alphas_tp:

			for beta_tp in betas_tp:

				for gamma_tp in gammas_tp:


					for alpha_fp in alphas_fp:

						for beta_fp in betas_fp:

							for gamma_fp in gammas_fp:

								global_path_cost = result_path +'alpha_tp '+str(alpha_tp)+'/beta_tp '+str(beta_tp)+'/gamma_tp '+str(gamma_tp)+'/alpha_fp '+str(alpha_fp)+'/beta_fp '+str(beta_fp)+'/gamma_fp '+str(gamma_fp)+'/cost_comparison/'
								global_path_plot = result_path + 'alpha_tp '+str(alpha_tp)+'/beta_tp '+str(beta_tp)+'/gamma_tp '+str(gamma_tp)+'/alpha_fp '+str(alpha_fp)+'/beta_fp '+str(beta_fp)+'/gamma_fp '+str(gamma_fp)+'/plot_analysis/'

								all_run_human_cost_adp_path = global_path_cost +'all_human_cost_adp.npy'
								
								all_run_human_cost_k_path = global_path_cost+'all_human_cost_k.npy'

								all_run_automation_cost_adp_path = global_path_cost+'all_auto_cost_adp.npy'

								all_run_automation_cost_k_path = global_path_cost+'all_auto_cost_k.npy'


								all_run_deferred_cost_k_path = global_path_cost+'all_deferred_cost_k.npy'
								all_run_deferred_cost_adp_path = global_path_cost+'all_deferred_cost_adp.npy'
								
								
								
								try:
									all_run_human_cost_adp = np.load(all_run_human_cost_adp_path)
									all_run_human_cost_k = np.load(all_run_human_cost_k_path)

									all_run_auto_cost_adp = np.load(all_run_automation_cost_adp_path)
									all_run_auto_cost_k =  np.load(all_run_automation_cost_k_path)

									all_run_deferred_cost_k = np.load(all_run_deferred_cost_k_path)
									all_run_deferred_cost_adp = np.load(all_run_deferred_cost_adp_path)
								except FileNotFoundError:

									
									print("File not found")
									#continue 
									


								# now loading the taskload 
								all_run_human_tl_adp_path = global_path_cost+'all_human_wl_adp.pkl'

								all_run_human_tl_k_path = global_path_cost+'all_human_wl_k.pkl'

								
								try:
									with open(all_run_human_tl_adp_path,'rb') as file1:

										all_run_human_wl_adp = pickle.load(file1)

									with open(all_run_human_tl_k_path, 'rb') as file2:
										
										all_run_human_wl_k = pickle.load(file2)
								
								except FileNotFoundError:
									print("File not found skipping to the next file")
									#continue 

							
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


								

								new_row  = pd.DataFrame([[str(alpha_tp), str(beta_tp), str(gamma_tp), str(alpha_fp), str(beta_fp), str(gamma_fp),
												str(expected_total_human_cost_adp)+'$pm$'+str(std_total_human_cost_adp),
												str(expected_total_human_cost_k)+'$pm$'+str(std_total_human_cost_k),
												str(expected_human_cost_per_wl_adp)+'$pm$'+str(std_human_cost_per_wl_adp),
												str(expected_human_cost_per_wl_k)+'$pm$'+str(std_human_cost_per_wl_k),
												str(expected_total_automation_cost_adp)+'$pm$'+str(std_total_automation_cost_adp),
												str(expected_total_automation_cost_k)+'$pm$'+str(std_total_automation_cost_k),
												str(expected_automation_cost_per_wl_adp)+'$pm$'+str(std_automation_cost_per_wl_adp),
												str(expected_automation_cost_per_wl_k)+'$pm$'+str(std_automation_cost_per_wl_k),
												str(expected_total_deferred_cost_adp)+'$pm$'+str(std_total_deferred_cost_adp),
												str(expected_total_deferred_cost_k)+'$pm$'+str(std_total_deferred_cost_k),
												str(expected_total_cost_adp)+'$pm$'+str(std_total_cost_adp),
												str(expected_total_cost_k)+'$pm$'+str(std_total_cost_k),
												str(expected_taskload_human_adp)+'$pm$'+str(std_taskload_human_adp),
												str(expected_taskload_human_k)+'$pm$'+str(std_taskload_human_k)]],columns=final_data.columns)
								
								
								final_data = pd.concat([final_data,new_row],ignore_index=True)
								
					
		
		final_data.to_csv(result_path+ '/refined_performance_table.csv')

	elif fatigue_model.lower()=='fatigue_model_3':


		result_path = result_path + 'fatigue_model_3/num_tasks '+str(num_tasks_per_batch)+'/'

		total_num_tasks=num_tasks_per_batch

		round_decimal_places=5
	
		final_data = pd.DataFrame(columns=['Expected Total Human Cost-ADP', ' Expected Total Human Cost-K','Expected Human Cost Per Taskload-ADP','Expected Human Cost Per Taskload-K',
											'Expected Total Automation Cost-ADP','Expected Total Automation Cost-K', 'Expected Automation Cost Per Taskload-ADP', 'Expected Automation Cost Per Taskload-K',
											'Expected Total Deferred Cost-ADP','Expected Total Deferred Cost-K',
											'Expected Total Cost-ADP','Expected Total Cost-K', 'Expected Taskload of the Human-ADP','Expected Taskload of the Human-K'])




		all_run_human_cost_adp_path = result_path+'cost_comparison/all_human_cost_adp.npy'

		all_run_human_cost_k_path = result_path +'cost_comparison/all_human_cost_k.npy'

		all_run_automation_cost_adp_path = result_path+'cost_comparison/all_auto_cost_adp.npy'

		all_run_automation_cost_k_path = result_path+'cost_comparison/all_auto_cost_k.npy'


		all_run_deferred_cost_k_path = result_path+'cost_comparison/all_deferred_cost_k.npy'
		all_run_deferred_cost_adp_path = result_path+'cost_comparison/all_deferred_cost_adp.npy'
		
		

		try:
			all_run_human_cost_adp = np.load(all_run_human_cost_adp_path)
			all_run_human_cost_k = np.load(all_run_human_cost_k_path)

			all_run_auto_cost_adp = np.load(all_run_automation_cost_adp_path)
			all_run_auto_cost_k =  np.load(all_run_automation_cost_k_path)

			all_run_deferred_cost_k = np.load(all_run_deferred_cost_k_path)
			all_run_deferred_cost_adp = np.load(all_run_deferred_cost_adp_path)
		except FileNotFoundError:

			
			print("File not found")
			#continue 
			


		# now loading the taskload 
		all_run_human_tl_adp_path = result_path + 'cost_comparison/all_human_wl_adp.pkl'

		all_run_human_tl_k_path = result_path + 'cost_comparison/all_human_wl_k.pkl'

		
		try:
			with open(all_run_human_tl_adp_path,'rb') as file1:

				all_run_human_wl_adp = pickle.load(file1)

			with open(all_run_human_tl_k_path, 'rb') as file2:
				
				all_run_human_wl_k = pickle.load(file2)
		
		except FileNotFoundError:
			print("File not found skipping to the next file")
			#continue 


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


		

		new_row  = pd.DataFrame([[
						str(expected_total_human_cost_adp)+'$pm$'+str(std_total_human_cost_adp),
						str(expected_total_human_cost_k)+'$pm$'+str(std_total_human_cost_k),
						str(expected_human_cost_per_wl_adp)+'$pm$'+str(std_human_cost_per_wl_adp),
						str(expected_human_cost_per_wl_k)+'$pm$'+str(std_human_cost_per_wl_k),
						str(expected_total_automation_cost_adp)+'$pm$'+str(std_total_automation_cost_adp),
						str(expected_total_automation_cost_k)+'$pm$'+str(std_total_automation_cost_k),
						str(expected_automation_cost_per_wl_adp)+'$pm$'+str(std_automation_cost_per_wl_adp),
						str(expected_automation_cost_per_wl_k)+'$pm$'+str(std_automation_cost_per_wl_k),
						str(expected_total_deferred_cost_adp)+'$pm$'+str(std_total_deferred_cost_adp),
						str(expected_total_deferred_cost_k)+'$pm$'+str(std_total_deferred_cost_k),
						str(expected_total_cost_adp)+'$pm$'+str(std_total_cost_adp),
						str(expected_total_cost_k)+'$pm$'+str(std_total_cost_k),
						str(expected_taskload_human_adp)+'$pm$'+str(std_taskload_human_adp),
						str(expected_taskload_human_k)+'$pm$'+str(std_taskload_human_k)]],columns=final_data.columns)
		
	
		final_data = pd.concat([final_data,new_row],ignore_index=True)
		

	
		final_data.to_csv(result_path+ '/refined_performance_table.csv')

	

		
	
	return





def create_complete_performance_table(fatigue_model):

	result_path = 'results/'
	num_tasks_per_batch=20
	
	# beta = 0.5

	# alphas = [1.0, 2.0, 5.0, 7.0, 9.0]

	# mus = [0.1, 0.003, 0.05, 0.07]

	# lamdas = [0.1,0.03,0.07, 0.007]
	if fatigue_model.lower()=='fatigue_model_1':
		alphas_tp = [0.2]
		alphas_fp = [0.3]
		betas_tp = [0.1]
		betas_fp = [0.1]
		gammas_tp = [2.5]
		gammas_fp =[3]

	elif fatigue_model.lower()=='fatigue_model_3':
		
		alphas_tp = None
		alphas_fp = None
		betas_tp = None
		betas_fp = None
		gammas_tp = None
		gammas_fp =None

	
	create_performance_table(alphas_tp,betas_tp,gammas_tp,alphas_fp, betas_fp, gammas_fp, num_tasks_per_batch, result_path,fatigue_model)

	return


def plot_taskload_comparison(result_path, num_tasks, alpha_tp,alpha_fp,beta_tp,beta_fp, gamma_tp, gamma_fp, stime=20,computation='median',fatigue_model='fatigue_model_1'):

	if fatigue_model.lower()=='fatigue_model_1':
		path = result_path + 'fatigue_model_1/num_tasks '+str(num_tasks)+'/alpha_tp '+str(alpha_tp)+'/beta_tp '+str(beta_tp)+'/gamma_tp '+str(gamma_tp)+'/alpha_fp '+str(alpha_fp)+'/beta_fp '+str(beta_fp)+'/gamma_fp '+str(gamma_fp)+'/plot_analysis/'

	elif fatigue_model.lower()=='fatigue_model_3':
		path = result_path+'fatigue_model_3/num_tasks '+str(num_tasks)+'/plot_analysis/'
	
	else:
		raise ValueError("Invalid path")

	adp_path = path + 'all_taskload_adp.pkl'

	k_path = path + 'all_taskload_k.pkl'

	## loding the files

	with open(adp_path, 'rb') as file1:

		data_adp = pickle.load(file1)
	
	with open(k_path, 'rb') as file2:
		data_k = pickle.load(file2)
	
	if computation.lower()=='median':
		median_workload_adp = np.median(data_adp, axis=0)[:stime]

		q1_adp = np.percentile(data_adp, q = 25, axis=0)[:stime]

		q3_adp = np.percentile(data_adp, q=75, axis=0)[:stime]


		median_workload_k = np.median(data_k, axis=0)[:stime]

		q1_k = np.percentile(data_k, q = 25, axis=0)[:stime]

		q3_k = np.percentile(data_k, q=75, axis=0)[:stime]
	
	


		simulation_time = np.arange(1, stime+1)

		
		plt.step(simulation_time, median_workload_adp, label='Approximate Dynamic Program', color='blue', where='mid')

		
		plt.fill_between(simulation_time, q1_adp, q3_adp, step='mid', alpha=0.2, color='blue')


		
		plt.step(simulation_time, median_workload_k, label='Algorithm in [3]', color='black', where='mid')


		plt.fill_between(simulation_time, q1_k, q3_k, step='mid', alpha=0.2, color='black')

		plt.grid(True)
		plt.xlabel('Time Horizon T')
		plt.ylabel('Taskload')

		#tick_positions = [i+1 for i in range(100)]

		#tick_labels = [str(i+1) for i in range(100) ]

		#plt.xticks(tick_positions,tick_labels)

		plt.legend()
		plt.savefig(path+'Median-Taskload-Plot-Comparison.pdf')
		plt.close()
		plt.clf()
	
	elif computation=='mean':

		mean_workload_adp = np.mean(data_adp, axis=0)[:stime]

		std_workload_adp = np.std(data_adp, axis=0)[:stime]

		upper_adp= mean_workload_adp + std_workload_adp

		lower_adp = mean_workload_adp-std_workload_adp


		mean_workload_k = np.mean(data_k, axis=0)[:stime]

		std_workload_k = np.std(data_k,axis=0)[:stime]

		upper_k = mean_workload_k + std_workload_k

		lower_k = mean_workload_k-std_workload_k
	
	
		
		

		simulation_time = np.arange(1, stime+1)

		
		plt.step(simulation_time, mean_workload_adp, label='Approximate Dynamic Program', color='black', where='mid',linestyle='--')

		
		fill=plt.fill_between(simulation_time, lower_adp, upper_adp, step='mid', alpha=0.3, color='gray',edgecolor='k')
		fill.set_hatch('..')

		
		plt.step(simulation_time, mean_workload_k, label='Algorithm in [3]', color='black', where='mid')


		fill2=plt.fill_between(simulation_time, lower_k, upper_k, step='mid', alpha=0.3, color='gray', edgecolor='k')

		fill2.set_hatch('//')
		plt.grid(True)
		plt.xlabel('Time Horizon T',fontsize=16)
		plt.ylabel('Taskload',fontsize=16)

		#tick_positions = [i+1 for i in range(100)]

		#tick_labels = [str(i+1) for i in range(100) ]

		#plt.xticks(tick_positions,tick_labels)
		plt.tick_params(axis='both', which='major', labelsize=18)  # Set major tick label size
		plt.tick_params(axis='both', which='minor', labelsize=16)  # Set minor tick label size (if applicable)


		plt.legend(fontsize=16)
		plt.savefig(path+'Mean-Taskload-Plot-Comparison.pdf')
		plt.close()
		plt.clf()
	
	else:
		raise ValueError("Invalid computation")


	



def plot_fatigue_comparison(result_path, num_tasks,alpha_tp,alpha_fp, beta_tp, beta_fp, gamma_tp, gamma_fp,stime=20,computation='median',fatigue_model='fatigue_model_1'):

	path = result_path + 'num_tasks '+str(num_tasks)+'/alpha_tp '+str(alpha_tp)+'/beta_tp '+str(beta_tp)+'/gamma_tp '+str(gamma_tp)+'/alpha_fp '+str(alpha_fp)+'/beta_fp '+str(beta_fp)+'/gamma_fp '+str(gamma_fp)+'/plot_analysis/'

	if fatigue_model.lower()=='fatigue_model_1':
		path = result_path + 'fatigue_model_1/num_tasks '+str(num_tasks)+'/alpha_tp '+str(alpha_tp)+'/beta_tp '+str(beta_tp)+'/gamma_tp '+str(gamma_tp)+'/alpha_fp '+str(alpha_fp)+'/beta_fp '+str(beta_fp)+'/gamma_fp '+str(gamma_fp)+'/plot_analysis/'

	elif fatigue_model.lower()=='fatigue_model_3':
		path = result_path+'fatigue_model_3/num_tasks '+str(num_tasks)+'/plot_analysis/'
	
	
	adp_path = path + 'all_fatigue_adp.pkl'

	k_path = path + 'all_fatigue_k.pkl'

	## loding the files

	with open(adp_path, 'rb') as file1:

		data_adp = pickle.load(file1)
	
	with open(k_path, 'rb') as file2:
		data_k = pickle.load(file2)
	

	if computation.lower()=='median':
		median_fatigue_adp = np.median(data_adp, axis=0)[:stime]

		q1_adp = np.percentile(data_adp, q = 25, axis=0)[:stime]

		q3_adp = np.percentile(data_adp, q=75, axis=0)[:stime]


		median_fatigue_k = np.median(data_k, axis=0)[:stime]

		q1_k = np.percentile(data_k, q = 25, axis=0)[:stime]

		q3_k = np.percentile(data_k, q=75, axis=0)[:stime]


		simulation_time = np.arange(1,stime+1)

		
		plt.step(simulation_time, median_fatigue_adp, label='Approximate Dynamic Program', color='blue', where='mid')

		
		plt.fill_between(simulation_time, q1_adp, q3_adp, step='mid', alpha=0.2, color='blue')


		
		plt.step(simulation_time, median_fatigue_k, label='Algorithm in [3]', color='black', where='mid')


		plt.fill_between(simulation_time, q1_k, q3_k, step='mid', alpha=0.2, color='black')




		plt.grid(True)
		plt.xlabel('Time Horizon T')
		plt.ylabel('Fatigue')

		#tick_positions = [i+1 for i in range(100)]

		#tick_labels = [str(i+1) for i in range(100) ]

		#plt.xticks(tick_positions,tick_labels)

		plt.tick_params(axis='both', which='major', labelsize=18)  # Set major tick label size
		plt.tick_params(axis='both', which='minor', labelsize=16)  # Set minor tick label size (if applicable)

		plt.legend()
		plt.savefig(path+'Median-Fatigue-Plot-Comparison.pdf')
		plt.close()
		plt.clf()
	
	elif computation.lower()=='mean':
		
		mean_fatigue_adp = np.mean(data_adp, axis=0)[:stime]

		std_fatigue_adp = np.std(data_adp, axis=0)[:stime]

		lower_adp = mean_fatigue_adp-std_fatigue_adp

		upper_adp = mean_fatigue_adp+std_fatigue_adp

		mean_fatigue_k = np.mean(data_k, axis=0)[:stime]

		std_fatigue_k = np.std(data_k, axis=0)[:stime]

		lower_k = mean_fatigue_k-std_fatigue_k

		upper_k = mean_fatigue_k + std_fatigue_k


		simulation_time = np.arange(1,stime+1)

		
		plt.step(simulation_time, mean_fatigue_adp, label='Approximate Dynamic Program', color='black', where='mid',linestyle='--')

		
		fill=plt.fill_between(simulation_time, lower_adp, upper_adp, step='mid', alpha=0.3, color='gray',edgecolor='k')

		fill.set_hatch('..')


		
		plt.step(simulation_time, mean_fatigue_k, label='Algorithm in [3]', color='black', where='mid')


		fill2=plt.fill_between(simulation_time, lower_k, upper_k, step='mid', alpha=0.3, color='gray',edgecolor='k')
		fill2.set_hatch('//')




		plt.grid(True)
		plt.xlabel('Time Horizon T',fontsize=16)
		plt.ylabel('Fatigue',fontsize=16)

		#tick_positions = [i+1 for i in range(100)]

		#tick_labels = [str(i+1) for i in range(100) ]

		#plt.xticks(tick_positions,tick_labels)

		plt.tick_params(axis='both', which='major', labelsize=18)  # Set major tick label size
		plt.tick_params(axis='both', which='minor', labelsize=16)  # Set minor tick label size (if applicable)


		plt.legend(fontsize=16,loc='upper left')
		plt.savefig(path+'Mean-Fatigue-Plot-Comparison.pdf')
		plt.close()
		plt.clf()

	else:
		raise ValueError("Incorrect computation")


def plot_performance(result_path, num_tasks, alpha_tp, alpha_fp, beta_tp, beta_fp,gamma_tp, gamma_fp,fatigue_model='fatigue_model_1'):

	if fatigue_model.lower()=='fatigue_model_1':
		path = result_path + 'fatigue_model_1/num_tasks '+str(num_tasks)+'/alpha_tp '+str(alpha_tp)+'/beta_tp '+str(beta_tp)+'/gamma_tp '+str(gamma_tp)+'/alpha_fp '+str(alpha_fp)+'/beta_fp '+str(beta_fp)+'/gamma_fp '+str(gamma_fp)+'/cost_comparison/'
	elif fatigue_model.lower()=='fatigue_model_2':
		path = result_path+'fatigue_model_2/num_tasks '+str(num_tasks)+'/cost_comparison/'

	else:
		raise ValueError("Invalid fatigue model")
	
	auto_cost_adp_path = path +'all_auto_cost_adp.npy'

	hum_cost_adp_path = path  + 'all_human_cost_adp.npy'

	auto_cost_k_path = path +'all_auto_cost_k.npy'

	hum_cost_k_path = path  + 'all_human_cost_k.npy'


	auto_cost_adp = np.load(auto_cost_adp_path)

	human_cost_adp = np.load(hum_cost_adp_path)

	auto_cost_k = np.load(auto_cost_k_path)
	human_cost_k = np.load(hum_cost_k_path)

	total_cost_adp = auto_cost_adp + human_cost_adp
	total_cost_k = auto_cost_k + human_cost_k


	all_arrays = {'AutoCost-ADP':auto_cost_adp, 'AutoCost in [3]':auto_cost_k, 'HumCost-ADP':human_cost_adp, 'HumCost in [3]':human_cost_k, 'TotalCost-ADP':total_cost_adp, 'TotalCost in [3]':total_cost_k
			   
			   }

	df = pd.DataFrame(all_arrays,columns=['AutoCost-ADP','AutoCost in [3]', 'HumCost-ADP','HumCost in [3]','TotalCost-ADP','TotalCost in [3]'])
	
	df_tc= df[['TotalCost-ADP', 'TotalCost in [3]']].melt(var_name='Category', value_name='Costs')

	df_hc= df[['HumCost-ADP', 'HumCost in [3]']].melt(var_name='Category', value_name='Costs')

	df_ac= df[['AutoCost-ADP', 'AutoCost in [3]']].melt(var_name='Category', value_name='Costs')
	
	
	clr = sns.color_palette("hls", 8)
	# Set default linewidths for boxes and medians
	plt.rcParams['boxplot.boxprops.linewidth'] = 2.0
	plt.rcParams['boxplot.medianprops.linewidth'] = 2.0
	plt.rcParams['boxplot.flierprops.linewidth'] = 2.0
	plt.rcParams['boxplot.whiskerprops.linewidth'] = 2.0
	plt.rcParams['boxplot.capprops.linewidth'] = 2.0
	figs, axes = plt.subplots(1,3,figsize=(15,5))
	# Create the boxplots for each type of cost
	# Create the boxplots with custom hatching patterns
	sns.boxplot(x='Category', y='Costs', data=df_ac, ax=axes[0], flierprops={"marker": "x"})
	# Set all box colors to gray
	#for box in axes[0].patches + axes[1].patches + axes[2].patches:
		
	for i, box in enumerate(axes[0].patches):
		if i ==0:  # First category
			box.set_hatch('..')
			box.set_facecolor('blue')
		else:  # Second category 
			box.set_hatch('//')
			box.set_facecolor('red')
	axes[0].tick_params(axis='both', labelsize=13)

	sns.boxplot(x='Category', y='Costs', data=df_hc, ax=axes[1], flierprops={"marker": "x"}) 
	
	for i, box in enumerate(axes[1].patches):
		if i==0:  # First category
			box.set_hatch('..')
			box.set_facecolor('blue')
		else:  # Second category
			box.set_hatch('//')
			box.set_facecolor('red')
	axes[1].tick_params(axis='both', labelsize=13)

	sns.boxplot(x='Category', y='Costs', data=df_tc, ax=axes[2], flierprops={"marker": "x"})
	
		
	for i, box in enumerate(axes[2].patches):
		if i==0:  # First category
			box.set_hatch('..')
			box.set_facecolor('blue')
		else:  # Second category
			box.set_hatch('//')
			box.set_facecolor('red')
	axes[2].tick_params(axis='both', labelsize=13)
	# sns.boxplot(x='Category', y='Costs', hue='Category', data=df_ac, ax=axes[0],flierprops={"marker": "x"})
	# axes[0].tick_params(axis='both', labelsize=13)  # Set tick label size for first plot

	# sns.boxplot(x='Category', y='Costs', hue='Category', data=df_hc, ax=axes[1],flierprops={"marker": "x"})
	# axes[1].tick_params(axis='both', labelsize=13)  # Set tick label size for second plot

	# sns.boxplot(x='Category', y='Costs', hue='Category', data=df_tc, ax=axes[2],flierprops={"marker": "x"})
	# axes[2].tick_params(axis='both', labelsize=13)  # Set tick label size for third plot
	

	plt.savefig(path+'BoxPlot-Cost-Comparison.pdf')
	plt.clf()
	plt.close()

	return

def compare_cum_cost(result_path, num_tasks, alpha_tp, alpha_fp, beta_tp, beta_fp,gamma_tp, gamma_fp,fatigue_model='fatigue_model_1'):
	

	if fatigue_model.lower()=='fatigue_model_1':
		path = result_path + 'fatigue_model_1/num_tasks '+str(num_tasks)+'/alpha_tp '+str(alpha_tp)+'/beta_tp '+str(beta_tp)+'/gamma_tp '+str(gamma_tp)+'/alpha_fp '+str(alpha_fp)+'/beta_fp '+str(beta_fp)+'/gamma_fp '+str(gamma_fp)+'/cost_comparison/'
	elif fatigue_model.lower()=='fatigue_model_2':
		path = result_path+'fatigue_model_2/num_tasks '+str(num_tasks)+'/cost_comparison/'

	else:
		raise ValueError("Invalid fatigue model")

	
	all_run_cum_cost_k = path + 'all_cum_cost_k.pkl'
	all_run_cum_cost_adp = path + 'all_cum_cost_adp.pkl'

	with open(all_run_cum_cost_k, 'rb') as file1:

		cum_cost_k = pickle.load(file1)
	
	with open(all_run_cum_cost_adp, 'rb') as file2:

		cum_cost_adp = pickle.load(file2)
	

	mean_cum_cost_k  = np.mean(cum_cost_k, axis=0)
	std_cum_cost_k = np.std(cum_cost_k, axis=0)	

	mean_cum_cost_adp = np.mean(cum_cost_adp, axis=0)
	std_cum_cost_adp = np.std(cum_cost_adp, axis=0)	
	
	std_k_squared = [item**2 for item in std_cum_cost_k]
	std_adp_squared = [item**2 for item in std_cum_cost_adp]

	running_std_k = np.sqrt(np.cumsum(std_k_squared))	
	running_std_adp = np.sqrt(np.cumsum(std_adp_squared))

	running_cost_k = np.cumsum(mean_cum_cost_k)
	running_cost_adp = np.cumsum(mean_cum_cost_adp)

	plt.rcParams['lines.linewidth'] = 3.0
	plt.rcParams['grid.alpha'] = 0.3
	
	plt.plot(np.arange(len(running_cost_k)),running_cost_k, label='Algorithm in [3]', color='red',linestyle='dotted')
	fill=plt.fill_between(np.arange(len(running_cost_k)), running_cost_k-running_std_k, running_cost_k+running_std_k, alpha=0.3, color='red',edgecolor='red')
	plt.rcParams['hatch.linewidth'] = 2.0
	plt.rcParams['hatch.color'] = 'black'
	fill.set_hatch('////')
	plt.xlabel('Time Horizon T', fontsize=16)
	plt.ylabel('Mean Cumulative Cost', fontsize=16)
	plt.plot(np.arange(len(running_cost_adp)),running_cost_adp, label='Approximate Dynamic Program', color='blue',linestyle='-.')
	fill2=plt.fill_between(np.arange(len(running_cost_adp)), running_cost_adp-running_std_adp, running_cost_adp+running_std_adp, alpha=0.3, color='blue', edgecolor='blue')
	fill2.set_hatch('..')
	plt.legend(fontsize=16)

	tick_positions = [i for i in range(10)]

	tick_labels = [str(i+1) for i in range(10)]

	plt.xticks(tick_positions,tick_labels)
	plt.grid(True)
	plt.tick_params(axis='both', which='major', labelsize=18)  # Set major tick label size
	plt.tick_params(axis='both', which='minor', labelsize=16)  # Set minor tick label size (if applicable)

	plt.savefig(path+'Cum-Cost-Comparison.pdf')



	
	return



def plot_single_run(run_number, initial_fatigue_state):

	result_path = os.path.join('results_single_run', 'run_'+str(run_number),initial_fatigue_state)
	
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
	
	# Plot taskload comparison
	
	with open(os.path.join(result_path,'fatigue_evolve_adp.pkl') , 'rb') as f1:
		data_adp = pickle.load(f1)
	with open(os.path.join(result_path,'fatigue_evolve_k.pkl'), 'rb') as f2:
		data_k = pickle.load(f2)
		
	stime = 10
	sim_time = np.arange(1, stime+1)
	plt.rcParams['lines.linewidth'] = 3.0
	plt.rcParams['grid.alpha'] = 0.3
	ax1.step(sim_time, data_adp[:stime], label='ADP', color='blue', linestyle='-.', where='mid')
	ax1.step(sim_time, data_k[:stime], label='Algorithm in [3]', color='red', where='mid',linestyle='dotted')
	ax1.set_xlabel('Time Horizon T', fontsize=16)
	ax1.set_ylabel('Fatigue', fontsize=16)
	ax1.grid(True)
	ax1.legend(fontsize=14)
	ax1.tick_params(labelsize=14)
	
	# Plot fatigue comparison
	with open(os.path.join(result_path,'taskload_evolve_adp.pkl'), 'rb') as f3:
		fatigue_adp = pickle.load(f3)
	with open(os.path.join(result_path,'taskload_evolve_k.pkl'), 'rb') as f4:
		fatigue_k = pickle.load(f4)
		
	ax2.step(sim_time, fatigue_adp[:stime], label='ADP', color='blue', linestyle='-.', where='mid')
	ax2.step(sim_time, fatigue_k[:stime], label='Algorithm in [3]', color='red', where='mid',linestyle='dotted')
	ax2.set_xlabel('Time Horizon T', fontsize=16) 
	ax2.set_ylabel('Taskload', fontsize=16)

	ax2.grid(True)
	ax2.legend(fontsize=14)
	ax2.tick_params(labelsize=14)
	
	plt.tight_layout()
	plt.savefig('results_single_run/' + f'Run_{run_number}_Comparison_{initial_fatigue_state}.pdf')
	plt.close()





if __name__=='__main__':

	fatigue_model = 'fatigue_model_1'
	# create_complete_performance_table(fatigue_model)


      
	result_path = 'results/'
	
	num_tasks=20

	horizon=10
	
	alpha_tp = 0.2
	alpha_fp = 0.3
	beta_tp = 0.1
	beta_fp = 0.1
	gamma_tp = 2.3
	gamma_fp = 3


	#plot_taskload_comparison(result_path, num_tasks, alpha_tp,alpha_fp, beta_tp, beta_fp, gamma_tp, gamma_fp, stime=10,computation='mean',fatigue_model='fatigue_model_1')
	#plot_fatigue_comparison(result_path, num_tasks, alpha_tp,alpha_fp, beta_tp, beta_fp, gamma_tp, gamma_fp, stime=10, computation='mean',fatigue_model='fatigue_model_1')

	plot_performance(result_path, num_tasks, alpha_tp,alpha_fp, beta_tp, beta_fp, gamma_tp, gamma_fp,fatigue_model='fatigue_model_1')
    
	# compare_cum_cost(result_path, num_tasks, alpha_tp, alpha_fp, beta_tp, beta_fp, gamma_tp, gamma_fp,fatigue_model='fatigue_model_1')

	# run_number = list(range(1,11))
	# initial_fatigue_state = ['fatigue_low', 'fatigue_high']

	# for run in run_number:

	# 	for initial_fatigue in initial_fatigue_state:
	# 		plot_single_run(run,initial_fatigue)