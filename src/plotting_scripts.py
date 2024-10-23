import numpy as np 
import pandas as pd 
import pickle 
import matplotlib.pyplot as plt 
import seaborn as sns 




def create_performance_table(alphas,betas, gammas, mus,lamdas,num_tasks_per_batch, result_path):

	result_path = result_path + 'num_tasks '+str(num_tasks_per_batch)+'/'

	total_num_tasks=num_tasks_per_batch

	round_decimal_places=3
      
	final_data = pd.DataFrame(columns=['Beta','Alpha','Mu','Lambda', 'Expected Total Human Cost-ADP', ' Expected Total Human Cost-K','Expected Human Cost Per Taskload-ADP','Expected Human Cost Per Taskload-K',
										'Expected Total Automation Cost-ADP','Expected Total Automation Cost-K', 'Expected Automation Cost Per Taskload-ADP', 'Expected Automation Cost Per Taskload-K',
										'Expected Total Deferred Cost-ADP','Expected Total Deferred Cost-K',
										'Expected Total Cost-ADP','Expected Total Cost-K', 'Expected Taskload of the Human-ADP','Expected Taskload of the Human-K'])



	for alpha in alphas:

		for beta in betas:

			for gamma in gammas:

				for mu in mus:

					for lamda in lamdas:

						all_run_human_cost_adp_path = result_path + 'alpha '+str(alpha)+'/beta '+str(beta)+'/gamma '+str(gamma)+'/mu_'+str(mu)+'_lambda_'+str(lamda)+'/plot_analysis/cost_comparison/all_human_cost_adp.npy'

						all_run_human_cost_k_path = result_path  + 'alpha '+str(alpha)+'/beta '+str(beta)+'/gamma '+str(gamma)+'/mu_'+str(mu)+'_lambda_'+str(lamda)+'/plot_analysis/cost_comparison/all_human_cost_k.npy'

						all_run_automation_cost_adp_path = result_path + 'alpha '+str(alpha)+'/beta '+str(beta)+'/gamma '+str(gamma)+'/mu_'+str(mu)+'_lambda_'+str(lamda)+'/plot_analysis/cost_comparison/all_auto_cost_adp.npy'

						all_run_automation_cost_k_path = result_path + 'alpha '+str(alpha)+'/beta '+str(beta)+'/gamma '+str(gamma)+'/mu_'+str(mu)+'_lambda_'+str(lamda)+'/plot_analysis/cost_comparison/all_auto_cost_k.npy'


						all_run_deferred_cost_k_path = result_path + 'alpha '+str(alpha)+'/beta '+str(beta)+'/gamma '+str(gamma)+'/mu_'+str(mu)+'_lambda_'+str(lamda)+'/plot_analysis/cost_comparison/all_deferred_cost_k.npy'
						all_run_deferred_cost_adp_path = result_path + 'alpha '+str(alpha)+'/beta '+str(beta)+'/gamma '+str(gamma)+'/mu_'+str(mu)+'_lambda_'+str(lamda)+'/plot_analysis/cost_comparison/all_deferred_cost_adp.npy'
						
						

						try:
							all_run_human_cost_adp = np.load(all_run_human_cost_adp_path)
							all_run_human_cost_k = np.load(all_run_human_cost_k_path)

							all_run_auto_cost_adp = np.load(all_run_automation_cost_adp_path)
							all_run_auto_cost_k =  np.load(all_run_automation_cost_k_path)

							all_run_deferred_cost_k = np.load(all_run_deferred_cost_k_path)
							all_run_deferred_cost_adp = np.load(all_run_deferred_cost_adp_path)
						except FileNotFoundError:

							
							print("File not found, for alpha "+str(alpha), ' beta '+str(beta)+' mu '+str(mu)+' lamda '+str(lamda))
							#continue 
							


						# now loading the taskload 
						all_run_human_tl_adp_path = result_path + 'alpha '+str(alpha)+'/beta '+str(beta)+'/gamma '+str(gamma)+'/mu_'+str(mu)+'_lambda_'+str(lamda)+'/plot_analysis/cost_comparison/all_human_wl_adp.pkl'

						all_run_human_tl_k_path = result_path + 'alpha '+str(alpha)+'/beta '+str(beta)+'/gamma '+str(gamma)+'/mu_'+str(mu)+'_lambda_'+str(lamda)+'/plot_analysis/cost_comparison/all_human_wl_k.pkl'

						
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


						

						new_row  = pd.DataFrame([[str(beta), str(alpha), str(mu), str(lamda), 
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




def create_complete_performance_table():

	result_path = 'results/'
	num_tasks_per_batch=20
	
	# beta = 0.5

	# alphas = [1.0, 2.0, 5.0, 7.0, 9.0]

	# mus = [0.1, 0.003, 0.05, 0.07]

	# lamdas = [0.1,0.03,0.07, 0.007]

	alphas = [0.8]
	betas = [0.01]
	gammas = [0.1]
	mus = [0.05]
	lamdas = [0.07]
	
	
	create_performance_table(alphas,betas,gammas, mus, lamdas, num_tasks_per_batch, result_path)

	return


def plot_taskload_comparison(result_path, num_tasks, beta, alpha, mu, lamda):

	path = result_path + 'num_tasks '+str(num_tasks)+'/beta '+str(beta)+'/alpha '+str(alpha)+'/mu_'+str(mu)+'_lambda_'+str(lamda)+'/plot_analysis/'

	adp_path = path + 'all_taskload_adp.pkl'

	k_path = path + 'all_taskload_k.pkl'

	## loding the files

	with open(adp_path, 'rb') as file1:

		data_adp = pickle.load(file1)
	
	with open(k_path, 'rb') as file2:
		data_k = pickle.load(file2)
	 
	median_workload_adp = np.median(data_adp, axis=0)

	q1_adp = np.percentile(data_adp, q = 25, axis=0)

	q3_adp = np.percentile(data_adp, q=75, axis=0)


	median_workload_k = np.median(data_k, axis=0)

	q1_k = np.percentile(data_k, q = 25, axis=0)

	q3_k = np.percentile(data_k, q=75, axis=0)


	simulation_time = np.arange(1, 21)

	
	plt.step(simulation_time, median_workload_adp, label='Approximate Dynamic Program', color='blue', where='mid')

	
	plt.fill_between(simulation_time, q1_adp, q3_adp, step='mid', alpha=0.2, color='blue')


	
	plt.step(simulation_time, median_workload_k, label='Algorithm 1 in [3]', color='black', where='mid')


	plt.fill_between(simulation_time, q1_k, q3_k, step='mid', alpha=0.2, color='black')

	plt.grid(True)
	plt.xlabel('Time Steps')
	plt.ylabel('Taskload')

	tick_positions = [i+1 for i in range(20)]

	tick_labels = [str(i+1) for i in range(20) ]

	plt.xticks(tick_positions,tick_labels)

	plt.legend()
	plt.savefig(path+'Median-Taskload-Plot-Comparison.pdf')
	plt.close()
	plt.clf()



def plot_fatigue_comparison(result_path, num_tasks, beta, alpha, mu, lamda):

	path = result_path + 'num_tasks '+str(num_tasks)+'/beta '+str(beta)+'/alpha '+str(alpha)+'/mu_'+str(mu)+'_lambda_'+str(lamda)+'/plot_analysis/'

	adp_path = path + 'all_fatigue_adp.pkl'

	k_path = path + 'all_fatigue_k.pkl'

	## loding the files

	with open(adp_path, 'rb') as file1:

		data_adp = pickle.load(file1)
	
	with open(k_path, 'rb') as file2:
		data_k = pickle.load(file2)
	 
	median_fatigue_adp = np.median(data_adp, axis=0)

	q1_adp = np.percentile(data_adp, q = 25, axis=0)

	q3_adp = np.percentile(data_adp, q=75, axis=0)


	median_fatigue_k = np.median(data_k, axis=0)

	q1_k = np.percentile(data_k, q = 25, axis=0)

	q3_k = np.percentile(data_k, q=75, axis=0)


	simulation_time = np.arange(1, 21)

	
	plt.step(simulation_time, median_fatigue_adp, label='Approximate Dynamic Program', color='blue', where='mid')

	
	plt.fill_between(simulation_time, q1_adp, q3_adp, step='mid', alpha=0.2, color='blue')


	
	plt.step(simulation_time, median_fatigue_k, label='Algorithm 1 in [3]', color='black', where='mid')


	plt.fill_between(simulation_time, q1_k, q3_k, step='mid', alpha=0.2, color='black')




	plt.grid(True)
	plt.xlabel('Time Steps')
	plt.ylabel('Fatigue')

	tick_positions = [i+1 for i in range(20)]

	tick_labels = [str(i+1) for i in range(20) ]

	plt.xticks(tick_positions,tick_labels)


	plt.legend()
	plt.savefig(path+'Median-Fatigue-Plot-Comparison.pdf')
	plt.close()
	plt.clf()


def plot_performance(result_path, num_tasks, beta, alpha, mu, lamda):

	path=result_path + 'num_tasks '+str(num_tasks)+'/beta '+str(beta)+'/alpha '+str(alpha)+'/mu_'+str(mu)+'_lambda_'+str(lamda)+'/plot_analysis/cost_comparison/'

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


	all_arrays = {'AutoCost-ADP':auto_cost_adp, 'AutoCost-Alg:1':auto_cost_k, 'HumanCost-ADP':human_cost_adp, 'HumanCost-Alg:1':human_cost_k, 'TotalCost-ADP':total_cost_adp, 'TotalCost-Alg:1':total_cost_k
			   
			   }

	df = pd.DataFrame(all_arrays,columns=['AutoCost-ADP','AutoCost-Alg:1', 'HumanCost-ADP','HumanCost-Alg:1','TotalCost-ADP','TotalCost-Alg:1'])
	
	df_tc= df[['TotalCost-ADP', 'TotalCost-Alg:1']].melt(var_name='Category', value_name='Costs')

	df_hc= df[['HumanCost-ADP', 'HumanCost-Alg:1']].melt(var_name='Category', value_name='Costs')

	df_ac= df[['AutoCost-ADP', 'AutoCost-Alg:1']].melt(var_name='Category', value_name='Costs')
	
	
	clr = sns.color_palette("hls", 8)

	figs, axes = plt.subplots(1,3,figsize=(15,5))

	sns.boxplot(x='Category', y='Costs',hue='Category',data=df_ac, ax=axes[0])
	sns.boxplot(x='Category', y='Costs',hue='Category',data=df_hc, ax=axes[1])
	sns.boxplot(x='Category', y='Costs',hue='Category',data=df_tc, ax=axes[2])
	plt.savefig(path+'BoxPlot-Cost-Comparison.pdf')

	return




if __name__=='__main__':

	create_complete_performance_table()


      
	# result_path = 'results/'
	
	# num_tasks=20
	
	# beta=0.5
	
	# alpha=1.0
	
	# mu=0.05
	
	# lamda=0.07


	# plot_taskload_comparison(result_path, num_tasks, beta, alpha, mu, lamda)
	# plot_fatigue_comparison(result_path, num_tasks, beta, alpha, mu, lamda)

	# plot_performance(result_path, num_tasks, beta, alpha, mu, lamda)
