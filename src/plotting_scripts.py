import numpy as np 
import pandas as pd 
import pickle 
import matplotlib.pyplot as plt 
import seaborn as sns 
import argparse
import os
import glob
from matplotlib.patches import Patch
plt.rcParams["figure.figsize"] = (6.4, 5.3)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')



def plot_taskload_comparison(result_path, stime=20,computation='median'):

	
	path = result_path+'/plot_analysis/' 


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


	



def plot_fatigue_comparison(result_path, stime=20,computation='median'):

	path = result_path+'/plot_analysis/' 
	
	
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


def plot_performance(result_path):

	path = result_path+'/cost_comparison/'	

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
	#'ADP': '#b2b2ff', 'Algorithm in [3]': '#f9bebb'	
	for i, box in enumerate(axes[0].patches):
		if i ==0:  # First category
			box.set_hatch('..')
			box.set_facecolor('#b2b2ff')
			box.set_edgecolor('blue')
		else:  # Second category 
			box.set_hatch('//')
			box.set_facecolor('#f9bebb')
			box.set_edgecolor('red')
	axes[0].tick_params(axis='both', labelsize=13)

	sns.boxplot(x='Category', y='Costs', data=df_hc, ax=axes[1], flierprops={"marker": "x"}) 
	
	for i, box in enumerate(axes[1].patches):
		if i==0:  # First category
			box.set_hatch('..')
			box.set_facecolor('#b2b2ff')
			box.set_edgecolor('blue')
		else:  # Second category
			box.set_hatch('//')
			box.set_facecolor('#f9bebb')
			box.set_facecolor('#f9bebb')
			box.set_edgecolor('red')
	axes[1].tick_params(axis='both', labelsize=13)

	sns.boxplot(x='Category', y='Costs', data=df_tc, ax=axes[2], flierprops={"marker": "x"})
	
		
	for i, box in enumerate(axes[2].patches):
		if i==0:  # First category
			box.set_hatch('..')
			box.set_facecolor('#b2b2ff')
			box.set_edgecolor('blue')
		else:  # Second category
			box.set_hatch('//')
			box.set_facecolor('#f9bebb')
			box.set_edgecolor('red')
	axes[2].tick_params(axis='both', labelsize=13)
	

	plt.savefig(path+'BoxPlot-Cost-Comparison.pdf')
	plt.clf()
	plt.close()

	return

def compare_cum_cost(result_path):
	

	
	path = result_path+'/cost_comparison/'
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
    
	
	plt.figure(figsize=(7.552, 3.84))
	plt.rcParams['lines.linewidth'] = 3
	plt.rcParams['grid.alpha'] = 0.3
	
	plt.plot(np.arange(len(running_cost_k)),running_cost_k, label='Algorithm in [3]', color='red',linestyle='dotted')
	fill=plt.fill_between(np.arange(len(running_cost_k)), running_cost_k-running_std_k, running_cost_k+running_std_k, alpha=0.3, color='red',edgecolor='red')
	plt.rcParams['hatch.linewidth'] = 2
	plt.rcParams['hatch.color'] = 'black'
	fill.set_hatch('////')
	plt.xlabel('Time Horizon', fontsize=11)
	plt.ylabel('Mean Cumulative Cost', fontsize=11)
	plt.plot(np.arange(len(running_cost_adp)),running_cost_adp, label='Approximate Dynamic Program', color='blue',linestyle='-.')
	fill2=plt.fill_between(np.arange(len(running_cost_adp)), running_cost_adp-running_std_adp, running_cost_adp+running_std_adp, alpha=0.3, color='blue', edgecolor='blue')
	fill2.set_hatch('..')
	plt.legend(fontsize=11)

	tick_positions = [i for i in range(10)]

	tick_labels = [str(i+1) for i in range(10)]

	plt.xticks(tick_positions,tick_labels)
	plt.grid(True)
	plt.tick_params(axis='both', which='major', labelsize=11)  # Set major tick label size
	plt.tick_params(axis='both', which='minor', labelsize=11)  # Set minor tick label size (if applicable)

	plt.savefig(path+'Cum-Cost-Comparison.pdf')



	
	return



def plot_single_run(run_number, initial_fatigue_state):

	result_path = os.path.join('results_single_run', 'run_'+str(run_number),initial_fatigue_state)
	
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
	
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



def plot_performance_perturbed_fatigue(result_path):
	
	true_fatigue_path = result_path+'/cost_comparison/'
	true_auto_cost_adp_path = os.path.join(true_fatigue_path ,'all_auto_cost_adp.npy')

	true_hum_cost_adp_path = os.path.join(true_fatigue_path  , 'all_human_cost_adp.npy')

	true_auto_cost_k_path = os.path.join(true_fatigue_path ,'all_auto_cost_k.npy')

	true_hum_cost_k_path = os.path.join(true_fatigue_path  , 'all_human_cost_k.npy')


	true_auto_cost_adp = np.load(true_auto_cost_adp_path)

	true_human_cost_adp = np.load(true_hum_cost_adp_path)

	true_auto_cost_k = np.load(true_auto_cost_k_path)
	true_human_cost_k = np.load(true_hum_cost_k_path)

	true_total_cost_adp = true_auto_cost_adp + true_human_cost_adp
	true_total_cost_k = true_auto_cost_k + true_human_cost_k

	perturbed_model = sorted(glob.glob('results_perturbed_fatigue/model_*'))[:5]

	all_arrays = {'Perturbed_Model':[], 'TotalCost-ADP':[], 'TotalCost in [3]':[]}

	all_arrays['Perturbed_Model'].append('Nominal System')

	all_arrays['TotalCost-ADP'].append(true_total_cost_adp)

	all_arrays['TotalCost in [3]'].append(true_total_cost_k)
	
	for idx, model in enumerate(perturbed_model):
		
		hum_cost_adp = np.load(os.path.join(model,'cost_comparison','all_human_cost_adp.npy'))
		auto_cost_adp = np.load(os.path.join(model,'cost_comparison','all_auto_cost_adp.npy'))

		total_cost_adp = hum_cost_adp + auto_cost_adp

		hum_cost_k = np.load(os.path.join(model,'cost_comparison','all_human_cost_k.npy'))
		auto_cost_k = np.load(os.path.join(model,'cost_comparison','all_auto_cost_k.npy'))

		total_cost_k = hum_cost_k + auto_cost_k

		all_arrays['Perturbed_Model'].append('System-'+str(idx+1))

		all_arrays['TotalCost-ADP'].append(total_cost_adp)

		all_arrays['TotalCost in [3]'].append(total_cost_k)
	
	
	# Prepare the data for plotting
	data = []
	for i in range(len(all_arrays['Perturbed_Model'])):
		model = all_arrays['Perturbed_Model'][i]
		# Add ADP costs
		data.extend([(model, cost, 'ADP') for cost in all_arrays['TotalCost-ADP'][i]])
		# Add costs from Algorithm in [3]
		data.extend([(model, cost, 'Algorithm in [3]') for cost in all_arrays['TotalCost in [3]'][i]])

	# Create DataFrame
	df = pd.DataFrame(data, columns=['Model', 'Total Cost', 'Algorithm'])
	
	plt.figure(figsize=(11, 6))

	plt.rcParams['boxplot.boxprops.linewidth'] = 2.0
	plt.rcParams['boxplot.medianprops.linewidth'] = 2.0
	plt.rcParams['boxplot.flierprops.linewidth'] = 2.0
	plt.rcParams['boxplot.whiskerprops.linewidth'] = 2.0
	plt.rcParams['boxplot.capprops.linewidth'] = 2.0
	# Create boxplot
	# Create boxplot with custom colors
	sns.boxplot(x='Model', y='Total Cost', hue='Algorithm', data=df, 
				palette={'ADP': '#b2b2ff', 'Algorithm in [3]': '#f9bebb'},flierprops={"marker": "x"})

	# Reduce width of boxes by setting width parameter
	
	for box in plt.gca().patches:
		
		if box.get_facecolor()[:-1] == (0.73578431372549, 0.73578431372549, 0.9622549019607844):  # Blue boxes
			box.set_hatch('..')
			box.set_edgecolor('blue')
			
		else:  # Red boxes  
			box.set_hatch('//')
			box.set_edgecolor('red')
		  # Set edge color to black for all boxes
		
	# Create custom legend handles with hatches
	legend_elements = [
		Patch(facecolor='#b2b2ff', hatch='..', edgecolor='blue', label='ADP'),
		Patch(facecolor='#f9bebb', hatch='//', edgecolor='red', label='Algorithm in [3]')
	]

	# Add spacing between groups
	plt.gca().set_xticklabels(plt.gca().get_xticklabels())
	for i, label in enumerate(plt.gca().get_xticklabels()):
		if i % 2 == 1:  # After every 2nd label
			plt.axvline(x=i+0.5, color='white', linewidth=2)
	plt.legend(handles=legend_elements, fontsize=14,loc='upper right')
	plt.xlabel('Model', fontsize=16)
	plt.ylabel('Total Cost', fontsize=16)
	plt.tick_params(axis='both', which='major', labelsize=14)
	plt.xticks()
	plt.grid(True)
	plt.tight_layout()
	plt.savefig('results_perturbed_fatigue/perturbed_model_comparison.pdf')
	plt.close()
	
		


	return




if __name__=='__main__':

	
      
	result_path = 'results/'
	
	num_tasks=20

	horizon=10
	
	alpha_tp = 0.2
	alpha_fp = 0.3
	beta_tp = 0.1
	beta_fp = 0.1
	gamma_tp = 2.3
	gamma_fp = 3


	plot_taskload_comparison(result_path,computation='mean')
	
	plot_fatigue_comparison(result_path,computation='mean')

	plot_performance(result_path)
    
	plot_performance_perturbed_fatigue(result_path)

	compare_cum_cost(result_path)

	run_number = list(range(1,11))
	initial_fatigue_state = ['fatigue_low', 'fatigue_high']

	for run in run_number:

		for initial_fatigue in initial_fatigue_state:
			plot_single_run(run,initial_fatigue)