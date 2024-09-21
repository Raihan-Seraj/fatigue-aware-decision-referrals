import matplotlib.pyplot as plt 
import os
import numpy as np 
from tqdm import tqdm
import json 
import pandas as pd
from utils import Utils 
import seaborn as sns
import matplotlib
import multiprocessing
import pickle
import wandb
matplotlib.use('Agg')
wandb.require("legacy-service")

class Evaluations(object):
    def __init__(self,args):

        self.run_info = wandb.init(project="Example 1",settings=wandb.Settings(start_method="fork"),)
        self.args = args



    '''

    Function that computes Kesav's algorithm

    Args:
        batched_obs: A batch of observation for the automation
        batched_posterior_h0: A list consisting of posterior value of H0 given observation. 
        batched_posterior_h1: A list consisting of posterior value of H1 given observation. 
        F_t: The fatigue state at time t in the range [0.1]
        ut: a utility object from the class Utility

    Returns:
        optimal workload for the human

    '''

    def compute_kesavs_algo(self,
        batched_posterior_h0,
        batched_posterior_h1,
        F_t,
        ut
    ):

        all_cost = []
    
        all_deferred_indices =[]

        for w_t in range(ut.num_tasks_per_batch+1):

            cstar, deferred_indices, gbar = ut.compute_cstar(
                F_t,
                w_t,
                batched_posterior_h0,
                batched_posterior_h1,
            
            )
    

        
        

            all_cost.append(cstar)
            

            all_deferred_indices.append(deferred_indices)
        
    
        min_idx = np.argmin(all_cost)

        min_cost = all_cost[min_idx]

        deferred_idx_dp = all_deferred_indices[min_idx]

        w_star = min_idx

        return w_star, deferred_idx_dp


    '''
    Function that computes the optimal workload using the adp solution
    '''
    def compute_adp_solution(self,
        batched_posterior_h0,
        batched_posterior_h1,
        F_t,
        V_bar,
        ut
    ):

        all_cost = []

        all_deferred_indices=[]

        F_t_idx = ut.discretize_fatigue_state(F_t)

        all_cost = []

        f_t_evol = []
        all_cstars = []

        for w_t in range(ut.num_tasks_per_batch+1):

            F_tp1 = ut.get_fatigue(F_t, w_t)
            
            F_tp1_idx = ut.discretize_fatigue_state(F_tp1)

            f_t_evol.append(F_tp1)
            
            cstar, deferred_indices, gbar = ut.compute_cstar(
                F_t,
                w_t,
                batched_posterior_h0,
                batched_posterior_h1
            )
            
            ## fixme
            future_cost = V_bar[F_tp1_idx]

            total_cost = cstar + future_cost
            
            all_cost.append(total_cost)
            all_cstars.append(cstar)

            all_deferred_indices.append(deferred_indices)

        
        min_idx = np.argmin(all_cost)

        min_cost = all_cost[min_idx]

        wl_dp = min_idx
        
        defrred_idx_dp = all_deferred_indices[min_idx]

        return wl_dp, defrred_idx_dp


    def compute_perf_multiprocess(self,beta,result_path, lamda_new, simulation_time, num_runs,num_tasks_per_batch):

        self.run_info.name= "beta "+str(beta)+' mu '+str(self.args.mu)+' lambda '+str(self.args.lamda)

        print("Computing peformance with beta = "+str(beta)+'\n')

        param_path = result_path + 'num_tasks '+str(num_tasks_per_batch)+'/beta '+str(beta)+'/mu_'+str(self.args.mu)+'_lambda_'+str(self.args.lamda)+'/params.json'

        with open(param_path,'r') as file:
            params = json.load(file)
        
        num_tasks_per_batch = params["num_tasks_per_batch"]

        mu = params["mu"]

        lamda = params["lamda"]

        H0 = params["H0"]

        H1 = params["H1"]

        prior = params["prior"]

        d_0 = params["d_0"]

        beta = params["beta"]

        sigma_h = params["sigma_h"]
        sigma_a = params["sigma_a"]

        ctp = params["ctp"]
        ctn = params["ctn"]
        cfp = params["cfp"]
        cfn = params["cfn"]
        cm = params["cm"]
        num_bins_fatigue = params["num_bins_fatigue"]
        T = params["T"]
        num_expecation_samples = params["num_expectation_samples"]

        w_0 = params["w_0"]

        ut = Utils(num_tasks_per_batch, mu, lamda, w_0, sigma_a, H0, H1, prior, d_0, beta, sigma_h, ctp, ctn, cfp, cfn, cm, num_bins_fatigue)
        
        #initializing the utility with a different value of lambda (different than the one used for training)
        ut_new = Utils(num_tasks_per_batch, mu, lamda_new, w_0, sigma_a, H0, H1, prior, d_0, beta, sigma_h, ctp, ctn, cfp, cfn, cm, num_bins_fatigue)

        ut_k = Utils(num_tasks_per_batch, mu, lamda, w_0, sigma_a, H0, H1, prior, d_0, beta, sigma_h, ctp, ctn, cfp, cfn,cm, num_bins_fatigue)


        V_bar = np.load(result_path + 'num_tasks '+str(num_tasks_per_batch)+'/beta '+str(beta)+'/mu_'+str(self.args.mu)+'_lambda_'+str(self.args.lamda)+'/V_bar.npy')

        all_auto_cost_k = np.zeros(num_runs)
        all_human_cost_k = np.zeros(num_runs)
        all_deferred_cost_k = np.zeros(num_runs)

        all_auto_cost_adp = np.zeros(num_runs)
        all_human_cost_adp = np.zeros(num_runs)
        all_deferred_cost_adp = np.zeros(num_runs)
        
        
        all_auto_cost_adp_new = np.zeros(num_runs)
        all_human_cost_adp_new = np.zeros(num_runs)
        all_deferred_cost_adp_new = np.zeros(num_runs)

        all_human_wl_adp = {}
        all_human_wl_k={}

        all_mega_batch = {}

        for run in tqdm(range(num_runs)):
        
            ## initial fatigue is for kesav 0
            F_k = 0

            #initial fatigue for adp
            F_adp = 0

            F_adp_new=0

            auto_cost_adp = 0
            human_cost_adp = 0
            deferred_cost_adp = 0

            auto_cost_k = 0
            human_cost_k=0
            deferred_cost_k=0

            auto_cost_adp_new=0
            human_cost_adp_new=0
            deferred_cost_adp_new = 0

            mega_batch = [ut.get_auto_obs() for _ in range(simulation_time)]
            all_mega_batch['Run-'+str(run+1)]=mega_batch
            hum_wl_adp = np.zeros(simulation_time)
            hum_wl_k = np.zeros(simulation_time)
            for t in range(simulation_time):


                batched_obs, batched_posterior_h0, batched_posterior_h1=mega_batch[t]

                wl_k, deferred_idx_k = self.compute_kesavs_algo(batched_posterior_h0, batched_posterior_h1, F_k, ut_k)

                wl_adp, deferred_idx_adp = self.compute_adp_solution(batched_posterior_h0,batched_posterior_h1,F_adp, V_bar,ut)

                wl_adp_new, deferred_idx_adp_new = self.compute_adp_solution(batched_posterior_h0,batched_posterior_h1,F_adp_new, V_bar,ut_new)

                self.run_info.log({'Run-'+str(run)+'-t-'+str(t)+'-batched_obs':batched_obs,
                                   'Run-'+str(run)+'-t-'+str(t)+'-batched_posterior_h0':batched_posterior_h0,
                                   'Run-'+str(run)+'-t-'+str(t)+'-batched_posterior_h1':batched_posterior_h1,
                                   'Run-'+str(run)+'-t-'+str(t)+'-workload_K':wl_k,
                                   'Run-'+str(run)+'-t-'+str(t)+'-workload_ADP':wl_adp,
                                   'Run-'+str(run)+'-t-'+str(t)+'-deferred_idx_K':deferred_idx_k,
                                   'Run-'+str(run)+'-t-'+str(t)+'-deferred_idx_ADP':deferred_idx_adp})

                

                hum_wl_adp[t]=wl_adp
                hum_wl_k[t]=wl_k
                
                
                a_cost_adp, h_cost_adp,def_cost_adp = ut.per_step_cost(F_adp,batched_posterior_h1,deferred_idx_adp)

                a_cost_k, h_cost_k, def_cost_k = ut_k.per_step_cost(F_k,batched_posterior_h1,deferred_idx_k)

                a_cost_adp_new, h_cost_adp_new, def_cost_adp_new = ut_new.per_step_cost(F_adp_new,batched_posterior_h1,deferred_idx_adp_new)
                

                self.run_info.log({'Run-'+str(run)+'-t-'+str(t)+'-AutoCost-K':a_cost_k,
                                   'Run-'+str(run)+'-t-'+str(t)+'-AutoCost-ADP':a_cost_adp,
                                   'Run-'+str(run)+'-t-'+str(t)+'-HumCost-K':h_cost_k,
                                   'Run-'+str(run)+'-t-'+str(t)+'-HumCost-ADP':h_cost_adp,
                                   'Run-'+str(run)+'-t-'+str(t)+'-DefferedCost-K':def_cost_k,
                                   'Run-'+str(run)+'-t-'+str(t)+'-DeferredCost-ADP':def_cost_adp,
                                   'Run-'+str(run)+'-t-'+str(t)+'-Fatigue-K':F_k,
                                   'Run-'+str(run)+'-t-'+str(t)+'-Fatigue-ADP':F_adp
                                   })



                self.run_info.log({'Steps-Run-'+str(run): t, 'Batched-Obs':batched_obs, 'Batched-Posterior-H0': batched_posterior_h0,
                                   'Batched-Posterior-H1':batched_posterior_h1, 'Workload-K':wl_k, 'Workload-ADP':wl_adp,
                                   'Deferred-IDX-K':deferred_idx_k, 'Deferred-IDX-ADP':deferred_idx_adp, 'Fatigue-K':F_k, 'Fatigue-ADP':F_adp,
                                   'AutoCost-K':a_cost_k, 'AutoCost-ADP':a_cost_adp, 'HumCost-K':h_cost_k, 'HumCost-ADP':h_cost_adp, 
                                   'DeferredCost-K':def_cost_k, 'DeferredCost-ADP':def_cost_adp})

                auto_cost_adp+=a_cost_adp
                human_cost_adp+=h_cost_adp
                deferred_cost_adp += def_cost_adp

                auto_cost_k += a_cost_k
                human_cost_k += h_cost_k
                deferred_cost_k += def_cost_k

                auto_cost_adp_new+=a_cost_adp_new
                human_cost_adp_new += h_cost_adp_new
                deferred_cost_adp_new += def_cost_adp_new


                #get the next fatigue state for kesav
                F_k = ut_k.get_fatigue(F_k, wl_k)

                #get the next fatigue state for adp

                F_adp = ut.get_fatigue(F_adp,wl_adp)

                F_adp_new = ut_new.get_fatigue(F_adp_new,wl_adp_new)

            all_human_wl_adp['Run-'+str(run+1)]=hum_wl_adp
            all_human_wl_k['Run-'+str(run+1)]=hum_wl_k

            self.run_info.log({'Run':run,'Run-'+str(run)+'-TotalAutoCost-K':auto_cost_k,
                               'Run-'+str(run)+'-TotalAutoCost-ADP':auto_cost_adp,
                               'Run-'+str(run)+'-TotalHumCost-K':human_cost_k,
                               'Run-'+str(run)+'-TotalHumCost-ADP':human_cost_adp,
                               'Run-'+str(run)+'-TotalCost-K':human_cost_k+auto_cost_k+deferred_cost_k,
                               'Run-'+str(run)+'-TotalCost-ADP':human_cost_adp+auto_cost_adp+deferred_cost_adp})
            

            



            all_auto_cost_adp[run]= auto_cost_adp
            all_human_cost_adp[run] = human_cost_adp
            all_deferred_cost_adp[run] = deferred_cost_adp

            all_auto_cost_k[run]= auto_cost_k
            all_human_cost_k[run] = human_cost_k
            all_deferred_cost_k[run] = deferred_cost_k


            all_auto_cost_adp_new[run]= auto_cost_adp_new
            all_human_cost_adp_new[run] = human_cost_adp_new
            all_deferred_cost_adp_new[run] = deferred_cost_adp_new
            

             

        path1 = result_path + 'num_tasks '+str(num_tasks_per_batch)+'/beta '+str(beta)+'/mu_'+str(self.args.mu)+'_lambda_'+str(self.args.lamda)+'/plot_analysis/cost_comparison/'
        if not os.path.exists(path1):
            try:
                os.makedirs(path1,exist_ok=True)
            except FileExistsError:
                pass
        
        # path2 = path1+'cost_comparison/'

        # if not os.path.exists(path2):
        #     try:
        #         os.makedirs(path2,exist_ok=True)
        #     except FileExistsError:
        #         pass

        # path3 = path2+'beta '+str(beta)+'/'

        # if not os.path.exists(path3):
        #     try:
        #         os.makedirs(path3,exist_ok=True)
        #     except FileExistsError:
        #         pass


        
        with open(path1 + 'all_human_wl_adp.pkl','wb') as file:
            pickle.dump(all_human_wl_adp,file)

        
        with open(path1 + 'all_human_wl_k.pkl','wb') as file2:
            pickle.dump(all_human_wl_k,file2)
        

        with open(path1 + 'all_mega_batch.pkl','wb') as file3:
            pickle.dump(all_mega_batch,file3)

        
        np.save(path1+'all_auto_cost_adp.npy',all_auto_cost_adp)

        np.save(path1+'all_human_cost_adp.npy',all_human_cost_adp)

        np.save(path1+'all_deferred_cost_adp.npy',all_deferred_cost_adp)

        
        np.save(path1+'all_auto_cost_adp_new.npy',all_auto_cost_adp_new)

        np.save(path1+'all_human_cost_adp_new.npy',all_human_cost_adp_new)

        np.save(path1+'all_deferred_cost_adp_new.npy',all_deferred_cost_adp_new)


        np.save(path1+'all_auto_cost_k.npy',all_auto_cost_k)

        np.save(path1+'all_human_cost_k.npy',all_human_cost_k)

        np.save(path1+'all_deferred_cost_k.npy',all_deferred_cost_k)


        self.run_info.finish()
        
        return




    def compute_performance(self,beta,result_path,lamda_new, simulation_time,num_tasks_per_batch, num_runs=10):

               
        
        self.compute_perf_multiprocess(beta,result_path,lamda_new,simulation_time, num_runs, num_tasks_per_batch)

        







    def run_evaluation(self,beta, result_path,lamda_new, simulation_time,num_tasks_per_batch):

        ## loading the parameters

        param_path = result_path+'num_tasks '+str(num_tasks_per_batch)+'/beta '+str(beta)+'/mu_'+str(self.args.mu)+'_lambda_'+str(self.args.lamda)+'/params.json'

        with open(param_path,'r') as file:
            params = json.load(file)
        
        num_tasks_per_batch = params["num_tasks_per_batch"]

        mu = params["mu"]

        lamda = params["lamda"]

        H0 = params["H0"]

        H1 = params["H1"]

        prior = params["prior"]

        d_0 = params["d_0"]

        beta = params["beta"]

        sigma_h = params["sigma_h"]
        sigma_a = params["sigma_a"]

        ctp = params["ctp"]
        ctn = params["ctn"]
        cfp = params["cfp"]
        cfn = params["cfn"]
        cm = params["cm"]
        num_bins_fatigue = params["num_bins_fatigue"]
        T = params["T"]
        num_expecation_samples = params["num_expectation_samples"]

        w_0 = params["w_0"]

        

        ut = Utils(num_tasks_per_batch, mu, lamda, w_0, sigma_a, H0, H1, prior, d_0, beta, sigma_h, ctp, ctn, cfp, cfn, cm, num_bins_fatigue)

        ut_k = Utils(num_tasks_per_batch, mu, lamda, w_0, sigma_a, H0, H1, prior, d_0, beta, sigma_h, ctp, ctn, cfp, cfn, cm, num_bins_fatigue)

        ut_new = Utils(num_tasks_per_batch, mu, lamda_new, w_0, sigma_a, H0, H1, prior, d_0, beta, sigma_h, ctp, ctn, cfp, cfn, cm, num_bins_fatigue)


        V_bar = np.load(result_path+ 'num_tasks '+str(num_tasks_per_batch)+'/beta '+str(beta)+'/mu_'+str(mu)+'_lambda_'+str(lamda)+'/V_bar.npy')


        ## initial fatigue is for kesav 0
        F_k = 0

        #initial fatigue for adp
        F_adp = 0

        #initial fatigue for the new model
        F_adp_new=0

        fatigue_evolution_kesav = []
        fatigue_evolution_adp = []
        fatigue_evolution_adp_new = []

        taskload_evolution_kesav = []
        taskload_evolution_adp =[]
        taskload_evolution_adp_new =[]

        mega_obs = [ut.get_auto_obs() for _ in range(simulation_time)]
        
        for t in tqdm(range(simulation_time)):

            fatigue_evolution_kesav.append(F_k)
            fatigue_evolution_adp.append(F_adp)
            fatigue_evolution_adp_new.append(F_adp_new)

            batched_obs, batched_posterior_h0, batched_posterior_h1= mega_obs[t]

            wl_k, deferred_idx_k = self.compute_kesavs_algo(batched_posterior_h0, batched_posterior_h1, F_k, ut_k)

            wl_adp, deferred_idx_adp = self.compute_adp_solution(batched_posterior_h0,batched_posterior_h1,F_adp, V_bar,ut)

            
            wl_adp_new, deferred_idx_adp_new = self.compute_adp_solution(batched_posterior_h0,batched_posterior_h1,F_adp_new, V_bar,ut_new)




            #get the next fatigue state for kesav
            F_k = ut_k.get_fatigue(F_k, wl_k)

            #get the next fatigue state for adp

            F_adp = ut.get_fatigue(F_adp,wl_adp)

            F_adp_new = ut_new.get_fatigue(F_adp_new,wl_adp_new)

            taskload_evolution_adp.append(wl_adp)
            taskload_evolution_adp_new.append(wl_adp_new)
            taskload_evolution_kesav.append(wl_k)
        
        return fatigue_evolution_kesav,fatigue_evolution_adp,fatigue_evolution_adp_new, taskload_evolution_kesav, taskload_evolution_adp, taskload_evolution_adp_new




    def run_perf_eval(self,beta, result_path,lamda_new,simulation_time,num_tasks_per_batch, num_runs=10):

        all_fatigue_kesav = []
        all_fatigue_adp = []
        all_taskload_kesav = []
        all_taskload_adp = []

        for run in range(num_runs): 
        
            fatigue_evolution_kesav,fatigue_evolution_adp,fatigue_evolution_adp_new, taskload_evolution_kesav, taskload_evolution_adp, taskload_evolution_adp_new = self.run_evaluation(beta,result_path,lamda_new,simulation_time,num_tasks_per_batch)

            all_fatigue_kesav.append(fatigue_evolution_kesav)
            all_fatigue_adp.append(fatigue_evolution_adp)
            all_taskload_kesav.append(taskload_evolution_kesav)
            all_taskload_adp.append(taskload_evolution_adp)
            ## plotting the level of fatigue 

        
        
        
        path_name = result_path+ 'num_tasks '+str(num_tasks_per_batch)+'/beta '+str(beta)+'/mu_'+str(self.args.mu)+'_lambda_'+str(self.args.lamda)+'/plot_analysis/'

        if not os.path.exists(path_name):
            try:
                os.makedirs(path_name,exist_ok=True)
            except FileExistsError:
                pass
        
        
        
        with open(path_name+'all_fatigue_k.pkl','wb') as file1:
            pickle.dump(all_fatigue_kesav,file1)

        with open(path_name+'all_fatigue_adp.pkl','wb') as file2:
            pickle.dump(all_fatigue_adp,file2)
        
        with open(path_name+'all_taskload_k.pkl','wb') as file3:
            pickle.dump(all_taskload_kesav,file3)
        
        with open(path_name+'all_taskload_adp.pkl','wb') as file4:
            pickle.dump(all_taskload_adp,file4)

        
        mean_fatigue_k = np.mean(all_fatigue_kesav,axis=0)
        std_fatigue_k = np.std(all_fatigue_kesav,axis=0)


        mean_fatigue_adp = np.mean(all_fatigue_adp,axis=0)
        std_fatigue_adp = np.std(all_fatigue_adp,axis=0)


        mean_taskload_k = np.mean(all_taskload_kesav,axis=0)
        std_taskload_k = np.std(all_taskload_kesav,axis=0)

        mean_taskload_adp = np.mean(all_taskload_adp,axis=0)
        std_taskload_adp = np.std(all_taskload_adp,axis=0)


        plt.step(np.arange(1,simulation_time+1,1),mean_fatigue_k,color='black',where='post',label='K-Algorithm')
        plt.fill_between(np.arange(1,simulation_time+1,1), mean_fatigue_k-std_fatigue_k, mean_fatigue_k+std_fatigue_k,step='post', alpha=0.2, color='black')

        plt.step(np.arange(1,simulation_time+1,1),mean_fatigue_adp,color='orange',where='post',label='ADP')
        plt.fill_between(np.arange(1,simulation_time+1,1), mean_fatigue_adp-std_fatigue_adp, mean_fatigue_adp+std_fatigue_adp,step='post', alpha=0.2, color='orange')

        plt.grid(True)
        
        
        plt.xlabel('Time')
        plt.ylabel('Fatigue Level')
        plt.legend()
        plt.savefig(path_name+'beta_'+str(beta)+'_fatigue.pdf')

        plt.clf()
        plt.close()


        plt.step(np.arange(1,simulation_time+1,1),mean_taskload_k,color='black',where='post',label='K-Algorithm')
        plt.fill_between(np.arange(1,simulation_time+1,1), mean_taskload_k-std_taskload_k, mean_taskload_k+std_taskload_k,step='post', alpha=0.2, color='black')

        plt.step(np.arange(1,simulation_time+1,1),mean_taskload_adp,color='orange',where='post',label='ADP')
        plt.fill_between(np.arange(1,simulation_time+1,1), mean_taskload_adp-std_taskload_adp, mean_taskload_adp+std_taskload_adp,step='post', alpha=0.2, color='orange')
        
    
        plt.xlabel('Time')
        plt.ylabel('Workload level')
        plt.legend()
        plt.savefig(path_name+'beta_'+str(beta)+'_workload.pdf')

        plt.clf()
        plt.close()

        return

   


