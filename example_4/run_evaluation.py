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

        self.args = args

        if self.args.use_wandb:
            self.run_info = wandb.init(project="Example 4",settings=wandb.Settings(start_method="fork"),mode=self.args.wandb_sync)
        



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


    def compute_perf_multiprocess(self):
        
        if self.args.use_wandb:
            self.run_info.name= "beta "+str(self.args.beta)+' alpha '+str(self.args.alpha)+' gamma '+str(self.args.gamma)

        print("Computing peformance with beta = "+str(self.args.beta)+'\n')

        param_path = self.args.results_path + 'num_tasks '+str(self.args.num_tasks_per_batch)+'/beta '+str(self.args.beta)+'/alpha '+str(self.args.alpha)+'/gamma_'+str(self.args.gamma)+'/params.json'

        with open(param_path,'r') as file:
            params = json.load(file)
        
        
        H0 = params["H0"]

        H1 = params["H1"]

        
        ut = Utils(self.args, H0, H1)
        
        

        V_bar = np.load(self.args.results_path + 'num_tasks '+str(self.args.num_tasks_per_batch)+'/beta '+str(self.args.beta)+'/alpha '+str(self.args.alpha)+'/gamma_'+str(self.args.gamma)+'/V_bar.npy')

        num_runs = self.args.num_eval_runs

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

            

            auto_cost_adp = 0
            human_cost_adp = 0
            deferred_cost_adp = 0

            auto_cost_k = 0
            human_cost_k=0
            deferred_cost_k=0

            auto_cost_adp_new=0
            human_cost_adp_new=0
            deferred_cost_adp_new = 0

            mega_batch = [ut.get_auto_obs() for _ in range(self.args.horizon)]
            all_mega_batch['Run-'+str(run+1)]=mega_batch
            hum_wl_adp = np.zeros(self.args.horizon)
            hum_wl_k = np.zeros(self.args.horizon)
            for t in range(self.args.horizon):


                batched_obs, batched_posterior_h0, batched_posterior_h1=mega_batch[t]

                wl_k, deferred_idx_k = self.compute_kesavs_algo(batched_posterior_h0, batched_posterior_h1, F_k, ut)

                wl_adp, deferred_idx_adp = self.compute_adp_solution(batched_posterior_h0,batched_posterior_h1,F_adp, V_bar,ut)

                
                if self.args.use_wandb:

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

                a_cost_k, h_cost_k, def_cost_k = ut.per_step_cost(F_k,batched_posterior_h1,deferred_idx_k)

                
                
                if self.args.use_wandb:
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

                

                #get the next fatigue state for kesav
                F_k = ut.get_fatigue(F_k, wl_k)

                #get the next fatigue state for adp

                F_adp = ut.get_fatigue(F_adp,wl_adp)

                

            all_human_wl_adp['Run-'+str(run+1)]=hum_wl_adp
            all_human_wl_k['Run-'+str(run+1)]=hum_wl_k

            
            if self.args.use_wandb:
            
                self.run_info.log({'Run':run,'Run-'+str(run)+'-TotalAutoCost-K':auto_cost_k,
                                'Run-'+str(run)+'-TotalAutoCost-ADP':auto_cost_adp,
                                'Run-'+str(run)+'-TotalHumCost-K':human_cost_k,
                                'Run-'+str(run)+'-TotalHumCost-ADP':human_cost_adp,
                                'Run-'+str(run)+'-TotalCost-K':human_cost_k+auto_cost_k+deferred_cost_k,
                                'Run-'+str(run)+'-TotalCost-ADP':human_cost_adp+auto_cost_adp+deferred_cost_adp})
                
                self.run_info.log({'All-Run-AutoCost-K':auto_cost_k,
                                'All-Run-AutoCost-ADP':auto_cost_adp,
                                'All-Run-HumCost-K':human_cost_k,
                                'All-Run-HumCost-ADP':human_cost_adp,
                                'All-Run-DeffCost-K':deferred_cost_k,
                                'All-Run-DeffCost-ADP':deferred_cost_adp,
                                'All-Run-TotalCost-K':auto_cost_k+human_cost_k+deferred_cost_k,
                                'All-Run-TotalCost-ADP': auto_cost_adp+human_cost_adp+deferred_cost_adp})
            

            



            all_auto_cost_adp[run]= auto_cost_adp
            all_human_cost_adp[run] = human_cost_adp
            all_deferred_cost_adp[run] = deferred_cost_adp

            all_auto_cost_k[run]= auto_cost_k
            all_human_cost_k[run] = human_cost_k
            all_deferred_cost_k[run] = deferred_cost_k


            all_auto_cost_adp_new[run]= auto_cost_adp_new
            all_human_cost_adp_new[run] = human_cost_adp_new
            all_deferred_cost_adp_new[run] = deferred_cost_adp_new
            

             

        path1 = self.args.results_path + 'num_tasks '+str(self.args.num_tasks_per_batch)+'/beta '+str(self.args.beta)+'/alpha '+str(self.args.alpha)+'/gamma_'+str(self.args.gamma)+'/plot_analysis/cost_comparison/'
        if not os.path.exists(path1):
            try:
                os.makedirs(path1,exist_ok=True)
            except FileExistsError:
                pass
        
        

        
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


        if self.args.use_wandb:
            self.run_info.finish()
        
        return




    def compute_performance(self):

               
        
        self.compute_perf_multiprocess()

        







    def run_evaluation(self):

        ## loading the parameters

        param_path = self.args.results_path + 'num_tasks '+str(self.args.num_tasks_per_batch)+'/beta '+str(self.args.beta)+'/alpha '+str(self.args.alpha)+'/gamma_'+str(self.args.gamma)+'/params.json'

        with open(param_path,'r') as file:
            params = json.load(file)
        
        
        
        H0 = params["H0"]

        H1 = params["H1"]

        
        

        ut = Utils(self.args, H0, H1)

       
        V_bar = np.load(self.args.results_path + 'num_tasks '+str(self.args.num_tasks_per_batch)+'/beta '+str(self.args.beta)+'/alpha '+str(self.args.alpha)+'/gamma_'+str(self.args.gamma)+'/V_bar.npy')


        ## initial fatigue is for kesav 0
        F_k = 0

        #initial fatigue for adp
        F_adp = 0

     
        fatigue_evolution_kesav = []
        fatigue_evolution_adp = []
        

        taskload_evolution_kesav = []
        taskload_evolution_adp =[]
        

        mega_obs = [ut.get_auto_obs() for _ in range(self.args.horizon)]
        
        for t in range(self.args.horizon):

            fatigue_evolution_kesav.append(F_k)
            fatigue_evolution_adp.append(F_adp)
            

            batched_obs, batched_posterior_h0, batched_posterior_h1= mega_obs[t]

            wl_k, deferred_idx_k = self.compute_kesavs_algo(batched_posterior_h0, batched_posterior_h1, F_k, ut)

            wl_adp, deferred_idx_adp = self.compute_adp_solution(batched_posterior_h0,batched_posterior_h1,F_adp, V_bar,ut)



            #get the next fatigue state for kesav
            F_k = ut.get_fatigue(F_k, wl_k)

            #get the next fatigue state for adp

            F_adp = ut.get_fatigue(F_adp,wl_adp)

            
            taskload_evolution_adp.append(wl_adp)
            
            taskload_evolution_kesav.append(wl_k)
        
        return fatigue_evolution_kesav,fatigue_evolution_adp, taskload_evolution_kesav, taskload_evolution_adp




    def run_perf_eval(self):
        
        num_runs = self.args.num_eval_runs

        all_fatigue_kesav = []
        all_fatigue_adp = []
        all_taskload_kesav = []
        all_taskload_adp = []

        for run in tqdm(range(num_runs)): 
        
            fatigue_evolution_kesav,fatigue_evolution_adp, taskload_evolution_kesav, taskload_evolution_adp= self.run_evaluation()

            all_fatigue_kesav.append(fatigue_evolution_kesav)
            all_fatigue_adp.append(fatigue_evolution_adp)
            all_taskload_kesav.append(taskload_evolution_kesav)
            all_taskload_adp.append(taskload_evolution_adp)
            ## plotting the level of fatigue 

        
        
        
        path_name = self.args.results_path + 'num_tasks '+str(self.args.num_tasks_per_batch)+'/beta '+str(self.args.beta)+'/alpha '+str(self.args.alpha)+'/gamma_'+str(self.args.gamma)+'/plot_analysis/'

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


        plt.step(np.arange(1,self.args.horizon+1,1),mean_fatigue_k,color='black',where='post',label='K-Algorithm')
        plt.fill_between(np.arange(1,self.args.horizon+1,1), mean_fatigue_k-std_fatigue_k, mean_fatigue_k+std_fatigue_k,step='post', alpha=0.2, color='black')

        plt.step(np.arange(1,self.args.horizon+1,1),mean_fatigue_adp,color='orange',where='post',label='ADP')
        plt.fill_between(np.arange(1,self.args.horizon+1,1), mean_fatigue_adp-std_fatigue_adp, mean_fatigue_adp+std_fatigue_adp,step='post', alpha=0.2, color='orange')

        plt.grid(True)
        
        
        plt.xlabel('Time')
        plt.ylabel('Fatigue Level')
        plt.legend()
        plt.savefig(path_name+'beta_'+str(self.args.beta)+'_fatigue.pdf')

        plt.clf()
        plt.close()


        plt.step(np.arange(1,self.args.horizon+1,1),mean_taskload_k,color='black',where='post',label='K-Algorithm')
        plt.fill_between(np.arange(1,self.args.horizon+1,1), mean_taskload_k-std_taskload_k, mean_taskload_k+std_taskload_k,step='post', alpha=0.2, color='black')

        plt.step(np.arange(1,self.args.horizon+1,1),mean_taskload_adp,color='orange',where='post',label='ADP')
        plt.fill_between(np.arange(1,self.args.horizon+1,1), mean_taskload_adp-std_taskload_adp, mean_taskload_adp+std_taskload_adp,step='post', alpha=0.2, color='orange')
        
    
        plt.xlabel('Time')
        plt.ylabel('Workload level')
        plt.legend()
        plt.savefig(path_name+'beta_'+str(self.args.beta)+'_workload.pdf')

        plt.clf()
        plt.close()

        return

   


