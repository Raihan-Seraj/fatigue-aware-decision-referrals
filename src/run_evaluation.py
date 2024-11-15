import matplotlib.pyplot as plt 
import os
import numpy as np 
from tqdm import tqdm
from utils import Utils 
import matplotlib
import pickle
from envs.fatigue_model_1 import FatigueMDP
from envs.fatigue_model_2 import FatigueMDP2
import contextlib
matplotlib.use('Agg')


class Evaluations(object):
    def __init__(self,args):

        self.args = args
        
        self.model_name = self.args.model_name

        if self.model_name.lower()=='fatigue_model_1':

            self.env = FatigueMDP()
        elif self.model_name.lower()=='fatigue_model_2':
            self.env = FatigueMDP2()
        else:
            raise ValueError("Invalid fatigue model name")

        self.num_tasks_per_batch = args.num_tasks_per_batch

        
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

        all_cost = []

        f_t_evol = []
        all_cstars = []

        for w_t in range(self.num_tasks_per_batch+1):

            if self.model_name.lower()=='fatigue_model_1':

                w_t_discrete = ut.discretize_taskload(w_t)

                F_tp1 = self.env.next_state(F_t,w_t_discrete)
            
            elif self.model_name.lower()=='fatigue_model_2':

                _, F_tp1 = self.env.next_state(F_t,w_t)
            
            else:

                raise ValueError("Invalid fatigue model name")
            
            

            f_t_evol.append(F_tp1)
            
            cstar, deferred_indices, gbar = ut.compute_cstar(
                F_t,
                w_t,
                batched_posterior_h0,
                batched_posterior_h1
            )
            
            ## fixme
            future_cost = V_bar[F_tp1]

            total_cost = cstar + future_cost
            
            all_cost.append(total_cost)
            all_cstars.append(cstar)

            all_deferred_indices.append(deferred_indices)

        
        min_idx = np.argmin(all_cost)

        #min_cost = all_cost[min_idx]

        wl_adp = min_idx
        
        defrred_idx_adp = all_deferred_indices[min_idx]

        return wl_adp, defrred_idx_adp


    def compute_performance(self):
        
        
        print("Computing peformance")

        
        ut = Utils(self.args)
        
        

        V_bar = np.load(self.args.results_path + 'num_tasks '+str(self.args.num_tasks_per_batch)+'/alpha '+str(self.args.alpha)+'/beta '+str(self.args.beta)+'/gamma '+str(self.args.gamma)+'/V_bar.npy')

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


                _, batched_posterior_h0, batched_posterior_h1=mega_batch[t]

                wl_k, deferred_idx_k = ut.compute_kesav_policy(F_k,batched_posterior_h0, batched_posterior_h1)

                wl_adp, deferred_idx_adp = self.compute_adp_solution(batched_posterior_h0,batched_posterior_h1,F_adp, V_bar,ut)

                
                

                

                hum_wl_adp[t]=wl_adp
                hum_wl_k[t]=wl_k
                
                
                a_cost_adp, h_cost_adp,def_cost_adp = ut.per_step_cost(F_adp,batched_posterior_h1,deferred_idx_adp)

                a_cost_k, h_cost_k, def_cost_k = ut.per_step_cost(F_k,batched_posterior_h1,deferred_idx_k)

                
                
                
                auto_cost_adp+=a_cost_adp
                human_cost_adp+=h_cost_adp
                deferred_cost_adp += def_cost_adp

                auto_cost_k += a_cost_k
                human_cost_k += h_cost_k
                deferred_cost_k += def_cost_k

                
                if self.model_name.lower()=='fatigue_model_1':

                    #get the next fatigue state for kesav
                    wl_k_discrete = ut.discretize_taskload(wl_k)
                    F_k = self.env.next_state(F_k, wl_k_discrete)

                    #get the next fatigue state for adp
                    wl_adp_discrete = ut.discretize_taskload(wl_adp)
                    F_adp = self.env.next_state(F_adp,wl_adp_discrete)
                
                elif self.model_name.lower()=='fatigue_model_2':

                    F_k,_ = self.env.next_state(F_k, wl_k)
                    F_adp,_ = self.env.next_state(F_adp,wl_adp)
                
                else:
                    raise ValueError("Invalid fatigue model")

                

            all_human_wl_adp['Run-'+str(run+1)]=hum_wl_adp
            all_human_wl_k['Run-'+str(run+1)]=hum_wl_k

        



            all_auto_cost_adp[run]= auto_cost_adp
            all_human_cost_adp[run] = human_cost_adp
            all_deferred_cost_adp[run] = deferred_cost_adp

            all_auto_cost_k[run]= auto_cost_k
            all_human_cost_k[run] = human_cost_k
            all_deferred_cost_k[run] = deferred_cost_k


            all_auto_cost_adp_new[run]= auto_cost_adp_new
            all_human_cost_adp_new[run] = human_cost_adp_new
            all_deferred_cost_adp_new[run] = deferred_cost_adp_new
            

             

        path1 = self.args.results_path + 'num_tasks '+str(self.args.num_tasks_per_batch)+'/alpha '+str(self.args.alpha)+'/beta '+str(self.args.beta)+'/gamma '+str(self.args.gamma)+'/plot_analysis/cost_comparison/'
        if not os.path.exists(path1):
            with contextlib.suppress(FileExistsError):
                os.makedirs(path1,exist_ok=True)
        
        

        
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


        
        return




   

        







    def run_evaluation(self):
        

        ut = Utils(self.args)

       
        V_bar = np.load(self.args.results_path + 'num_tasks '+str(self.args.num_tasks_per_batch)+'/alpha '+str(self.args.alpha)+'/beta '+str(self.args.beta)+'/gamma '+str(self.args.gamma)+'/V_bar.npy')


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

            wl_k, deferred_idx_k = ut.compute_kesav_policy(F_k,batched_posterior_h0, batched_posterior_h1)

            wl_adp, deferred_idx_adp = self.compute_adp_solution(batched_posterior_h0,batched_posterior_h1,F_adp, V_bar,ut)


            if self.model_name.lower()=='fatigue_model_1':

                #get the next fatigue state for kesav
                wl_k_discrete = ut.discretize_taskload(wl_k)
                F_k =self.env.next_state(F_k, wl_k_discrete)

                #get the next fatigue state for adp
                wl_adp_discrete = ut.discretize_taskload(wl_adp)
                F_adp = self.env.next_state(F_adp,wl_adp_discrete)
            
            elif self.model_name.lower()=='fatigue_model_2':

                F_k,_ = self.env.next_state(F_k, wl_k)
                F_adp,_ = self.env.next_state(F_adp,wl_adp)
            
            else:
                raise ValueError("Invalid name for fatigue model")

            
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

        
        
        
        path_name = self.args.results_path + 'num_tasks '+str(self.args.num_tasks_per_batch)+'/alpha '+str(self.args.alpha)+'/beta '+str(self.args.beta)+'/gamma '+str(self.args.gamma)+'/plot_analysis/'

        if not os.path.exists(path_name):
            with contextlib.suppress(FileExistsError):
                os.makedirs(path_name,exist_ok=True)
        
        
        
        with open(path_name+'all_fatigue_k.pkl','wb') as file1:
            pickle.dump(all_fatigue_kesav,file1)

        with open(path_name+'all_fatigue_adp.pkl','wb') as file2:
            pickle.dump(all_fatigue_adp,file2)
        
        with open(path_name+'all_taskload_k.pkl','wb') as file3:
            pickle.dump(all_taskload_kesav,file3)
        
        with open(path_name+'all_taskload_adp.pkl','wb') as file4:
            pickle.dump(all_taskload_adp,file4)

        sample_path_choose = 3
        
        median_fatigue_k =  all_fatigue_kesav[sample_path_choose]#np.median(all_fatigue_kesav,axis=0)
        #std_fatigue_k_lower = np.percentile(all_fatigue_kesav,25,axis=0)
        #std_fatigue_k_higher = np.percentile(all_fatigue_kesav,75,axis=0)


        median_fatigue_adp = all_fatigue_adp[sample_path_choose]#np.median(all_fatigue_adp,axis=0)
        #std_fatigue_adp_lower = np.percentile(all_fatigue_adp,25,axis=0)
        #std_fatigue_adp_higher = np.percentile(all_fatigue_adp,75,axis=0)


        median_taskload_k = all_taskload_kesav[sample_path_choose]#np.median(all_taskload_kesav,axis=0)
        #std_taskload_k_lower = np.percentile(all_taskload_kesav,25,axis=0)
        #std_taskload_k_higher = np.percentile(all_taskload_kesav,75,axis=0)

        

        median_taskload_adp = all_taskload_adp[sample_path_choose]#np.median(all_taskload_adp,axis=0)
        #std_taskload_adp_lower = np.percentile(all_taskload_adp,25,axis=0)
        #std_taskload_adp_higher = np.percentile(all_taskload_adp,75,axis=0)


        plt.step(np.arange(1,self.args.horizon+1,1),median_fatigue_k,color='black',where='post',label='K-Algorithm')
        #plt.fill_between(np.arange(1,self.args.horizon+1,1), std_fatigue_k_lower, std_fatigue_k_higher,step='post', alpha=0.2, color='black')

        plt.step(np.arange(1,self.args.horizon+1,1),median_fatigue_adp,color='orange',where='post',label='ADP')
        #plt.fill_between(np.arange(1,self.args.horizon+1,1), std_fatigue_adp_lower, std_fatigue_adp_higher,step='post', alpha=0.2, color='orange')

        plt.grid(True)
        
        
        plt.xlabel('Time')
        plt.ylabel('Fatigue Level')
        plt.legend()
        plt.savefig(path_name+'beta_'+str(self.args.beta)+'_fatigue.pdf')

        plt.clf()
        plt.close()


        plt.step(np.arange(1,self.args.horizon+1,1),median_taskload_k,color='black',where='post',label='K-Algorithm')
        #plt.fill_between(np.arange(1,self.args.horizon+1,1), std_taskload_k_lower, std_taskload_k_higher,step='post', alpha=0.2, color='black')

        plt.step(np.arange(1,self.args.horizon+1,1),median_taskload_adp,color='orange',where='post',label='ADP')
        #plt.fill_between(np.arange(1,self.args.horizon+1,1), std_taskload_adp_lower, std_taskload_adp_higher,step='post', alpha=0.2, color='orange')
        
    
        plt.xlabel('Time')
        plt.ylabel('Workload level')
        plt.legend()
        plt.savefig(path_name+'beta_'+str(self.args.beta)+'_workload.pdf')

        plt.clf()
        plt.close()

        return

   


