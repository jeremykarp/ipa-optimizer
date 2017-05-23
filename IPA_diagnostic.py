__author__ = 'jeremykarp'
import pandas as pd
import numpy as np
import math
import random
import scipy.stats
import networkx as nx
import itertools
from collections import defaultdict
import copy
from pulp import *
from gurobipy import *
import time
from collections import deque
setParam("OutputFlag",0)


def formulate_clicks_bricks_lp(lp, demand_physical, accepted_online, inventory, ship_matrix, p, c):
    n=demand_physical.size
    demand_physical=np.around(demand_physical,decimals=3).flatten()
    accepted_online=np.around(accepted_online,decimals=3).flatten()
    i_dict = {i:'I_'+str(i) for i in range(n)} #Inventory nodes
    o_dict = {i:'O_'+str(i) for i in range(n)} #Online demand nodes
    p_dict = {i:'P_'+str(i) for i in range(n)} #Physical demand nodes
    i_o_vars = {}
    i_p_vars = {}
    c_o_vars = {}
    for i in range(n):
        c_o_vars[i]=lp.addVar(vtype=GRB.CONTINUOUS, name='CO_'+str(i), lb=0)
        i_p_vars[i]=lp.addVar(vtype=GRB.CONTINUOUS, name='IP_'+str(i),
                              lb=0)
        i_o_vars[i]={}
        for j in range(n):
            i_o_vars[i][j]=lp.addVar(vtype=GRB.CONTINUOUS, name='I'+str(i)+'_O'+str(j))
    lp.update()
    #First line is prices for physical demand
    #second line is prices - shipping cost
    #third line is cancel costs
    lp.setObjective(LinExpr([p[i] for i in range(n)]+\
                    [p[i]-ship_matrix[i][j] for i,j in itertools.product(range(n), repeat=2)]+\
                    [c[i] for i in range(n)],
                    [i_p_vars[i] for i in range(n)]+\
                    [i_o_vars[i][j] for i,j in itertools.product(range(n), repeat=2)]+\
                    [c_o_vars[i] for i in range(n)]),GRB.MAXIMIZE)
    for i in range(n):
        temp_coefficients = [1]
        temp_vars = [i_p_vars[i]]
        lp.addConstr(LinExpr(temp_coefficients,temp_vars) == min(demand_physical[i],inventory[i]),'Physical '+str(i))
    for i in range(n):
        temp_coefficients = [1 for j in range(n+1)]
        #temp_vars = [i_p_vars[i]]+[i_o_vars[j][i] for j in range(n)]
        temp_vars = [i_p_vars[i]]+[i_o_vars[i][j] for j in range(n)]
        lp.addConstr(LinExpr(temp_coefficients,temp_vars) <= inventory[i],'ConstraintsI '+str(i))
    for i in range(n):
        temp_coefficients = [1 for j in range(n+1)]
        temp_vars = [c_o_vars[i]]+[i_o_vars[j][i] for j in range(n)]
        lp.addConstr(LinExpr(temp_coefficients,temp_vars) == accepted_online[i],'ConstraintsO '+str(i))
    lp.update()
    return lp, i_o_vars, i_p_vars, c_o_vars



def update_clicks_bricks(lp, demand_physical, accepted_online, inventory, i_o_vars, i_p_vars, c_o_vars):
    n=accepted_online.size
    demand_physical=np.around(demand_physical,decimals=3).flatten()
    demand_online=np.around(accepted_online,decimals=3).flatten()
    for i in range(n):
        lp.remove(lp.getConstrByName('Physical '+str(i)))
        temp_coefficients = [1]
        temp_vars = [i_p_vars[i]]
        lp.addConstr(LinExpr(temp_coefficients,temp_vars) == min(demand_physical[i],inventory[i]),'Physical '+str(i))
    for i in range(n):
        lp.remove(lp.getConstrByName('ConstraintsI '+str(i)))
        lp.remove(lp.getConstrByName('ConstraintsO '+str(i)))
        temp_coefficients = [1 for j in range(n+1)]
        #temp_vars = [i_p_vars[i]]+[i_o_vars[j][i] for j in range(n)]
        temp_vars = [i_p_vars[i]]+[i_o_vars[i][j] for j in range(n)]
        lp.addConstr(LinExpr(temp_coefficients,temp_vars) <= inventory[i],'ConstraintsI '+str(i))
    for i in range(n):
        temp_coefficients = [1 for j in range(n+1)]
        temp_vars = [c_o_vars[i]]+[i_o_vars[j][i] for j in range(n)]
        lp.addConstr(LinExpr(temp_coefficients,temp_vars) == accepted_online[i],'ConstraintsO '+str(i))
    lp.update()
    return lp, i_o_vars, i_p_vars, c_o_vars




def solve_lp(lp):
    lp.optimize()
    return lp

def generate_demand(demand_physical_means,demand_physical_cov,demand_online_means,demand_online_cov):
    raw_demand_physical = np.around(np.random.multivariate_normal(demand_physical_means,demand_physical_cov,1),decimals=3)
    demand_physical = np.around(np.maximum(raw_demand_physical,np.zeros_like(raw_demand_physical)),decimals=0)
    raw_demand_online = np.around(np.random.multivariate_normal(demand_online_means,demand_online_cov,1),decimals=3)
    demand_online = np.around(np.maximum(raw_demand_online,np.zeros_like(raw_demand_online)),decimals=0).flatten()
    return demand_physical,demand_online

def generate_demand_poisson(demand_physical_means,demand_online_means):
    demand_physical = np.random.poisson(lam=demand_physical_means)
    demand_online = np.random.poisson(lam=demand_online_means)
    return demand_physical,demand_online

def generate_demand_physical_poisson(demand_physical_means):
    demand_physical = np.random.poisson(lam=demand_physical_means)
    return demand_physical


def determine_accepted_orders(demand_online,S,S_local):
    num_locations = demand_online.shape[0]
    online_orders = np.empty(0, dtype=int)
    last_order_local = {}
    for i in range(len(demand_online)):
        temp_orders = np.zeros(int(demand_online[i]), dtype=int)
        temp_orders.fill(int(i))
        last_order_local[i]=len(temp_orders)
        online_orders=np.append(online_orders,temp_orders)
    np.random.shuffle(online_orders)
    S_copy = S*1.0
    S_local_copy = S_local.copy()
    accepted_online_long = np.empty(0, dtype=int)
    for i in range(len(online_orders)):
        if S_local_copy[online_orders[i]]>=1 and S_copy>=1:
            S_copy -= 1
            S_local_copy[online_orders[i]]-=1
            accepted_online_long = np.append(accepted_online_long,online_orders[i])
        elif S_copy<1:
            break
    accepted_online = np.bincount(np.append(accepted_online_long,np.array(range(num_locations))))
    accepted_online = accepted_online-1
    if accepted_online.shape[0] == 0:
        accepted_online=np.zeros_like(demand_online)
    rejected_online = demand_online-accepted_online
    return accepted_online,rejected_online, S_copy, S_local_copy

def compute_prob_next(rejected_online):
    if rejected_online.sum()>0:
        prob_next_in_line = rejected_online/rejected_online.sum()
    else:
        prob_next_in_line = np.zeros_like(rejected_online)
    return prob_next_in_line

def compute_prob_last(accepted_online):
    if accepted_online.sum()>0:
        prob_last_accepted = accepted_online.astype(float)/accepted_online.sum()
    else:
        prob_last_accepted = np.zeros_like(accepted_online)
    return prob_last_accepted

def update_derivative_estimates(lp2,deriv_estimate_LP,deriv_estimate_LP_local,rejected_online,num_locations,
                                S_local_remaining,prob_next_in_line,prob_last_accepted,global_update,local_update, print_diagnostic=False,
                                demand_distribution=None):
    demand_distribution = prob_last_accepted if demand_distribution is None else demand_distribution
    for i in range(num_locations):
        if rejected_online.sum()>0:
            deriv_estimate_LP+=lp2.getConstrByName('ConstraintsO '+str(i)).getAttr('Pi')*demand_distribution[i]*global_update
        else:
            deriv_estimate_LP+=lp2.getConstrByName('ConstraintsO '+str(i)).getAttr('Pi')*demand_distribution[i]*.01*global_update
        if S_local_remaining[i]<1:
            #deriv_estimate_LP_local[i]+=lp2.getConstrByName('ConstraintsO '+str(i)).getAttr('Pi')*prob_next_in_line[i]*local_update
            deriv_estimate_LP_local[i]+=lp2.getConstrByName('ConstraintsO '+str(i)).getAttr('Pi')*local_update
            if print_diagnostic:
                print "type1",lp2.getConstrByName('ConstraintsO '+str(i)).getAttr('Pi')
        else:
            #deriv_estimate_LP_local[i]+=lp2.getConstrByName('ConstraintsO '+str(i)).getAttr('Pi')*prob_last_accepted[i]*.01*local_update
            deriv_estimate_LP_local[i]+=lp2.getConstrByName('ConstraintsO '+str(i)).getAttr('Pi')*.01*local_update
            if print_diagnostic:
                print "type2",lp2.getConstrByName('ConstraintsO '+str(i)).getAttr('Pi')
    return deriv_estimate_LP,deriv_estimate_LP_local


def optimizer(num_locations, K, U, demand_physical_means, demand_physical_cov, demand_online_means, demand_online_cov,
              inventory, S, S_local, ship_matrix, p, c, n_for_objVal_calcs,
              step_size_multiplier,step_size_multiplier_local, global_update, local_update, print_thresholds
              ):
    online_demand_distribution = demand_online_means/demand_online_means.sum()
    demand_physical,demand_online=generate_demand(demand_physical_means=demand_physical_means,
                                                  demand_physical_cov=demand_physical_cov,
                                                  demand_online_means=demand_online_means,
                                                  demand_online_cov=demand_online_cov)
    accepted_online,rejected_online,_,_=determine_accepted_orders(demand_online=demand_online,S=S,S_local=S_local)
    prob_next_in_line=compute_prob_next(rejected_online)
    lp2 = Model('Gur')
    lp2, i_o_vars, i_p_vars, c_o_vars =\
        formulate_clicks_bricks_lp(lp2, demand_physical, accepted_online, inventory, ship_matrix, p, c)
    objective_vals_queue=deque([],n_for_objVal_calcs)

    for k in range(K):
        #print "Iter",k
        deriv_estimate_LP_local=defaultdict(lambda: 0)
        deriv_estimate_LP=0
        exception_counter=0
        for u in range(U):
            try:
                demand_physical,demand_online=generate_demand(demand_physical_means=demand_physical_means,
                                                  demand_physical_cov=demand_physical_cov,
                                                  demand_online_means=demand_online_means,
                                                  demand_online_cov=demand_online_cov)
                accepted_online,rejected_online,S_remaining,S_local_remaining=determine_accepted_orders(demand_online=demand_online,S=S,S_local=S_local)
                prob_next_in_line=compute_prob_next(rejected_online)
                prob_last_accepted=compute_prob_last(accepted_online)
                lp2, i_o_vars, i_p_vars, c_o_vars = update_clicks_bricks(lp2, demand_physical, accepted_online, inventory,
                                           i_o_vars, i_p_vars, c_o_vars)
                lp2 = solve_lp(lp2)
                objective_vals_queue.append(lp2.objVal)
                deriv_estimate_LP, deriv_estimate_LP_local=update_derivative_estimates(lp2,deriv_estimate_LP,
                                                                                       deriv_estimate_LP_local,rejected_online,
                                                                                       num_locations,
                                                                                       S_local_remaining,
                                                                                       prob_next_in_line,
                                                                                       prob_last_accepted,
                                                                                       global_update,local_update,
                                                                                       demand_distribution=online_demand_distribution)

            except KeyError:
                exception_counter+=1
                continue
        for i in range(num_locations):
            S_local[i]+=(step_size_multiplier_local/(k+1))*deriv_estimate_LP_local[i]/max(U-exception_counter,1)
        S+=(step_size_multiplier/(k+1))*deriv_estimate_LP/max(U-exception_counter,1)
        S_local = np.maximum(S_local,np.zeros_like(S_local))
        S_local=np.around(S_local,decimals=5)
        S=np.around(S,decimals=5)
        S = max(S,0)
    if print_thresholds:
        print S, S_local
    return sum(objective_vals_queue)/len(objective_vals_queue)

def generate_hyperparameters(mean, var,n):
    return np.random.normal(mean,var,n)

def generate_demand_params(n,phys_mean_mean,phys_mean_var,phys_cov_mean,phys_cov_var,
                           online_mean_mean,online_mean_var,online_cov_mean,online_cov_var):
    demand_physical_means = generate_hyperparameters(phys_mean_mean,phys_mean_var,n)
    demand_physical_cov = np.diag(generate_hyperparameters(phys_cov_mean,phys_cov_var,n))
    demand_online_means = generate_hyperparameters(online_mean_mean,online_mean_var,n)
    demand_online_cov = np.diag(generate_hyperparameters(online_cov_mean,online_cov_var,n))
    return demand_physical_means,demand_physical_cov,demand_online_means,demand_online_cov


def step(action,remaining_orders, demand_physical, accepted_online, inventory, ship_matrix, p, c):
    done = 1 if len(remaining_orders)==1 else 0
    if action==1:
        accepted_online.append(remaining_orders[0])
    remaining_orders=remaining_orders[1:]
    reward=0
    if done==1:
        reward=get_reward(demand_physical, accepted_online, inventory, ship_matrix, p, c)
    observation=len(accepted_online)
    return observation, reward, done, {}, remaining_orders, accepted_online

def step_v2(action,remaining_orders, demand_physical, accepted_online, inventory, ship_matrix, p, c):
    done = 1 if len(remaining_orders)==1 else 0
    if action==1:
        accepted_online.append(remaining_orders[0])
    remaining_orders=remaining_orders[1:]
    reward=0.0
    #if done==1:
    #    reward=get_reward(demand_physical, accepted_online, inventory, ship_matrix, p, c)
    observation=len(accepted_online)
    return observation, reward, done, {}, remaining_orders, accepted_online

def step_v3(action,remaining_orders, demand_physical, accepted_online, inventory, ship_matrix, p, c):
    done = 1 if len(remaining_orders)==1 else 0
    if action==1:
        accepted_online.append(remaining_orders[0])
    remaining_orders=remaining_orders[1:]
    reward=0.0
    #if done==1:
    #    reward=get_reward(demand_physical, accepted_online, inventory, ship_matrix, p, c)
    observation=[1,len(accepted_online)]
    return observation, reward, done, {}, remaining_orders, accepted_online

def step_v4(action,remaining_orders, demand_physical, accepted_online, inventory, ship_matrix, p, c, cutoff):
    done = 1 if len(remaining_orders)==1 else 0
    reward=0.0
    if action==1:
        accepted_online.append(remaining_orders[0])
        if len(accepted_online)<cutoff:
            reward=1.0
        else:
            reward=-1.0
    remaining_orders=remaining_orders[1:]
    #if done==1:
    #    reward=get_reward(demand_physical, accepted_online, inventory, ship_matrix, p, c)
    observation=[1,len(accepted_online)/1000.0]
    return observation, reward, done, {}, remaining_orders, accepted_online




def get_reward(demand_physical, accepted_online, inventory, ship_matrix, p, c):
    lp2 = Model('Gur')
    lp2, i_o_vars, i_p_vars, c_o_vars =\
        formulate_clicks_bricks_lp(lp2, demand_physical, accepted_online, inventory, ship_matrix, p, c)
    lp2 = solve_lp(lp2)
    return lp2.objVal

def order_sequence(demand_online):
    online_orders = np.empty(0, dtype=int)
    for i in range(len(demand_online)):
        temp_orders = np.zeros(int(demand_online[i]), dtype=int)
        temp_orders.fill(int(i))
        online_orders=np.append(online_orders,temp_orders)
    np.random.shuffle(online_orders)
    return online_orders

def compute_reward(demand_physical,accepted_online,order_sequence,ys,inventory, ship_matrix, p, c):
    lp2 = Model('Gur')
    lp2, i_o_vars, i_p_vars, c_o_vars =\
        formulate_clicks_bricks_lp(lp2, demand_physical, accepted_online, inventory, ship_matrix, p, c)
    lp2 = solve_lp(lp2)
    cancels_dict=find_cancels(c_o_vars)
    shipments_dict=find_shipments(lp2,demand_physical.shape[1])
    actions_orders = zip(order_sequence,ys)
    #print actions_orders
    cancel_rewards = compute_cancel_rewards(actions_orders,cancels_dict,c)
    shipment_rewards = compute_shipment_rewards(actions_orders,shipments_dict,ship_matrix, cancel_rewards)
    fill_rewards = compute_fill_rewards(actions_orders,p)
    #print len(cancel_rewards),len(shipment_rewards),len(fill_rewards), len(actions_orders)
    for i in range(len(fill_rewards)):
        if cancel_rewards[i]!=0:
            fill_rewards[i]=0
    total_rewards=[sum(x) for x in zip(cancel_rewards, shipment_rewards,fill_rewards)][::-1]
    #print "cancel rewards"
    #print cancel_rewards
    #print "ship rewards"
    #print shipment_rewards
    #print "fill rewards"
    #print fill_rewards
    #print "total rewards"
    #print total_rewards
    return total_rewards

def find_cancels(c_o_vars):
    cancels_dict = {}
    #print "cancels:"
    for key,var in c_o_vars.iteritems():
        cancels_dict[key]=var.getAttr('X')
        #print key,var.getAttr('X')
    return cancels_dict

def find_shipments(lp,n):
    shipments_dict = {}
    #print "ship costs:"
    for i in range(n):
        shipments_dict[i]={}
        for j in range(n):
            shipments_dict[i][j]=lp.getVarByName('I'+str(i)+'_O'+str(j)).getAttr('x')
            #print i,j, lp.getVarByName('I'+str(i)+'_O'+str(j)).getAttr('x')
    #for key1,var1 in i_o_vars.iteritems():
    #    for key2,var2 in var1.iteritems():
    #        i_o_vars[key2]=var2.getAttr('X')
    #        print key2,var2.getAttr('X')
    return shipments_dict

def compute_cancel_rewards(actions_orders,cancels_dict,c):
    cancel_rewards = []
    for order in reversed(actions_orders):
        if order[1]==1:
            cancel_rewards.append(0)
        elif cancels_dict[order[0]]>0:
            cancels_dict[order[0]]-=1
            cancel_rewards.append(c[order[0]])
        else:
            cancel_rewards.append(0)
    return cancel_rewards

def compute_shipment_rewards(actions_orders,shipments_dict,ship_matrix,cancel_rewards):
    ship_rewards = []
    shipments_processed = process_shipment_data(shipments_dict,ship_matrix)
    for index, order in enumerate(reversed(actions_orders)):
        if order[1]==1:
            ship_rewards.append(0)
        elif cancel_rewards[index]!=0:
            ship_rewards.append(0)
        elif len(shipments_processed[order[0]])>0:
            ship_rewards.append(shipments_processed[order[0]][0])
            shipments_processed[order[0]]=shipments_processed[order[0]][1:]
        else:
            ship_rewards.append(0)
    return ship_rewards

def process_shipment_data(shipments_dict,ship_matrix):
    shipments_processed = {}
    for key1, val1 in shipments_dict.iteritems():
        costs = []
        for key2, val2 in val1.iteritems():
            costs+=[-1.0*ship_matrix[key1][key2]]*int(shipments_dict[key1][key2])
        costs.sort()
        shipments_processed[key1]=costs
    return shipments_processed

def compute_fill_rewards(actions_orders,p):
    fill_rewards = []
    for order in reversed(actions_orders):
        if order[1]==1:
            fill_rewards.append(0)
        else:
            fill_rewards.append(p[order[0]])
    return fill_rewards

def test_reward_old(observations,cutoff):
    rewards = []
    for obs in observations:
        if obs[0][1]<=cutoff:
            rewards.append(1.0)
        else:
            rewards.append(-1.0)
    return rewards

def test_reward(xs,ys,cutoff):
    xys = zip(xs,ys)
    rewards = []
    for obs,act in xys:
        if act==1:
            rewards.append(0.0)
        else:
            if obs[0][1]<=cutoff:
                rewards.append(1.0)
            else:
                rewards.append(-1.0)
    return rewards


def optimizer_poisson(num_locations, K, U, demand_physical_means, demand_online_means,
              inventory, S, S_local, ship_matrix, p, c, n_for_objVal_calcs,
              step_size_multiplier,step_size_multiplier_local, global_update, local_update, print_thresholds,
                      print_iter_thresholds=None
              ):
    online_demand_distribution = demand_online_means/demand_online_means.sum()
    demand_physical,demand_online=generate_demand_poisson(demand_physical_means=demand_physical_means,
                                                  demand_online_means=demand_online_means)
    accepted_online,rejected_online,_,_=determine_accepted_orders(demand_online=demand_online,S=S,S_local=S_local)
    prob_next_in_line=compute_prob_next(rejected_online)
    lp2 = Model('Gur')
    lp2, i_o_vars, i_p_vars, c_o_vars =\
        formulate_clicks_bricks_lp(lp2, demand_physical, accepted_online, inventory, ship_matrix, p, c)
    objective_vals_queue=deque([],n_for_objVal_calcs)

    for k in range(K):
        #print "Iter",k
        deriv_estimate_LP_local=defaultdict(lambda: 0)
        deriv_estimate_LP=0
        exception_counter=0
        for u in range(U):
            try:
                demand_physical,demand_online=generate_demand_poisson(demand_physical_means=demand_physical_means,
                                                  demand_online_means=demand_online_means)
                #print "Demand", demand_physical, demand_online
                #print "S_local", S_local
                accepted_online,rejected_online,S_remaining,S_local_remaining=determine_accepted_orders(demand_online=demand_online,S=S,S_local=S_local)
                #print "accepted", accepted_online
                prob_next_in_line=compute_prob_next(rejected_online)
                prob_last_accepted=compute_prob_last(accepted_online)
                lp2, i_o_vars, i_p_vars, c_o_vars = update_clicks_bricks(lp2, demand_physical, accepted_online, inventory,
                                           i_o_vars, i_p_vars, c_o_vars)
                lp2 = solve_lp(lp2)
                objective_vals_queue.append(lp2.objVal)
                deriv_estimate_LP, deriv_estimate_LP_local=update_derivative_estimates(lp2,deriv_estimate_LP,
                                                                                       deriv_estimate_LP_local,rejected_online,
                                                                                       num_locations,
                                                                                       S_local_remaining,
                                                                                       prob_next_in_line,
                                                                                       prob_last_accepted,
                                                                                       global_update,local_update,
                                                                                       demand_distribution=online_demand_distribution)

            except KeyError:
                exception_counter+=1
                continue
        if print_iter_thresholds is not None:
            if u%print_iter_thresholds==0:
                print S, S_local
                if local_update>0:
                    print "Local Gradient:",sum(deriv_estimate_LP_local.values())
                if global_update>0:
                    print "Global Gradient:",deriv_estimate_LP
                print sum(objective_vals_queue)/len(objective_vals_queue)
        for i in range(num_locations):
            S_local[i]+=(step_size_multiplier_local/(k+1))*deriv_estimate_LP_local[i]/max(U-exception_counter,1)
        S+=(step_size_multiplier/(k+1))*deriv_estimate_LP/max(U-exception_counter,1)
        S_local = np.maximum(S_local,np.zeros_like(S_local))
        S_local = np.minimum(S_local,inventory)
        S_local=np.around(S_local,decimals=5)
        S=np.around(S,decimals=5)
        S = max(S,0)
        S = min(inventory.sum(),S)
    if print_thresholds:
        print S, S_local
    return sum(objective_vals_queue)/len(objective_vals_queue)




def formulate_clicks_bricks_lp_salvage(lp, demand_physical, accepted_online, inventory, ship_matrix, p, c,salvage_rate):
    n=demand_physical.size
    demand_physical=np.around(demand_physical,decimals=3).flatten()
    accepted_online=np.around(accepted_online,decimals=3).flatten()
    i_dict = {i:'I_'+str(i) for i in range(n)} #Inventory nodes
    s_dict = {i:'S_'+str(i) for i in range(n)} #Salvaged inventory nodes
    o_dict = {i:'O_'+str(i) for i in range(n)} #Online demand nodes
    p_dict = {i:'P_'+str(i) for i in range(n)} #Physical demand nodes
    i_o_vars = {}
    i_p_vars = {}
    c_o_vars = {}
    s_vars = {}
    for i in range(n):
        c_o_vars[i]=lp.addVar(vtype=GRB.CONTINUOUS, name='CO_'+str(i), lb=0)
        i_p_vars[i]=lp.addVar(vtype=GRB.CONTINUOUS, name='IP_'+str(i),
                              lb=0)
        s_vars[i]=lp.addVar(vtype=GRB.CONTINUOUS, name='Salvage_'+str(i),
                              lb=0)
        i_o_vars[i]={}
        for j in range(n):
            i_o_vars[i][j]=lp.addVar(vtype=GRB.CONTINUOUS, name='I'+str(i)+'_O'+str(j))
    lp.update()
    #First line is prices for physical demand
    #second line is prices - shipping cost
    #third line is cancel costs
    lp.setObjective(LinExpr([p[i] for i in range(n)]+\
                    [p[i]-ship_matrix[i][j] for i,j in itertools.product(range(n), repeat=2)]+\
                    [c[i] for i in range(n)]+\
                    [p[i]*salvage_rate for i in range(n)],
                    [i_p_vars[i] for i in range(n)]+\
                    [i_o_vars[i][j] for i,j in itertools.product(range(n), repeat=2)]+\
                    [c_o_vars[i] for i in range(n)]+\
                    [s_vars[i] for i in range(n)]),GRB.MAXIMIZE)
    for i in range(n):
        temp_coefficients = [1]
        temp_vars = [i_p_vars[i]]
        lp.addConstr(LinExpr(temp_coefficients,temp_vars) == min(demand_physical[i],inventory[i]),'Physical '+str(i))
    for i in range(n):
        temp_coefficients = [1 for j in range(n+2)]
        #temp_vars = [i_p_vars[i]]+[i_o_vars[j][i] for j in range(n)]
        #Updated to reflect salvage value
        temp_vars = [i_p_vars[i]]+[i_o_vars[i][j] for j in range(n)]+[s_vars[i]]
        lp.addConstr(LinExpr(temp_coefficients,temp_vars) == inventory[i],'ConstraintsI '+str(i))
    for i in range(n):
        temp_coefficients = [1 for j in range(n+1)]
        temp_vars = [c_o_vars[i]]+[i_o_vars[j][i] for j in range(n)]
        lp.addConstr(LinExpr(temp_coefficients,temp_vars) == accepted_online[i],'ConstraintsO '+str(i))
    lp.update()
    return lp, i_o_vars, i_p_vars, c_o_vars, s_vars



def update_clicks_bricks_salvage(lp, demand_physical, accepted_online, inventory, i_o_vars, i_p_vars, c_o_vars,s_vars):
    n=accepted_online.size
    demand_physical=np.around(demand_physical,decimals=3).flatten()
    demand_online=np.around(accepted_online,decimals=3).flatten()
    for i in range(n):
        lp.remove(lp.getConstrByName('Physical '+str(i)))
        temp_coefficients = [1]
        temp_vars = [i_p_vars[i]]
        lp.addConstr(LinExpr(temp_coefficients,temp_vars) == min(demand_physical[i],inventory[i]),'Physical '+str(i))
    for i in range(n):
        lp.remove(lp.getConstrByName('ConstraintsI '+str(i)))
        lp.remove(lp.getConstrByName('ConstraintsO '+str(i)))
        temp_coefficients = [1 for j in range(n+2)]
        #temp_vars = [i_p_vars[i]]+[i_o_vars[j][i] for j in range(n)]
        #temp_vars = [i_p_vars[i]]+[i_o_vars[i][j] for j in range(n)]
        temp_vars = [i_p_vars[i]]+[i_o_vars[i][j] for j in range(n)]+[s_vars[i]]
        lp.addConstr(LinExpr(temp_coefficients,temp_vars) == inventory[i],'ConstraintsI '+str(i))
    for i in range(n):
        temp_coefficients = [1 for j in range(n+1)]
        temp_vars = [c_o_vars[i]]+[i_o_vars[j][i] for j in range(n)]
        lp.addConstr(LinExpr(temp_coefficients,temp_vars) == accepted_online[i],'ConstraintsO '+str(i))
    lp.update()
    return lp, i_o_vars, i_p_vars, c_o_vars, s_vars


def optimizer_poisson_salvage(num_locations, K, U, demand_physical_means, demand_online_means,
              inventory, S, S_local, ship_matrix, p, c,salvage_rate, n_for_objVal_calcs,
              step_size_multiplier,step_size_multiplier_local, global_update, local_update, print_thresholds,
                      print_iter_thresholds=None
              ):
    online_demand_distribution = demand_online_means/demand_online_means.sum()
    demand_physical,demand_online=generate_demand_poisson(demand_physical_means=demand_physical_means,
                                                  demand_online_means=demand_online_means)
    accepted_online,rejected_online,_,_=determine_accepted_orders(demand_online=demand_online,S=S,S_local=S_local)
    prob_next_in_line=compute_prob_next(rejected_online)
    lp2 = Model('Gur')
    lp2, i_o_vars, i_p_vars, c_o_vars, s_vars =\
        formulate_clicks_bricks_lp_salvage(lp2, demand_physical, accepted_online, inventory, ship_matrix, p, c,salvage_rate)
    objective_vals_queue=deque([],n_for_objVal_calcs)

    for k in range(K):
        #print "Iter",k
        deriv_estimate_LP_local=defaultdict(lambda: 0)
        deriv_estimate_LP=0
        exception_counter=0
        for u in range(U):
            try:
                demand_physical,demand_online=generate_demand_poisson(demand_physical_means=demand_physical_means,
                                                  demand_online_means=demand_online_means)
                #print "Demand", demand_physical, demand_online
                #print "S_local", S_local
                accepted_online,rejected_online,S_remaining,S_local_remaining=determine_accepted_orders(demand_online=demand_online,S=S,S_local=S_local)
                #print "accepted", accepted_online
                prob_next_in_line=compute_prob_next(rejected_online)
                prob_last_accepted=compute_prob_last(accepted_online)
                lp2, i_o_vars, i_p_vars, c_o_vars, s_vars = update_clicks_bricks_salvage(lp2, demand_physical, accepted_online, inventory,
                                           i_o_vars, i_p_vars, c_o_vars, s_vars)
                lp2 = solve_lp(lp2)
                objective_vals_queue.append(lp2.objVal)
                deriv_estimate_LP, deriv_estimate_LP_local=update_derivative_estimates(lp2,deriv_estimate_LP,
                                                                                       deriv_estimate_LP_local,rejected_online,
                                                                                       num_locations,
                                                                                       S_local_remaining,
                                                                                       prob_next_in_line,
                                                                                       prob_last_accepted,
                                                                                       global_update,local_update,
                                                                                       demand_distribution=online_demand_distribution)

            except KeyError:
                exception_counter+=1
                continue
        if print_iter_thresholds is not None:
            if u%print_iter_thresholds==0:
                print S, S_local
                if local_update>0:
                    print "Local Gradient:",sum(deriv_estimate_LP_local.values())
                if global_update>0:
                    print "Global Gradient:",deriv_estimate_LP
                print sum(objective_vals_queue)/len(objective_vals_queue)
        for i in range(num_locations):
            S_local[i]+=(step_size_multiplier_local/(k+1))*deriv_estimate_LP_local[i]/max(U-exception_counter,1)
        S+=(step_size_multiplier/(k+1))*deriv_estimate_LP/max(U-exception_counter,1)
        S_local = np.maximum(S_local,np.zeros_like(S_local))
        S_local = np.minimum(S_local,inventory)
        S_local=np.around(S_local,decimals=5)
        S=np.around(S,decimals=5)
        S = max(S,0)
        S = min(inventory.sum(),S)
    if print_thresholds:
        print S, S_local
    return sum(objective_vals_queue)/len(objective_vals_queue)


def optimizer_poisson_minimization_prob(num_locations, K, U, demand_physical_means, demand_online_means,
              inventory, S, S_local, ship_matrix, p, c,salvage_rate, n_for_objVal_calcs,
              step_size_multiplier,step_size_multiplier_local, global_update, local_update, print_thresholds,
                      print_iter_thresholds=None
              ):
    online_demand_distribution = demand_online_means/demand_online_means.sum()
    demand_physical,demand_online=generate_demand_poisson(demand_physical_means=demand_physical_means,
                                                  demand_online_means=demand_online_means)
    accepted_online,rejected_online,_,_=determine_accepted_orders(demand_online=demand_online,S=S,S_local=S_local)
    prob_next_in_line=compute_prob_next(rejected_online)
    lp2 = Model('Gur')
    lp2, i_o_vars, i_p_vars, c_o_vars, s_vars =\
        formulate_clicks_bricks_lp_salvage(lp2, demand_physical, accepted_online, inventory, ship_matrix, p, c,salvage_rate)
    objective_vals_queue=deque([],n_for_objVal_calcs)

    for k in range(K):
        #print "Iter",k
        deriv_estimate_LP_local=defaultdict(lambda: 0)
        deriv_estimate_LP=0
        exception_counter=0
        for u in range(U):
            try:
                demand_physical,demand_online=generate_demand_poisson(demand_physical_means=demand_physical_means,
                                                  demand_online_means=demand_online_means)
                #print "Demand", demand_physical, demand_online
                #print "S_local", S_local
                accepted_online,rejected_online,S_remaining,S_local_remaining=determine_accepted_orders(demand_online=demand_online,S=S,S_local=S_local)
                #print "accepted", accepted_online
                prob_next_in_line=compute_prob_next(rejected_online)
                prob_last_accepted=compute_prob_last(accepted_online)
                lp2, i_o_vars, i_p_vars, c_o_vars, s_vars = update_clicks_bricks_salvage(lp2, demand_physical, accepted_online, inventory,
                                           i_o_vars, i_p_vars, c_o_vars, s_vars)
                lp2 = solve_lp(lp2)
                objective_vals_queue.append(lp2.objVal)
                deriv_estimate_LP, deriv_estimate_LP_local=update_derivative_estimates(lp2,deriv_estimate_LP,
                                                                                       deriv_estimate_LP_local,rejected_online,
                                                                                       num_locations,
                                                                                       S_local_remaining,
                                                                                       prob_next_in_line,
                                                                                       prob_last_accepted,
                                                                                       global_update,local_update,
                                                                                       demand_distribution=online_demand_distribution)

            except KeyError:
                exception_counter+=1
                continue
        if print_iter_thresholds is not None:
            if u%print_iter_thresholds==0:
                print S, S_local
                if local_update>0:
                    print "Local Gradient:",sum(deriv_estimate_LP_local.values())
                if global_update>0:
                    print "Global Gradient:",deriv_estimate_LP
                print sum(objective_vals_queue)/len(objective_vals_queue)
        for i in range(num_locations):
            S_local[i]+=(step_size_multiplier_local/(k+1))*deriv_estimate_LP_local[i]/max(U-exception_counter,1)
        S+=(step_size_multiplier/(k+1))*deriv_estimate_LP/max(U-exception_counter,1)
        S_local = np.maximum(S_local,np.zeros_like(S_local))
        S_local = np.minimum(S_local,inventory)
        S_local=np.around(S_local,decimals=5)
        S=np.around(S,decimals=5)
        S = max(S,0)
        S = min(inventory.sum(),S)
    if print_thresholds:
        print S, S_local
    return sum(objective_vals_queue)/len(objective_vals_queue)


def formulate_clicks_bricks_no_ship(lp, demand_physical, accepted_online, inventory, ship_matrix, p, c,salvage_rate):
    n=demand_physical.size
    demand_physical=np.around(demand_physical,decimals=3).flatten()
    accepted_online=np.around(accepted_online,decimals=3).flatten()
    i_dict = {i:'I_'+str(i) for i in range(n)} #Inventory nodes
    s_dict = {i:'S_'+str(i) for i in range(n)} #Salvaged inventory nodes
    o_dict = {i:'O_'+str(i) for i in range(n)} #Online demand nodes
    p_dict = {i:'P_'+str(i) for i in range(n)} #Physical demand nodes
    i_o_vars = {}
    i_p_vars = {}
    c_o_vars = {}
    s_vars = {}
    for i in range(n):
        c_o_vars[i]=lp.addVar(vtype=GRB.CONTINUOUS, name='CO_'+str(i), lb=0)
        i_p_vars[i]=lp.addVar(vtype=GRB.CONTINUOUS, name='IP_'+str(i),
                              lb=0)
        s_vars[i]=lp.addVar(vtype=GRB.CONTINUOUS, name='Salvage_'+str(i),
                              lb=0)
        i_o_vars[i]=lp.addVar(vtype=GRB.CONTINUOUS, name='IO_'+str(i), lb=0)
    lp.update()
    #First line is prices for physical demand
    #second line is prices - shipping cost
    #third line is cancel costs
    lp.setObjective(LinExpr([p[i] for i in range(n)]+\
                    [p[i]-ship_matrix[i][i] for i in range(n)]+\
                    [c[i] for i in range(n)]+\
                    [p[i]*salvage_rate for i in range(n)],
                    [i_p_vars[i] for i in range(n)]+\
                    [i_o_vars[i] for i in range(n)]+\
                    [c_o_vars[i] for i in range(n)]+\
                    [s_vars[i] for i in range(n)]),GRB.MAXIMIZE)
    for i in range(n):
        temp_coefficients = [1]
        temp_vars = [i_p_vars[i]]
        lp.addConstr(LinExpr(temp_coefficients,temp_vars) == min(demand_physical[i],inventory[i]),'Physical '+str(i))
    for i in range(n):
        temp_coefficients = [1 for j in range(3)]
        #temp_vars = [i_p_vars[i]]+[i_o_vars[j][i] for j in range(n)]
        #Updated to reflect salvage value
        #temp_vars = [i_p_vars[i]]+[i_o_vars[i][j] for j in range(n)]+[s_vars[i]]
        temp_vars = [i_p_vars[i],i_o_vars[i],s_vars[i]]
        lp.addConstr(LinExpr(temp_coefficients,temp_vars) == inventory[i],'ConstraintsI '+str(i))
    for i in range(n):
        temp_coefficients = [1 for j in range(2)]
        temp_vars = [c_o_vars[i],i_o_vars[i]]
        lp.addConstr(LinExpr(temp_coefficients,temp_vars) == accepted_online[i],'ConstraintsO '+str(i))
    lp.update()
    return lp, i_o_vars, i_p_vars, c_o_vars, s_vars



def update_clicks_bricks_no_ship(lp, demand_physical, accepted_online, inventory, i_o_vars, i_p_vars, c_o_vars,s_vars):
    n=accepted_online.size
    demand_physical=np.around(demand_physical,decimals=3).flatten()
    demand_online=np.around(accepted_online,decimals=3).flatten()
    for i in range(n):
        lp.remove(lp.getConstrByName('Physical '+str(i)))
        temp_coefficients = [1]
        temp_vars = [i_p_vars[i]]
        lp.addConstr(LinExpr(temp_coefficients,temp_vars) == min(demand_physical[i],inventory[i]),'Physical '+str(i))
    for i in range(n):
        lp.remove(lp.getConstrByName('ConstraintsI '+str(i)))
        lp.remove(lp.getConstrByName('ConstraintsO '+str(i)))
        temp_coefficients = [1 for j in range(3)]
        #temp_vars = [i_p_vars[i]]+[i_o_vars[j][i] for j in range(n)]
        #temp_vars = [i_p_vars[i]]+[i_o_vars[i][j] for j in range(n)]
        #temp_vars = [i_p_vars[i]]+[i_o_vars[i][j] for j in range(n)]+[s_vars[i]]
        temp_vars = [i_p_vars[i],i_o_vars[i],s_vars[i]]
        lp.addConstr(LinExpr(temp_coefficients,temp_vars) == inventory[i],'ConstraintsI '+str(i))
    for i in range(n):
        temp_coefficients = [1 for j in range(2)]
        #temp_vars = [c_o_vars[i]]+[i_o_vars[j][i] for j in range(n)]
        temp_vars = [c_o_vars[i],i_o_vars[i]]
        lp.addConstr(LinExpr(temp_coefficients,temp_vars) == accepted_online[i],'ConstraintsO '+str(i))
    lp.update()
    return lp, i_o_vars, i_p_vars, c_o_vars, s_vars


def optimizer_poisson_no_ship(num_locations, K, U, demand_physical_means, demand_online_means,
              inventory, S, S_local, ship_matrix, p, c,salvage_rate, n_for_objVal_calcs,
              step_size_multiplier,step_size_multiplier_local, global_update, local_update, print_thresholds,
                      print_iter_thresholds=None
              ):
    online_demand_distribution = demand_online_means/demand_online_means.sum()
    demand_physical,demand_online=generate_demand_poisson(demand_physical_means=demand_physical_means,
                                                  demand_online_means=demand_online_means)
    accepted_online,rejected_online,_,_=determine_accepted_orders(demand_online=demand_online,S=S,S_local=S_local)
    prob_next_in_line=compute_prob_next(rejected_online)
    lp2 = Model('Gur')
    lp2, i_o_vars, i_p_vars, c_o_vars, s_vars =\
        formulate_clicks_bricks_no_ship(lp2, demand_physical, accepted_online, inventory, ship_matrix, p, c,salvage_rate)
    objective_vals_queue=deque([],n_for_objVal_calcs)

    for k in range(K):
        #print "Iter",k
        deriv_estimate_LP_local=defaultdict(lambda: 0)
        deriv_estimate_LP=0
        exception_counter=0
        for u in range(U):
            try:
                demand_physical,demand_online=generate_demand_poisson(demand_physical_means=demand_physical_means,
                                                  demand_online_means=demand_online_means)
                #print "Demand", demand_physical, demand_online
                #print "S_local", S_local
                accepted_online,rejected_online,S_remaining,S_local_remaining=determine_accepted_orders(demand_online=demand_online,S=S,S_local=S_local)
                #print "accepted", accepted_online
                prob_next_in_line=compute_prob_next(rejected_online)
                prob_last_accepted=compute_prob_last(accepted_online)
                lp2, i_o_vars, i_p_vars, c_o_vars, s_vars = update_clicks_bricks_no_ship(lp2, demand_physical, accepted_online, inventory,
                                           i_o_vars, i_p_vars, c_o_vars, s_vars)
                lp2 = solve_lp(lp2)
                objective_vals_queue.append(lp2.objVal)
                deriv_estimate_LP, deriv_estimate_LP_local=update_derivative_estimates(lp2,deriv_estimate_LP,
                                                                                       deriv_estimate_LP_local,rejected_online,
                                                                                       num_locations,
                                                                                       S_local_remaining,
                                                                                       prob_next_in_line,
                                                                                       prob_last_accepted,
                                                                                       global_update,local_update,
                                                                                       demand_distribution=online_demand_distribution)

            except KeyError:
                exception_counter+=1
                continue
        if print_iter_thresholds is not None:
            if u%print_iter_thresholds==0:
                print S, S_local
                if local_update>0:
                    print "Local Gradient:",sum(deriv_estimate_LP_local.values())
                if global_update>0:
                    print "Global Gradient:",deriv_estimate_LP
                print sum(objective_vals_queue)/len(objective_vals_queue)
        for i in range(num_locations):
            S_local[i]+=(step_size_multiplier_local/(k+1))*deriv_estimate_LP_local[i]/max(U-exception_counter,1)
        S+=(step_size_multiplier/(k+1))*deriv_estimate_LP/max(U-exception_counter,1)
        S_local = np.maximum(S_local,np.zeros_like(S_local))
        S_local = np.minimum(S_local,inventory)
        S_local=np.around(S_local,decimals=5)
        S=np.around(S,decimals=5)
        S = max(S,0)
        S = min(inventory.sum(),S)
    if print_thresholds:
        print S, S_local
    return sum(objective_vals_queue)/len(objective_vals_queue)

def first_stage_decision_optimal(cur_location,accepted_online,cur_objective_val,num_trials,demand_physical_means, inventory,ship_matrix,p,c,salvage_rate):
    tentative_accepted_online = np.copy(accepted_online)
    tentative_accepted_online[cur_location]+=1
    new_obj_vals = []
    old_obj_vals = []
    for i in range(num_trials):
        demand_physical = generate_demand_physical_poisson(demand_physical_means)
        lp2 = Model('Gur')
        lp2, i_o_vars, i_p_vars, c_o_vars, s_vars =\
        formulate_clicks_bricks_lp_salvage(lp2, demand_physical, tentative_accepted_online, inventory, ship_matrix, p, c,salvage_rate)
        lp2 = solve_lp(lp2)
        new_obj_vals.append(lp2.objVal)
        lp2 = Model('Gur')
        lp2, i_o_vars, i_p_vars, c_o_vars, s_vars =\
        formulate_clicks_bricks_lp_salvage(lp2, demand_physical, accepted_online, inventory, ship_matrix, p, c,salvage_rate)
        lp2 = solve_lp(lp2)
        old_obj_vals.append(lp2.objVal)
    average_new_value= 1.0*sum(new_obj_vals)/len(new_obj_vals)
    average_old_value = 1.0*sum(old_obj_vals)/len(old_obj_vals)
    #print average_value,cur_objective_val
    #print accepted_online
    if average_new_value>average_old_value:
        #print "Accept"
        return 1,average_new_value
    else:
        #print "Reject"
        return 0,average_old_value

def first_stage_decision_optimal_t_test(cur_location,accepted_online,conf_level,demand_physical_means, inventory,
                                        ship_matrix,p,c,salvage_rate, max_iters=50):
    tentative_accepted_online = np.copy(accepted_online)
    tentative_accepted_online[cur_location]+=1
    new_obj_vals = []
    old_obj_vals = []
    new_obj_sum = 0
    old_obj_sum =0
    n_trials = 0
    confidence = 1
    while confidence>conf_level:
        n_trials+=1
        demand_physical = generate_demand_physical_poisson(demand_physical_means)
        lp2 = Model('Gur')
        lp2, i_o_vars, i_p_vars, c_o_vars, s_vars =\
        formulate_clicks_bricks_lp_salvage(lp2, demand_physical, tentative_accepted_online, inventory, ship_matrix, p, c,salvage_rate)
        lp2 = solve_lp(lp2)
        new_obj_vals.append(lp2.objVal)
        new_obj_sum+=lp2.objVal
        lp2 = Model('Gur')
        lp2, i_o_vars, i_p_vars, c_o_vars, s_vars =\
        formulate_clicks_bricks_lp_salvage(lp2, demand_physical, accepted_online, inventory, ship_matrix, p, c,salvage_rate)
        lp2 = solve_lp(lp2)
        old_obj_sum+=lp2.objVal
        old_obj_vals.append(lp2.objVal)
        if n_trials>=2:
            result = scipy.stats.ttest_ind(new_obj_vals,old_obj_vals)
            confidence = result.pvalue
        if n_trials>=max_iters:
            break
    #print n_trials, confidence
    average_new_value= 1.0*sum(new_obj_vals)/len(new_obj_vals)
    average_old_value = 1.0*sum(old_obj_vals)/len(old_obj_vals)
    if result.statistic>0:
        return 1,average_new_value
    else:
        return 0,average_old_value


def determine_accepted_orders_optimal(num_trials,demand_physical_means,demand_online, inventory,ship_matrix,p,c,salvage_rate):
    num_locations = demand_physical_means.shape[0]
    cur_objective_val = 0
    online_orders = np.empty(0, dtype=int)
    accepted_online = np.zeros_like(demand_physical_means)
    last_order_local = {}
    ##This loop turns the vector of online demand quantities into an ordered sequence of orders
    for i in range(len(demand_online)):
        temp_orders = np.zeros(int(demand_online[i]), dtype=int)
        temp_orders.fill(int(i))
        last_order_local[i]=len(temp_orders)
        online_orders=np.append(online_orders,temp_orders)
    np.random.shuffle(online_orders)
    #Now we want to loop through each order and decide whether to accept or reject it
    for i in range(len(online_orders)):
        cur_location = online_orders[i]
        #cur_decision, cur_objective_val = first_stage_decision_optimal(cur_location,accepted_online,cur_objective_val,
                                                                       #num_trials,demand_physical_means, inventory,ship_matrix,p,c,salvage_rate)
        cur_decision, cur_objective_val = first_stage_decision_optimal_t_test(cur_location,accepted_online,
                                                                       num_trials,demand_physical_means, inventory,ship_matrix,p,c,salvage_rate)
        accepted_online[cur_location]+=cur_decision
    rejected_online = demand_online-accepted_online
    return accepted_online,rejected_online


def optimizer_optimal(num_locations,n_for_objVal_calcs,trials_for_first_stage_estimation, demand_physical_means, demand_online_means,
              inventory, ship_matrix, p, c,salvage_rate):
    obj_vals = []
    for k in range(n_for_objVal_calcs):
        demand_physical,demand_online=generate_demand_poisson(demand_physical_means=demand_physical_means,
                                                      demand_online_means=demand_online_means)
        accepted_online,rejected_online=determine_accepted_orders_optimal(num_trials=trials_for_first_stage_estimation,demand_physical_means=demand_physical_means,
                                                                          demand_online=demand_online, inventory=inventory,
                                                                          ship_matrix=ship_matrix,p=p,c=c,
                                                                          salvage_rate=salvage_rate)
        lp2 = Model('Gur')
        lp2, i_o_vars, i_p_vars, c_o_vars, s_vars =\
            formulate_clicks_bricks_lp_salvage(lp2, demand_physical, accepted_online, inventory, ship_matrix, p, c,salvage_rate)
        lp2 = solve_lp(lp2)
        obj_vals.append(lp2.objVal)
    return sum(obj_vals)*1.0/len(obj_vals)


def optimizer_poisson_salvage_params(num_locations, K, U, demand_physical_means, demand_online_means,
              inventory, S, S_local, ship_matrix, p, c,salvage_rate, n_for_objVal_calcs,
              step_size_multiplier,step_size_multiplier_local, global_update, local_update, print_thresholds,
                      print_iter_thresholds=None
              ):
    online_demand_distribution = demand_online_means/demand_online_means.sum()
    demand_physical,demand_online=generate_demand_poisson(demand_physical_means=demand_physical_means,
                                                  demand_online_means=demand_online_means)
    accepted_online,rejected_online,_,_=determine_accepted_orders(demand_online=demand_online,S=S,S_local=S_local)
    prob_next_in_line=compute_prob_next(rejected_online)
    lp2 = Model('Gur')
    lp2, i_o_vars, i_p_vars, c_o_vars, s_vars =\
        formulate_clicks_bricks_lp_salvage(lp2, demand_physical, accepted_online, inventory, ship_matrix, p, c,salvage_rate)
    objective_vals_queue=deque([],n_for_objVal_calcs)

    for k in range(K):
        #print "Iter",k
        deriv_estimate_LP_local=defaultdict(lambda: 0)
        deriv_estimate_LP=0
        exception_counter=0
        for u in range(U):
            try:
                demand_physical,demand_online=generate_demand_poisson(demand_physical_means=demand_physical_means,
                                                  demand_online_means=demand_online_means)
                #print "Demand", demand_physical, demand_online
                #print "S_local", S_local
                accepted_online,rejected_online,S_remaining,S_local_remaining=determine_accepted_orders(demand_online=demand_online,S=S,S_local=S_local)
                #print "accepted", accepted_online
                prob_next_in_line=compute_prob_next(rejected_online)
                prob_last_accepted=compute_prob_last(accepted_online)
                lp2, i_o_vars, i_p_vars, c_o_vars, s_vars = update_clicks_bricks_salvage(lp2, demand_physical, accepted_online, inventory,
                                           i_o_vars, i_p_vars, c_o_vars, s_vars)
                lp2 = solve_lp(lp2)
                objective_vals_queue.append(lp2.objVal)
                deriv_estimate_LP, deriv_estimate_LP_local=update_derivative_estimates(lp2,deriv_estimate_LP,
                                                                                       deriv_estimate_LP_local,rejected_online,
                                                                                       num_locations,
                                                                                       S_local_remaining,
                                                                                       prob_next_in_line,
                                                                                       prob_last_accepted,
                                                                                       global_update,local_update,
                                                                                       demand_distribution=online_demand_distribution)

            except KeyError:
                exception_counter+=1
                continue
        if print_iter_thresholds is not None:
            if u%print_iter_thresholds==0:
                print S, S_local
                if local_update>0:
                    print "Local Gradient:",sum(deriv_estimate_LP_local.values())
                if global_update>0:
                    print "Global Gradient:",deriv_estimate_LP
                print sum(objective_vals_queue)/len(objective_vals_queue)
        for i in range(num_locations):
            S_local[i]+=(step_size_multiplier_local/(k+1))*deriv_estimate_LP_local[i]/max(U-exception_counter,1)
        S+=(step_size_multiplier/(k+1))*deriv_estimate_LP/max(U-exception_counter,1)
        S_local = np.maximum(S_local,np.zeros_like(S_local))
        S_local = np.minimum(S_local,inventory.sum())
        S_local=np.around(S_local,decimals=5)
        S=np.around(S,decimals=5)
        S = max(S,0)
        S = min(inventory.sum(),S)
    if print_thresholds:
        print S, S_local
    return S, S_local


def optimizer_poisson_no_ship_params(num_locations, K, U, demand_physical_means, demand_online_means,
              inventory, S, S_local, ship_matrix, p, c,salvage_rate, n_for_objVal_calcs,
              step_size_multiplier,step_size_multiplier_local, global_update, local_update, print_thresholds,
                      print_iter_thresholds=None
              ):
    online_demand_distribution = demand_online_means/demand_online_means.sum()
    demand_physical,demand_online=generate_demand_poisson(demand_physical_means=demand_physical_means,
                                                  demand_online_means=demand_online_means)
    accepted_online,rejected_online,_,_=determine_accepted_orders(demand_online=demand_online,S=S,S_local=S_local)
    prob_next_in_line=compute_prob_next(rejected_online)
    lp2 = Model('Gur')
    lp2, i_o_vars, i_p_vars, c_o_vars, s_vars =\
        formulate_clicks_bricks_no_ship(lp2, demand_physical, accepted_online, inventory, ship_matrix, p, c,salvage_rate)
    objective_vals_queue=deque([],n_for_objVal_calcs)

    for k in range(K):
        #print "Iter",k
        deriv_estimate_LP_local=defaultdict(lambda: 0)
        deriv_estimate_LP=0
        exception_counter=0
        for u in range(U):
            try:
                demand_physical,demand_online=generate_demand_poisson(demand_physical_means=demand_physical_means,
                                                  demand_online_means=demand_online_means)
                #print "Demand", demand_physical, demand_online
                #print "S_local", S_local
                accepted_online,rejected_online,S_remaining,S_local_remaining=determine_accepted_orders(demand_online=demand_online,S=S,S_local=S_local)
                #print "accepted", accepted_online
                prob_next_in_line=compute_prob_next(rejected_online)
                prob_last_accepted=compute_prob_last(accepted_online)
                lp2, i_o_vars, i_p_vars, c_o_vars, s_vars = update_clicks_bricks_no_ship(lp2, demand_physical, accepted_online, inventory,
                                           i_o_vars, i_p_vars, c_o_vars, s_vars)
                lp2 = solve_lp(lp2)
                objective_vals_queue.append(lp2.objVal)
                deriv_estimate_LP, deriv_estimate_LP_local=update_derivative_estimates(lp2,deriv_estimate_LP,
                                                                                       deriv_estimate_LP_local,rejected_online,
                                                                                       num_locations,
                                                                                       S_local_remaining,
                                                                                       prob_next_in_line,
                                                                                       prob_last_accepted,
                                                                                       global_update,local_update,
                                                                                       demand_distribution=online_demand_distribution)

            except KeyError:
                exception_counter+=1
                continue
        if print_iter_thresholds is not None:
            if u%print_iter_thresholds==0:
                print S, S_local
                if local_update>0:
                    print "Local Gradient:",sum(deriv_estimate_LP_local.values())
                if global_update>0:
                    print "Global Gradient:",deriv_estimate_LP
                print sum(objective_vals_queue)/len(objective_vals_queue)
        for i in range(num_locations):
            S_local[i]+=(step_size_multiplier_local/(k+1))*deriv_estimate_LP_local[i]/max(U-exception_counter,1)
        S+=(step_size_multiplier/(k+1))*deriv_estimate_LP/max(U-exception_counter,1)
        S_local = np.maximum(S_local,np.zeros_like(S_local))
        S_local = np.minimum(S_local,inventory.sum())
        S_local=np.around(S_local,decimals=5)
        S=np.around(S,decimals=5)
        S = max(S,0)
        S = min(inventory.sum(),S)
    if print_thresholds:
        print S, S_local
    return S, S_local


def evaluate_poisson_salvage(demand_physical, demand_online,
              inventory, S, S_local, ship_matrix, p, c,salvage_rate
              ):
    accepted_online,rejected_online,_,_=determine_accepted_orders(demand_online=demand_online,S=S,S_local=S_local)
    lp2 = Model('Gur')
    lp2, i_o_vars, i_p_vars, c_o_vars, s_vars =\
        formulate_clicks_bricks_lp_salvage(lp2, demand_physical, accepted_online, inventory, ship_matrix, p, c,salvage_rate)
    lp2 = solve_lp(lp2)
    return lp2.objVal


def evaluate_poisson_no_ship(demand_physical, demand_online,
              inventory, S, S_local, ship_matrix, p, c,salvage_rate
              ):
    accepted_online,rejected_online,_,_=determine_accepted_orders(demand_online=demand_online,S=S,S_local=S_local)
    lp2 = Model('Gur')
    lp2, i_o_vars, i_p_vars, c_o_vars, s_vars =\
        formulate_clicks_bricks_no_ship(lp2, demand_physical, accepted_online, inventory, ship_matrix, p, c,salvage_rate)
    lp2 = solve_lp(lp2)
    return lp2.objVal