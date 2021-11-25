"""
Code for running IPW and DRO on Warfarin dataset.

python full_warfarin_cb_experiments.py --num_runs 5 --alpha 0.8 --nonsep 3

From Mostly exploration-free paper's author
'Column 1: is male
Column 2: is female
Columns 3-5: Race
Columns 6-7: Ethnicity
Columns 8-16: Age
column 9 is an indicator of age between 50-59.
Column 17: Height (imputed for NAs)
Column 18: Height is NA or not
Column 19: Weight
Column 20: Weight is NA or not'
"""

import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import warfarin_cb as med
import argparse
import pickle
import os
import multiprocessing
from joblib import Parallel, delayed
num_cores = min(10, multiprocessing.cpu_count())

parser = argparse.ArgumentParser()
parser.add_argument('--outdir', default='saved_results/warfarin/')
# parser.add_argument('--filename', default='toygaussian')
parser.add_argument('--filename', default='warfarin')
parser.add_argument('--alpha', default=0.8, type=float)
parser.add_argument('--num_runs', default=5, type=int)
parser.add_argument('--nonsep', default=3, type=int)
args = parser.parse_args()

default_args_dict = {"steps":200, "dlr":0.1, "ntrain":2000, "ndim":93, "alpha":args.alpha, "kdual":2.0, "joint":False, "lip":0.01, "penalty":0.0,
                    "min_p":0.025, "model_type":"baseline", "turk":False, "oracle":False, "shuffle":0.0, "subcov":False, "dataset": "warfarin",
                    "outdir":args.outdir, "ntest":20000, "sigma_x":1, "sigma_y":0.1, "nonsep":args.nonsep, "filename":args.filename,
                    "num_runs":args.num_runs, "run_id":0}
print("Output Data Path: "+str(default_args_dict["outdir"]))
print("Alpha: "+str(default_args_dict["alpha"]))
print("Selection on feature index: "+str(default_args_dict["nonsep"]))
print(default_args_dict)

if not os.path.exists(default_args_dict["outdir"]):
    os.makedirs(default_args_dict["outdir"])

def oracle_eval(default_args_dict):
    args_dict = default_args_dict.copy()
    args_dict["model_type"] = "oracle_eval"
    all_acc = []
    all_loss = []
    all_params = []
    # all_metrics, all_params = med.run_shift_experiment(args_dict)
    # all_acc = [metrics['acc'] for metrics in all_metrics]
    # all_loss = [metrics['loss'] for metrics in all_metrics]
    for shift in shifts:
        args_dict["min_p"] = shift
        acc, loss, params = med.run_shift_experiment(args_dict)
        all_acc.append(acc)
        all_loss.append(loss)
        all_params.append(params)
    print("Oracle Eval: "+str(all_acc)+" "+str(all_loss))
    return all_acc, all_loss, all_params

def subcovariate_shift_dro_eval(default_args_dict):
    args_dict = default_args_dict.copy()
    args_dict["subcov"] = True
    args_dict["model_type"] = "dro_eval"
    # args_dict["dlr"] = 0.1
    # args_dict["nonsep"] = 88
    all_acc = []
    all_loss = []
    all_params = []
    # all_metrics, all_params = med.run_shift_experiment(args_dict)
    # all_acc = [metrics['acc'] for metrics in all_metrics]
    # all_loss = [metrics['loss'] for metrics in all_metrics]
    for shift in shifts:
        args_dict["min_p"] = shift
        acc, loss, params = med.run_shift_experiment(args_dict)
        all_acc.append(acc)
        all_loss.append(loss)
        all_params.append(params)
    print("Subcovariate Shift DRO Eval: "+str(all_acc)+" "+str(all_loss))
    return all_acc, all_loss, all_params

def covariate_shift_dro_eval(default_args_dict):
    args_dict = default_args_dict.copy()
    args_dict["model_type"] = "dro_eval"
    # args_dict["dlr"] = 0.1
    all_acc = []
    all_loss = []
    all_params = []
    # all_metrics, all_params = med.run_shift_experiment(args_dict)
    # all_acc = [metrics['acc'] for metrics in all_metrics]
    # all_loss = [metrics['loss'] for metrics in all_metrics]
    for shift in shifts:
        args_dict["min_p"] = shift
        acc, loss, params = med.run_shift_experiment(args_dict)
        all_acc.append(acc)
        all_loss.append(loss)
        all_params.append(params)
    print("Covariate Shift DRO Eval: "+str(all_acc)+" "+str(all_loss))
    return all_acc, all_loss, all_params

def joint_dro_eval(default_args_dict):
    args_dict = default_args_dict.copy()
    # args_dict["alpha"] = 0.9 #higher alpha makes joint dro more stable
    args_dict["joint"] = True
    args_dict["model_type"] = "dro_eval"
    # args_dict["dlr"] = 0.1
    all_acc = []
    all_loss = []
    all_params = []
    # all_metrics, all_params = med.run_shift_experiment(args_dict)
    # all_acc = [metrics['acc'] for metrics in all_metrics]
    # all_loss = [metrics['loss'] for metrics in all_metrics]
    for shift in shifts:
        args_dict["min_p"] = shift
        acc, loss, params = med.run_shift_experiment(args_dict)
        all_acc.append(acc)
        all_loss.append(loss)
        all_params.append(params)
    print("Joint DRO Eval: "+str(all_acc)+" "+str(all_loss))
    return all_acc, all_loss, all_params

def no_transfer_eval(default_args_dict):
    args_dict = default_args_dict.copy()
    args_dict["model_type"] = "no_transfer_eval"
    all_acc = []
    all_loss = []
    all_params = []
    # all_metrics, all_params = med.run_shift_experiment(args_dict)
    # all_acc = [metrics['acc'] for metrics in all_metrics]
    # all_loss = [metrics['loss'] for metrics in all_metrics]
    for shift in shifts:
        args_dict["min_p"] = shift
        acc, loss, params = med.run_shift_experiment(args_dict)
        all_acc.append(acc)
        all_loss.append(loss)
        all_params.append(params)
    print("Baseline No Transfer Eval: "+str(all_acc)+" "+str(all_loss))
    return all_acc, all_loss, all_params

def is_eval(default_args_dict):
    args_dict = default_args_dict.copy()
    args_dict["model_type"] = "is_eval"
    all_acc = []
    all_loss = []
    all_params = []
    # all_metrics, all_params = med.run_shift_experiment(args_dict)
    # all_acc = [metrics['acc'] for metrics in all_metrics]
    # all_loss = [metrics['loss'] for metrics in all_metrics]
    for shift in shifts:
        args_dict["min_p"] = shift
        acc, loss, params = med.run_shift_experiment(args_dict)
        all_acc.append(acc)
        all_loss.append(loss)
        all_params.append(params)
    print("Baseline IS Eval: "+str(all_acc)+" "+str(all_loss))
    return all_acc, all_loss, all_params

def wis_eval(default_args_dict):
    args_dict = default_args_dict.copy()
    args_dict["model_type"] = "wis_eval"
    all_acc = []
    all_loss = []
    all_params = []
    # all_metrics, all_params = med.run_shift_experiment(args_dict)
    # all_acc = [metrics['acc'] for metrics in all_metrics]
    # all_loss = [metrics['loss'] for metrics in all_metrics]
    for shift in shifts:
        args_dict["min_p"] = shift
        acc, loss, params = med.run_shift_experiment(args_dict)
        all_acc.append(acc)
        all_loss.append(loss)
        all_params.append(params)
    print("Baseline WIS Eval: "+str(all_acc)+" "+str(all_loss))
    return all_acc, all_loss, all_params

def covariate_shift_dro(shifts):
    args_dict = default_args_dict.copy()
    args_dict["model_type"] = "dro"
    args_dict["dlr"] = .0001
    all_acc = []
    all_loss = []
    all_w1 = []
    all_w2 = []
    for shift in shifts:
        args_dict["min_p"] = shift
        acc, loss, params = med.run_shift_experiment(args_dict)
        all_acc.append(acc)
        all_loss.append(loss)
        all_w1.append(float(params[0][0][0]))
        all_w2.append(float(params[0][0][1]))
    print("Covariate Shift DRO: "+str(all_acc)+" "+str(all_loss))
    return all_acc, all_loss, all_w1, all_w2

def subcovariate_shift_dro(shifts):
    args_dict = default_args_dict.copy()
    args_dict["subcov"] = True
    args_dict["model_type"] = "dro"
    args_dict["dlr"] = .0001
    args_dict["nonsep"] = 0
    all_acc = []
    all_loss = []
    all_w1 = []
    all_w2 = []
    for shift in shifts:
        args_dict["min_p"] = shift
        acc, loss, params = med.run_shift_experiment(args_dict)
        all_acc.append(acc)
        all_loss.append(loss)
        all_w1.append(float(params[0][0][0]))
        all_w2.append(float(params[0][0][1]))
    print("Subovariate Shift DRO: "+str(all_acc)+" "+str(all_loss))
    return all_acc, all_loss, all_w1, all_w2

def joint_dro(shifts):
    args_dict = default_args_dict.copy()
    # args_dict["alpha"] = 0.9 #higher alpha makes joint dro more stable
    args_dict["joint"] = True
    args_dict["model_type"] = "dro"
    # args_dict["dlr"] = .001
    args_dict["dlr"] = .0001
    all_acc = []
    all_loss = []
    all_w1 = []
    all_w2 = []
    for shift in shifts:
        args_dict["min_p"] = shift
        acc, loss, params = med.run_shift_experiment(args_dict)
        all_acc.append(acc)
        all_loss.append(loss)
        all_w1.append(float(params[0][0][0]))
        all_w2.append(float(params[0][0][1]))
    print("Joint DRO: "+str(all_acc)+" "+str(all_loss))
    return all_acc, all_loss, all_w1, all_w2

def erm(shifts):
    args_dict = default_args_dict.copy()
    args_dict["alpha"] = 1.0 
    args_dict["model_type"] = "baseline"
    args_dict["dlr"] = .0002
    all_acc = []
    all_loss = []
    all_w1 = []
    all_w2 = []
    for shift in shifts:
        args_dict["min_p"] = shift
        acc, loss, params = med.run_shift_experiment(args_dict)
        all_acc.append(acc)
        all_loss.append(loss)
        all_w1.append(float(params[0][0][0]))
        all_w2.append(float(params[0][0][1]))
    print("Baseline ERM: "+str(all_acc)+" "+str(all_loss))
    return all_acc, all_loss, all_w1, all_w2

def causal(shifts):
    args_dict = default_args_dict.copy()
    args_dict["alpha"] = 1.0 
    args_dict["model_type"] = "causal"
    args_dict["dlr"] = .0002
    # args_dict["nonsep"] = 0
    args_dict["nonsep"] = None
    all_acc = []
    all_loss = []
    all_w1 = []
    all_w2 = []
    for shift in shifts:
        args_dict["min_p"] = shift
        acc, loss, params = med.run_shift_experiment(args_dict)
        all_acc.append(acc)
        all_loss.append(loss)
        all_w1.append(float(params[0][0][0]))
        all_w2.append(float(params[0][0][1]))
    print("Invariant Causal Predictor: "+str(all_acc)+" "+str(all_loss))
    return all_acc, all_loss, all_w1, all_w2

def oracle_dro(shifts):
    args_dict = default_args_dict.copy()
    args_dict["lip"] = 5.0
    args_dict["oracle"] = True
    args_dict["model_type"] = "dro"
    args_dict["dlr"] = .0001
    all_acc = []
    all_loss = []
    all_w1 = []
    all_w2 = []
    for shift in shifts:
        args_dict["min_p"] = shift
        acc, loss, params = med.run_shift_experiment(args_dict)
        all_acc.append(acc)
        all_loss.append(loss)
        all_w1.append(float(params[0][0][0]))
        all_w2.append(float(params[0][0][1]))
    print("Oracle UV-DRO: "+str(all_acc)+" "+str(all_loss))
    return all_acc, all_loss, all_w1, all_w2

# def uv_dro(shifts):
#     args_dict = default_args_dict.copy()
#     args_dict["lip"] = 5.0
#     args_dict["turk"] = True
#     args_dict["penalty"] = 0.0
#     args_dict["model_type"] = "dro"
#     all_acc = []
#     all_loss = []
    # all_w1 = []
    # all_w2 = []
#     for shift in shifts:
#         args_dict["min_p"] = shift
#         acc, loss, params = med.run_shift_experiment(args_dict)
#         all_acc.append(acc)
#         all_loss.append(loss)
        # all_w1.append(float(params[0][0][0]))
        # all_w2.append(float(params[0][0][1]))
#     print("Crowdsource UV-DRO: "+str(acc)+" "+str(loss))
#     return all_acc, all_loss, all_w1, all_w2

# def random_uv_dro(shifts, shuffle_amt=1.0):
#     args_dict = default_args_dict.copy()
#     args_dict["lip"] = 5.0
#     args_dict["turk"] = True
#     args_dict["penalty"] = 0.0
#     args_dict["model_type"] = "dro"
#     args_dict["shuffle"] = shuffle_amt
#     all_acc = []
#     all_loss = []
    # all_w1 = []
    # all_w2 = []
#     for shift in shifts:
#         args_dict["min_p"] = shift
#         acc, loss, params = med.run_shift_experiment(args_dict)
#         all_acc.append(acc)
#         all_loss.append(loss)
        # all_w1.append(float(params[0][0][0]))
        # all_w2.append(float(params[0][0][1]))
#     print("Shuffled Crowdsource UV-DRO: "+str(acc)+" "+str(loss))
#     return all_acc, all_loss, all_w1, all_w2

def single_run(default_args_dict, shifts, run_id):
    file_prefix = default_args_dict["outdir"]+"/"+default_args_dict["filename"]\
            +"_r_"+str(run_id)+"_a_"+str(default_args_dict["alpha"])+"_s_"+str(default_args_dict["nonsep"])
                    # +"_p_"+str(default_args_dict["penalty"])
    default_args_dict["shifts"] = shifts
    default_args_dict["run_id"] = run_id
    no_transfer_eval_acc, no_transfer_eval_loss, no_transfer_eval_params = no_transfer_eval(default_args_dict)
    is_eval_acc, is_eval_loss, is_eval_params = is_eval(default_args_dict)
    wis_eval_acc, wis_eval_loss, wis_eval_params = wis_eval(default_args_dict)
    joint_dro_eval_acc, joint_dro_eval_loss, joint_dro_eval_params = joint_dro_eval(default_args_dict)
    covariate_shift_dro_eval_acc, covariate_shift_dro_eval_loss, covariate_shift_dro_eval_params = covariate_shift_dro_eval(default_args_dict)
    subcovariate_shift_dro_eval_acc, subcovariate_shift_dro_eval_loss, subcovariate_shift_dro_eval_params = subcovariate_shift_dro_eval(default_args_dict)
    oracle_eval_acc, oracle_eval_loss, oracle_eval_params = oracle_eval(default_args_dict)

    with open(file_prefix+"_out.p", "wb") as fw:
        pickle.dump((no_transfer_eval_acc, no_transfer_eval_loss, no_transfer_eval_params), fw)
        pickle.dump((is_eval_acc, is_eval_loss, is_eval_params), fw)
        pickle.dump((wis_eval_acc, wis_eval_loss, wis_eval_params), fw)
        pickle.dump((joint_dro_eval_acc, joint_dro_eval_loss, None), fw)
        pickle.dump((covariate_shift_dro_eval_acc, covariate_shift_dro_eval_loss, None), fw)
        pickle.dump((subcovariate_shift_dro_eval_acc, subcovariate_shift_dro_eval_loss, None), fw)
        pickle.dump((oracle_eval_acc, oracle_eval_loss, None), fw)
        pickle.dump(default_args_dict, fw)
    
def multiple_run(default_args_dict, shifts):
    # shifts = [0.2, 0.4, 0.6, 0.8]

    Parallel(n_jobs=num_cores)(delayed(single_run)(default_args_dict, shifts, run_id) for run_id in range(default_args_dict["num_runs"]))

    print("Done")

def plot_confidence(default_args_dict, shifts):
    num_shifts = len(shifts)
    num_algos = 7
    algo_acc_test = np.ones((num_algos, num_shifts, default_args_dict["num_runs"]))*-1
    algo_loss_test = np.ones((num_algos, num_shifts, default_args_dict["num_runs"]))*-1

    for run_id in range(default_args_dict["num_runs"]):
        
        file_prefix = default_args_dict["outdir"]+"/"+default_args_dict["filename"]\
            +"_r_"+str(run_id)+"_a_"+str(default_args_dict["alpha"])+"_s_"+str(default_args_dict["nonsep"])
                    # +"_p_"+str(default_args_dict["penalty"])

        with open(file_prefix+"_out.p", "rb") as fr:
            (no_transfer_eval_acc, no_transfer_eval_loss, no_transfer_eval_params) = pickle.load(fr)
            (is_eval_acc, is_eval_loss, is_eval_params) = pickle.load(fr)
            (wis_eval_acc, wis_eval_loss, wis_eval_params) = pickle.load(fr)
            (joint_dro_eval_acc, joint_dro_eval_loss, _) = pickle.load(fr)
            (covariate_shift_dro_eval_acc, covariate_shift_dro_eval_loss, _) = pickle.load(fr)
            (subcovariate_shift_dro_eval_acc, subcovariate_shift_dro_eval_loss, _) = pickle.load(fr)
            (oracle_eval_acc, oracle_eval_loss, _) = pickle.load(fr)
            # default_args_dict = pickle.load(fr)
        
        algo_acc_test[0,:,run_id] = no_transfer_eval_acc
        algo_acc_test[1,:,run_id] = is_eval_acc
        algo_acc_test[2,:,run_id] = wis_eval_acc
        algo_acc_test[3,:,run_id] = joint_dro_eval_acc
        algo_acc_test[4,:,run_id] = covariate_shift_dro_eval_acc
        algo_acc_test[5,:,run_id] = subcovariate_shift_dro_eval_acc
        algo_acc_test[6,:,run_id] = oracle_eval_acc

        algo_loss_test[0,:,run_id] = no_transfer_eval_loss
        algo_loss_test[1,:,run_id] = is_eval_loss
        algo_loss_test[2,:,run_id] = wis_eval_loss
        algo_loss_test[3,:,run_id] = joint_dro_eval_loss
        algo_loss_test[4,:,run_id] = covariate_shift_dro_eval_loss
        algo_loss_test[5,:,run_id] = subcovariate_shift_dro_eval_loss
        algo_loss_test[6,:,run_id] = oracle_eval_loss

    mean_algo_acc_test, std_algo_acc_test = np.mean(algo_acc_test, axis=2), np.std(algo_acc_test, axis=2)
    mean_algo_loss_test, std_algo_loss_test = np.mean(algo_loss_test, axis=2), np.std(algo_loss_test, axis=2)
    print("Test value", mean_algo_acc_test, std_algo_acc_test)
    print("Test error from true value", mean_algo_loss_test, std_algo_loss_test)

    file_prefix = default_args_dict["outdir"]+"/"+default_args_dict["filename"]\
        +"_nr_"+str(default_args_dict["num_runs"])+"_a_"+str(default_args_dict["alpha"])+"_s_"+str(default_args_dict["nonsep"])

    #Test Value Plot
    ind = np.arange(mean_algo_acc_test.shape[1])  # the x locations for the groups
    width = 0.1  # the width of the bars

    fig, ax = plt.subplots()
    fig.set_size_inches(26, 16)
    rects1 = ax.errorbar(ind, mean_algo_acc_test[0,:], yerr=std_algo_acc_test[0,:],
                    linewidth=10, capsize=20, capthick=5, alpha=0.5, color="lightseagreen", label='No Transfer')
    rects2 = ax.errorbar(ind, mean_algo_acc_test[1,:], yerr=std_algo_acc_test[1,:],
                    linewidth=10, capsize=20, capthick=5, alpha=0.5, color="lightcoral", label='IPW')
    rects3 = ax.errorbar(ind, mean_algo_acc_test[2,:], yerr=std_algo_acc_test[2,:],
                    linewidth=10, capsize=20, capthick=5, alpha=0.5, color="royalblue", label='Weighted IPW')
    rects4 = ax.errorbar(ind, mean_algo_acc_test[3,:], yerr=std_algo_acc_test[3,:],
                    linewidth=10, capsize=20, capthick=5, alpha=0.5, color="olive", label='Joint DRO')
    rects5 = ax.errorbar(ind, mean_algo_acc_test[4,:], yerr=std_algo_acc_test[4,:],
                    linewidth=10, capsize=20, capthick=5, alpha=0.5, color="orange", label='Covariate DRO')
    rects6 = ax.errorbar(ind, mean_algo_acc_test[5,:], yerr=std_algo_acc_test[5,:],
                    linewidth=10, capsize=20, capthick=5, alpha=0.5, color="mediumvioletred", label='Our Approach')
    rects7 = ax.errorbar(ind, mean_algo_acc_test[6,:], yerr=std_algo_acc_test[6,:],
                    linewidth=10, capsize=20, capthick=5, alpha=0.5, linestyle='dashed', color="cornflowerblue", label='Oracle')
    ax.set_ylim(bottom=-0.4, top=-0.25)
    ax.set_ylabel("Estimated Value in Test", fontsize=70, labelpad=20)
    ax.set_xlabel("Train-Test Shift", fontsize=70)
    # ax.set_title('alpha = '+str(default_args_dict["alpha"]), fontsize=60)
    ax.set_xticks(ind)
    ax.set_xticklabels([np.round(a, decimals=2) for a in shifts], rotation=35)
    fig.gca().tick_params(labelsize=60)
    # ax.legend(loc='best', fontsize=60)
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True, fontsize=20)
    ax.legend(loc="lower left", bbox_to_anchor=(0,1.02,1,0.2), mode="expand", borderaxespad=0, ncol=3, fontsize=60)
    fig.tight_layout()
    plt.savefig(file_prefix+"_val_test_line_conf.png", bbox_inches='tight')
    plt.close()

    #Test Error Plot
    ind = np.arange(mean_algo_loss_test.shape[1])  # the x locations for the groups
    width = 0.1  # the width of the bars

    fig, ax = plt.subplots()
    fig.set_size_inches(26, 16)
    rects1 = ax.errorbar(ind, mean_algo_loss_test[0,:], yerr=std_algo_loss_test[0,:],
                    linewidth=10, capsize=20, capthick=5, alpha=0.5, color="lightseagreen", marker='X', label='Standard')
    rects2 = ax.errorbar(ind, mean_algo_loss_test[1,:], yerr=std_algo_loss_test[1,:],
                    linewidth=10, capsize=20, capthick=5, alpha=0.5, color="lightcoral", label='IPW')
    # rects3 = ax.errorbar(ind, mean_algo_loss_test[2,:], yerr=std_algo_loss_test[2,:],
    #                 linewidth=10, capsize=20, capthick=5, alpha=0.5, color="royalblue", label='Weighted IPW')
    rects4 = ax.errorbar(ind, mean_algo_loss_test[3,:], yerr=std_algo_loss_test[3,:],
                    linewidth=10, capsize=20, capthick=5, alpha=0.5, marker='o', color="olive", label='JointDRO')
    rects5 = ax.errorbar(ind, mean_algo_loss_test[4,:], yerr=std_algo_loss_test[4,:],
                    linewidth=10, capsize=20, capthick=5, alpha=0.5, color="orange", label='RISe(all X)')
    # rects6 = ax.errorbar(ind, mean_algo_loss_test[5,:], yerr=std_algo_loss_test[5,:],
                    # linewidth=10, capsize=20, capthick=5, alpha=0.5, marker='D', color="mediumvioletred", label='RISe(only Z)')
    # rects7 = ax.errorbar(ind, mean_algo_loss_test[6,:], yerr=std_algo_loss_test[6,:],
    #                 linewidth=10, capsize=20, capthick=5, alpha=0.5, color="cornflowerblue", label='Oracle')
    ax.set_ylim(bottom=0.0, top=0.02)
    ax.set_ylabel("MSE in Avg Reward at Test", fontsize=70, labelpad=20)
    ax.set_xlabel("Train-Test Shift", fontsize=70)
    # ax.set_title('alpha = '+str(default_args_dict["alpha"]), fontsize=60)
    ax.set_xticks(ind)
    ax.set_xticklabels([np.round(a, decimals=2) for a in shifts], rotation=35)
    fig.gca().tick_params(labelsize=60)
    ax.legend(loc='best', fontsize=60)
    fig.tight_layout()
    plt.savefig(file_prefix+"_error_test_line_conf.png", bbox_inches='tight')
    plt.close()

def plot():
    shifts = [0.1, 0.2, 0.4, 0.6, 0.8]
    # shifts = [.05, .1, .2, .3, .4, .5, .6, .7, .8]

    causal_acc, causal_loss, causal_w1, causal_w2 = causal(shifts)
    covariate_acc, covariate_loss, covariate_w1, covariate_w2 = covariate_shift_dro(shifts)
    subcovariate_acc, subcovariate_loss, subcovariate_w1, subcovariate_w2 = subcovariate_shift_dro(shifts)
    joint_acc, joint_loss, joint_w1, joint_w2 = joint_dro(shifts)
    oracle_acc, oracle_loss, oracle_w1, oracle_w2 = oracle_dro(shifts)
    erm_acc, erm_loss, erm_w1, erm_w2 = erm(shifts)

    file_prefix = default_args_dict["outdir"]+"/"+default_args_dict["filename"]+"_a_"+str(default_args_dict["alpha"])
    
    with open(file_prefix+"_out.p", "wb") as fw:
        pickle.dump((causal_acc, causal_loss, causal_w1, causal_w2), fw)
        pickle.dump((covariate_acc, covariate_loss, covariate_w1, covariate_w2), fw)
        pickle.dump((subcovariate_acc, subcovariate_loss, subcovariate_w1, subcovariate_w2), fw)
        pickle.dump((joint_acc, joint_loss, joint_w1, joint_w2), fw)
        pickle.dump((oracle_acc, oracle_loss, oracle_w1, oracle_w2), fw)
        pickle.dump((erm_acc, erm_loss, erm_w1, erm_w2), fw)
        pickle.dump(shifts, fw)
        pickle.dump(default_args_dict, fw)

    #Accuracy Plot
    fig = plt.gcf()
    fig.set_size_inches(22, 16)
    plt.plot([a/0.8 for a in shifts], causal_acc, color="lightcoral", linewidth=12, label="Invariant Causal Pred")
    # plt.plot([a/0.8 for a in shifts], covariate_acc, color="mediumvioletred", linewidth=12, label="Covariate Shift DRO")
    plt.plot([a/0.8 for a in shifts], subcovariate_acc, color="brown", linewidth=12, label="Subset DRO")
    plt.plot([a/0.8 for a in shifts], joint_acc, color="olive", linewidth=12, label="Joint DRO")
    # plt.plot([a/0.8 for a in shifts], oracle_acc, color="orange", linewidth=12, label="Oracle DRO")
    plt.plot([a/0.8 for a in shifts], erm_acc, color="cornflowerblue", linewidth=12, label="ERM")
    plt.xlabel("Train-Test Overlap (alpha*)", fontsize=70)
    plt.ylabel("Test Accuracy", fontsize=70, labelpad=20)
    plt.gca().tick_params(labelsize=60)
    plt.legend(loc='best', bbox_to_anchor=(0.3,0.5), fontsize=60)
    plt.title('alpha = '+str(default_args_dict["alpha"]), fontsize=60)
    plt.tight_layout()
    plt.savefig(file_prefix+"_acc_line.png")
    plt.close()



    #Loss Line Plot
    fig = plt.gcf()
    fig.set_size_inches(22, 16)
    plt.plot([a/0.8 for a in shifts], causal_loss, color="lightcoral", linewidth=12, label="Invariant Causal Pred")
    # plt.plot([a/0.8 for a in shifts], covariate_loss, color="mediumvioletred", linewidth=12, label="Covariate Shift DRO")
    plt.plot([a/0.8 for a in shifts], covariate_loss, color="brown", linewidth=12, label="Subset DRO")
    plt.plot([a/0.8 for a in shifts], joint_loss, color="olive", linewidth=12, label="Joint DRO")
    # plt.plot([a/0.8 for a in shifts], oracle_loss, color="orange", linewidth=12, label="Oracle DRO")
    plt.plot([a/0.8 for a in shifts], erm_loss, color="cornflowerblue", linewidth=12, label="ERM")
    plt.xlabel("Train-Test Overlap (alpha*)", fontsize=70)
    plt.ylabel("Test MSE-Loss", fontsize=70, labelpad=20)
    plt.gca().tick_params(labelsize=60)
    plt.legend(loc='best', bbox_to_anchor=(0.3,0.5), fontsize=60)
    plt.title('alpha = '+str(default_args_dict["alpha"]), fontsize=60)
    plt.tight_layout()
    plt.savefig(file_prefix+"_loss_line.png")
    plt.close()


    #Accuracy Bar Plot
    # uvdro_acc = uv_dro(shifts=[0.2])[0][0]
    # random_uvdro_acc = random_uv_dro(shifts=[0.2], shuffle_amt=1.0)[0][0]
    # keys = ['Baseline ERM', 'Baseline DRO', 'Covariate Shift DRO', 'Rand. Crowdsrc UV-DRO', 'Crowdsrc UV-DRO', 'Oracle UV-DRO']
    # acc = [erm_acc[1], joint_acc[1], covariate_acc[1], random_uvdro_acc, uvdro_acc, oracle_acc[1]]
    # keys = ['Invariant Causal Pred', 'ERM', 'Joint DRO', 'Covariate Shift DRO', 'Oracle DRO']
    # acc = [causal_acc[1], erm_acc[1], joint_acc[1], covariate_acc[1], oracle_acc[1]]
    keys = ['Invariant Causal Pred', 'ERM', 'Joint DRO', 'Subset DRO']
    acc = [causal_acc[1], erm_acc[1], joint_acc[1], subcovariate_acc[1]]

    plt.figure(figsize=(40,32))
    # plt.bar([i*50 for i in range(len(keys))], [a/0.8 for a in acc], width=25, align='center', linewidth=3, edgecolor=['black']*len(keys), tick_label=keys, color=('lightcoral', 'cornflowerblue', 'olive', 'mediumvioletred', 'orange'))
    plt.bar([i*50 for i in range(len(keys))], [a/0.8 for a in acc], width=25, align='center', linewidth=3, edgecolor=['black']*len(keys), tick_label=keys, color=('lightcoral', 'cornflowerblue', 'olive', 'brown'))
    plt.xticks([-35, 20, 45, 90, 155, 210], rotation=35, fontsize=120)
    plt.yticks(fontsize=95)
    plt.ylabel("Target Accuracy", fontsize=100, labelpad=20)
    # plt.ylim(0.59, 0.64)
    plt.xlim(-35, 280)
    plt.gca().tick_params(pad=15)
    plt.title('alpha = '+str(default_args_dict["alpha"]), fontsize=60)
    plt.tight_layout()
    plt.savefig(file_prefix+"_acc_bar.png")
    plt.close()

    #Weight Line Plot
    plt.figure(figsize=(22,16))
    causal_weights = [causal_w2[i]/(((causal_w1[i]**2)+(causal_w2[i]**2))**0.5) for i in range(len(causal_w1))]
    covariate_weights = [covariate_w2[i]/(((covariate_w1[i]**2)+(covariate_w2[i]**2))**0.5) for i in range(len(covariate_w1))]
    subcovariate_weights = [subcovariate_w2[i]/(((subcovariate_w1[i]**2)+(subcovariate_w2[i]**2))**0.5) for i in range(len(subcovariate_w1))]
    joint_weights = [joint_w2[i]/(((joint_w1[i]**2)+(joint_w2[i]**2))**0.5) for i in range(len(joint_w1))]
    oracle_weights = [oracle_w2[i]/(((oracle_w1[i]**2)+(oracle_w2[i]**2))**0.5) for i in range(len(oracle_w1))]
    erm_weights = [erm_w2[i]/(((erm_w1[i]**2)+(erm_w2[i]**2))**0.5) for i in range(len(erm_w1))]
    print(erm_weights)
    plt.plot([a/0.8 for a in shifts], causal_weights, c="lightcoral", linewidth=12, label="Invariant Causal Pred.")
    # plt.plot([a/0.8 for a in shifts], covariate_weights, c="mediumvioletred", linewidth=12, label="Covariate Shift DRO")
    plt.plot([a/0.8 for a in shifts], covariate_weights, c="brown", linewidth=12, label="Subset DRO")
    plt.plot([a/0.8 for a in shifts], joint_weights, c="olive", linewidth=12, label="Joint DRO")
    # plt.plot([a/0.8 for a in shifts], oracle_weights, c="orange", linewidth=12, label="Oracle DRO")
    plt.plot([a/0.8 for a in shifts], erm_weights, c="cornflowerblue", linewidth=12, label="ERM")
    plt.xlabel("Train-Test Overlap (alpha*)", fontsize=70)
    plt.ylabel("X2 Weight (Normalized)", fontsize=70)
    plt.xticks(fontsize=60)
    plt.yticks(fontsize=60)
    plt.gca().tick_params(axis='x', pad=15)
    plt.legend(loc='lower right', fontsize=60)
    plt.title('alpha = '+str(default_args_dict["alpha"]), fontsize=60)
    plt.tight_layout()
    plt.savefig(file_prefix+"_w2_line.png")

def replot():
    shifts = [0.1, 0.3, 0.5, 0.7]
    # shifts = [.05, .1, .2, .3, .4, .5, .6, .7, .8]
    global default_args_dict

    file_prefix = default_args_dict["outdir"]+"/"+default_args_dict["filename"]+"_a_"+str(default_args_dict["alpha"])

    with open(file_prefix+"_out.p", "rb") as fr:
        (causal_acc, causal_loss, causal_w1, causal_w2) = pickle.load(fr)
        (covariate_acc, covariate_loss, covariate_w1, covariate_w2) = pickle.load(fr)
        (subcovariate_acc, subcovariate_loss, subcovariate_w1, subcovariate_w2) = pickle.load(fr)
        (joint_acc, joint_loss, joint_w1, joint_w2) = pickle.load(fr)
        (oracle_acc, oracle_loss, oracle_w1, oracle_w2) = pickle.load(fr)
        (erm_acc, erm_loss, erm_w1, erm_w2) = pickle.load(fr)
        shifts = pickle.load(fr)
        default_args_dict = pickle.load(fr)

    #Accuracy Plot
    fig = plt.gcf()
    fig.set_size_inches(22, 16)
    plt.plot([a/0.8 for a in shifts], causal_acc, color="lightcoral", alpha=1.0, linewidth=12, label="Invariant Causal Pred")
    # plt.plot([a/0.8 for a in shifts], covariate_acc, color="mediumvioletred", alpha=1.0, linewidth=12, label="Covariate Shift DRO")
    plt.plot([a/0.8 for a in shifts], subcovariate_acc, color="brown", alpha=1.0, linewidth=12, label="Subset DRO")
    plt.plot([a/0.8 for a in shifts], joint_acc, color="olive", alpha=1.0, linewidth=12, label="Joint DRO")
    # plt.plot([a/0.8 for a in shifts], oracle_acc, color="orange", alpha=1.0, linewidth=12, label="Oracle DRO")
    plt.plot([a/0.8 for a in shifts], erm_acc, color="cornflowerblue", alpha=1.0, linewidth=12, label="ERM")
    plt.xlabel("Train-Test Overlap (alpha*)", fontsize=70)
    plt.ylabel("Test Accuracy", fontsize=70, labelpad=20)
    plt.gca().tick_params(labelsize=60)
    plt.legend(loc='best', bbox_to_anchor=(0.3,0.5), fontsize=60)
    plt.title('alpha = '+str(default_args_dict["alpha"]), fontsize=60)
    plt.tight_layout()
    plt.savefig(file_prefix+"_acc_line.png")
    plt.close()



    #Loss Line Plot
    fig = plt.gcf()
    fig.set_size_inches(22, 16)
    plt.plot([a/0.8 for a in shifts], causal_loss, color="lightcoral", alpha=1.0, linewidth=12, label="Invariant Causal Pred")
    # plt.plot([a/0.8 for a in shifts], covariate_loss, color="mediumvioletred", alpha=1.0, linewidth=12, label="Covariate Shift DRO")
    plt.plot([a/0.8 for a in shifts], covariate_loss, color="brown", alpha=1.0, linewidth=12, label="Subset DRO")
    plt.plot([a/0.8 for a in shifts], joint_loss, color="olive", alpha=1.0, linewidth=12, label="Joint DRO")
    # plt.plot([a/0.8 for a in shifts], oracle_loss, color="orange", alpha=1.0, linewidth=12, label="Oracle DRO")
    plt.plot([a/0.8 for a in shifts], erm_loss, color="cornflowerblue", alpha=1.0, linewidth=12, label="ERM")
    plt.xlabel("Train-Test Overlap (alpha*)", fontsize=70)
    plt.ylabel("Test MSE-Loss", fontsize=70, labelpad=20)
    plt.gca().tick_params(labelsize=60)
    # plt.legend(loc='best', bbox_to_anchor=(0.3,0.5), fontsize=60)
    plt.title('alpha = '+str(default_args_dict["alpha"]), fontsize=60)
    plt.tight_layout()
    plt.savefig(file_prefix+"_loss_line.png")
    plt.close()


    #Accuracy Bar Plot
    # uvdro_acc = uv_dro(shifts=[0.2])[0][0]
    # random_uvdro_acc = random_uv_dro(shifts=[0.2], shuffle_amt=1.0)[0][0]
    # keys = ['Baseline ERM', 'Baseline DRO', 'Covariate Shift DRO', 'Rand. Crowdsrc UV-DRO', 'Crowdsrc UV-DRO', 'Oracle UV-DRO']
    # acc = [erm_acc[1], joint_acc[1], covariate_acc[1], random_uvdro_acc, uvdro_acc, oracle_acc[1]]
    # keys = ['Invariant Causal Pred', 'ERM', 'Joint DRO', 'Covariate Shift DRO', 'Oracle DRO']
    # acc = [causal_acc[1], erm_acc[1], joint_acc[1], covariate_acc[1], oracle_acc[1]]
    keys = ['Invariant Causal Pred', 'ERM', 'Joint DRO', 'Subset DRO']
    acc = [causal_acc[1], erm_acc[1], joint_acc[1], subcovariate_acc[1]]

    plt.figure(figsize=(40,32))
    # plt.bar([i*50 for i in range(len(keys))], [a/0.8 for a in acc], width=25, align='center', linewidth=3, edgecolor=['black']*len(keys), tick_label=keys, color=('lightcoral', 'cornflowerblue', 'olive', 'mediumvioletred', 'orange'))
    plt.bar([i*50 for i in range(len(keys))], [a/0.8 for a in acc], width=25, align='center', linewidth=3, edgecolor=['black']*len(keys), tick_label=keys, color=('lightcoral', 'cornflowerblue', 'olive', 'brown'))
    plt.xticks([-35, 20, 45, 90, 155, 210], rotation=35, fontsize=120)
    plt.yticks(fontsize=95)
    plt.ylabel("Target Accuracy", fontsize=100, labelpad=20)
    # plt.ylim(0.59, 0.64)
    plt.xlim(-35, 280)
    plt.gca().tick_params(pad=15)
    plt.title('alpha = '+str(default_args_dict["alpha"]), fontsize=60)
    plt.tight_layout()
    plt.savefig(file_prefix+"_acc_bar.png")
    plt.close()

    #Weight Line Plot
    plt.figure(figsize=(22,16))
    causal_weights = [causal_w2[i]/(((causal_w1[i]**2)+(causal_w2[i]**2))**0.5) for i in range(len(causal_w1))]
    covariate_weights = [covariate_w2[i]/(((covariate_w1[i]**2)+(covariate_w2[i]**2))**0.5) for i in range(len(covariate_w1))]
    subcovariate_weights = [subcovariate_w2[i]/(((subcovariate_w1[i]**2)+(subcovariate_w2[i]**2))**0.5) for i in range(len(subcovariate_w1))]
    joint_weights = [joint_w2[i]/(((joint_w1[i]**2)+(joint_w2[i]**2))**0.5) for i in range(len(joint_w1))]
    oracle_weights = [oracle_w2[i]/(((oracle_w1[i]**2)+(oracle_w2[i]**2))**0.5) for i in range(len(oracle_w1))]
    erm_weights = [erm_w2[i]/(((erm_w1[i]**2)+(erm_w2[i]**2))**0.5) for i in range(len(erm_w1))]
    print(erm_weights)
    plt.plot([a/0.8 for a in shifts], causal_weights, c="lightcoral", alpha=1.0, linewidth=12, label="Invariant Causal Pred.")
    # plt.plot([a/0.8 for a in shifts], covariate_weights, c="mediumvioletred", alpha=1.0, linewidth=12, label="Covariate Shift DRO")
    plt.plot([a/0.8 for a in shifts], covariate_weights, c="brown", alpha=1.0, linewidth=12, label="Subset DRO")
    plt.plot([a/0.8 for a in shifts], joint_weights, c="olive", alpha=1.0, linewidth=12, label="Joint DRO")
    # plt.plot([a/0.8 for a in shifts], oracle_weights, c="orange", alpha=1.0, linewidth=12, label="Oracle DRO")
    plt.plot([a/0.8 for a in shifts], erm_weights, c="cornflowerblue", alpha=1.0, linewidth=12, label="ERM")
    plt.xlabel("Train-Test Overlap (alpha*)", fontsize=70)
    plt.ylabel("X2 Weight (Normalized)", fontsize=70)
    plt.xticks(fontsize=60)
    plt.yticks(fontsize=60)
    plt.gca().tick_params(axis='x', pad=15)
    plt.legend(loc='lower right', fontsize=60)
    plt.title('alpha = '+str(default_args_dict["alpha"]), fontsize=60)
    plt.tight_layout()
    plt.savefig(file_prefix+"_w2_line.png")

# plot()

# replot()

# shifts = [0.8]
# shifts = [0.2, 0.8]
# shifts = [0.1, 0.9]
shifts = [0.2, 0.6, 0.8, 0.9]
# shifts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]

# single_run(default_args_dict, shifts, 0)

multiple_run(default_args_dict, shifts)

plot_confidence(default_args_dict, shifts)





 

