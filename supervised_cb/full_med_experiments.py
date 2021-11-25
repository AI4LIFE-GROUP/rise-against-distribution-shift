"""
Evaluation code for running ERM and DRO on synthetic dataset with two features - one cause and one effect of outcome.

python full_med_experiments.py --filename medc --alpha 0.2 --lip 1 --dlr 0.01 --ntrain 2000 --ntest 2000 --steps 3000 --kdual 2 --trshift 0.8
"""

import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import med
import argparse
import pickle
import os

parser = argparse.ArgumentParser()
parser.add_argument('--outdir', default='saved_results/medce')
# parser.add_argument('--filename', default='mede')
parser.add_argument('--filename', default='medce')
# parser.add_argument('--filename', default='medc_l2')
# parser.add_argument('--filename', default='medce')
parser.add_argument('--alpha', default=0.2, type=float)
parser.add_argument('--kdual', default=2, type=float)
parser.add_argument('--ntrain', default=2000, type=int)
parser.add_argument('--ntest', default=2000, type=int)
parser.add_argument('--steps', default=3000, type=int)
parser.add_argument('--lip', default=1.0, type=float)
parser.add_argument('--dlr', default=0.01, type=float)
parser.add_argument('--trshift', default=0.8, type=float)
parser.add_argument("--intercept", action='store_true')
args = parser.parse_args()

default_args_dict = {"steps":args.steps, "ntrain":args.ntrain, "alpha":args.alpha, "joint":False, "lip":args.lip, "kdual":args.kdual, "penalty":0.0,
                    "model_type":"baseline", "turk":False, "oracle":False, "shuffle":0.0, "subcov":False, "cov":False,
                    "outdir":args.outdir, "ntest":args.ntest, "sigma":1, "filename":args.filename, "dlr":args.dlr, "trshift":args.trshift,
                    "intercept":args.intercept, "shifts":[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
# shifts = [.05, .1, .2, .3, .4, .5, .6, .7, .8]

print("Output Data Path: "+str(default_args_dict["outdir"]))
print("Alpha: "+str(default_args_dict["alpha"]))
print(default_args_dict)

if not os.path.exists(default_args_dict["outdir"]):
    os.makedirs(default_args_dict["outdir"])

def covariate_shift_dro(default_args_dict):
    args_dict = default_args_dict.copy()
    args_dict["cov"] = True
    args_dict["model_type"] = "dro"
    # args_dict["dlr"] = .01
    # all_w1 = []
    # all_w2 = []
    all_metrics, all_params = med.run_shift_experiment(args_dict)
    all_acc = [metrics['acc'] for metrics in all_metrics]
    all_loss = [metrics['loss'] for metrics in all_metrics]
    # for shift in shifts:
    #     args_dict["min_p"] = shift
    #     acc, loss, params = med.run_shift_experiment(args_dict)
    #     all_acc.append(acc)
    #     all_loss.append(loss)
    #     all_w1.append(float(params[0][0][0]))
    #     all_w2.append(float(params[0][0][1]))
    # print("Covariate Shift DRO: "+str(all_acc)+" "+str(all_loss))
    return all_acc, all_loss, all_params

def subcovariate_shift_dro(default_args_dict):
    args_dict = default_args_dict.copy()
    args_dict["subcov"] = True
    args_dict["model_type"] = "dro"
    # args_dict["dlr"] = .01
    all_metrics, all_params = med.run_shift_experiment(args_dict)
    all_acc = [metrics['acc'] for metrics in all_metrics]
    all_loss = [metrics['loss'] for metrics in all_metrics]
    # print("Subcovariate Shift DRO: "+str(all_acc)+" "+str(all_loss))
    return all_acc, all_loss, all_params

def joint_dro(default_args_dict):
    args_dict = default_args_dict.copy()
    # args_dict["alpha"] = 0.9 #higher alpha makes joint dro more stable
    args_dict["joint"] = True
    args_dict["model_type"] = "dro"
    # args_dict["dlr"] = .01
    all_metrics, all_params = med.run_shift_experiment(args_dict)
    all_acc = [metrics['acc'] for metrics in all_metrics]
    all_loss = [metrics['loss'] for metrics in all_metrics]
    # print("Joint DRO: "+str(all_acc)+" "+str(all_loss))
    return all_acc, all_loss, all_params

def erm(default_args_dict):
    args_dict = default_args_dict.copy()
    # args_dict["alpha"] = 1.0
    args_dict["model_type"] = "baseline"
    # args_dict["dlr"] = .01
    # all_acc = []
    # all_loss = []
    # all_params = []
    all_metrics, all_params = med.run_shift_experiment(args_dict)
    all_acc = [metrics['acc'] for metrics in all_metrics]
    all_loss = [metrics['loss'] for metrics in all_metrics]
    # print("Baseline ERM: "+str(all_acc)+" "+str(all_loss))
    return all_acc, all_loss, all_params

def causal(default_args_dict):
    args_dict = default_args_dict.copy()
    # args_dict["alpha"] = 1.0
    args_dict["model_type"] = "causal"
    # args_dict["dlr"] = .01
    all_metrics, all_params = med.run_shift_experiment(args_dict)
    all_acc = [metrics['acc'] for metrics in all_metrics]
    all_loss = [metrics['loss'] for metrics in all_metrics]
    # print("Invariant Causal Predictor: "+str(all_acc)+" "+str(all_loss))
    return all_acc, all_loss, all_params

# def oracle_dro(shifts):
#     args_dict = default_args_dict.copy()
#     args_dict["lip"] = 5.0
#     args_dict["oracle"] = True
#     args_dict["model_type"] = "dro"
#     args_dict["dlr"] = .0001
#     all_acc = []
#     all_loss = []
#     all_w1 = []
#     all_w2 = []
#     for shift in shifts:
#         args_dict["min_p"] = shift
#         acc, loss, params = med.run_shift_experiment(args_dict)
#         all_acc.append(acc)
#         all_loss.append(loss)
#         all_w1.append(float(params[0][0][0]))
#         all_w2.append(float(params[0][0][1]))
#     print("Oracle UV-DRO: "+str(all_acc)+" "+str(all_loss))
#     return all_acc, all_loss, all_w1, all_w2

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

def plot(default_args_dict):
    shifts = default_args_dict["shifts"]
    causal_acc, causal_loss, causal_params = causal(default_args_dict)
    covariate_acc, covariate_loss, covariate_params = covariate_shift_dro(default_args_dict)
    subcovariate_acc, subcovariate_loss, subcovariate_params = subcovariate_shift_dro(default_args_dict)
    joint_acc, joint_loss, joint_params = joint_dro(default_args_dict)
    # oracle_acc, oracle_loss, oracle_params = oracle_dro(default_args_dict)
    erm_acc, erm_loss, erm_params = erm(default_args_dict)

    file_prefix = "{}/{}_a{}_lip{}_lr{}_ntr{}_nte{}_trs{}_mins{}_maxs{}_p{}_it{}_int{}".format(default_args_dict["outdir"],\
                    default_args_dict["filename"],default_args_dict["alpha"],default_args_dict["lip"],\
                    default_args_dict["dlr"],default_args_dict["ntrain"],default_args_dict["ntest"],\
                    default_args_dict["trshift"],min(default_args_dict["shifts"]),max(default_args_dict["shifts"]),\
                    default_args_dict["kdual"],default_args_dict["steps"],default_args_dict["intercept"])
    
    if not os.path.exists(file_prefix):
        os.makedirs(file_prefix)
    
    with open("{}/result.p".format(file_prefix), "wb") as fw:
        pickle.dump((causal_acc, causal_loss, causal_params), fw)
        pickle.dump((covariate_acc, covariate_loss, covariate_params), fw)
        pickle.dump((subcovariate_acc, subcovariate_loss, subcovariate_params), fw)
        pickle.dump((joint_acc, joint_loss, joint_params), fw)
        # pickle.dump((oracle_acc, oracle_loss, oracle_params), fw)
        pickle.dump((erm_acc, erm_loss, erm_params), fw)
        pickle.dump(shifts, fw)
        pickle.dump(default_args_dict, fw)

    #Accuracy Plot
    fig = plt.gcf()
    fig.set_size_inches(22, 16)
    plt.plot([a for a in shifts], causal_acc, color="lightcoral", linewidth=12, label="Invariant Causal Pred")
    plt.plot([a for a in shifts], covariate_acc, color="mediumvioletred", linewidth=12, label="Covariate Shift DRO")
    plt.plot([a for a in shifts], subcovariate_acc, color="brown", linewidth=12, label="Subset DRO")
    plt.plot([a for a in shifts], joint_acc, color="olive", linewidth=12, label="Joint DRO")
    # plt.plot([a for a in shifts], oracle_acc, color="orange", linewidth=12, label="Oracle DRO")
    plt.plot([a for a in shifts], erm_acc, color="cornflowerblue", linewidth=12, label="ERM")
    plt.xlabel("Train-Test Shift", fontsize=70)
    plt.ylabel("Test Accuracy", fontsize=70, labelpad=20)
    plt.gca().tick_params(labelsize=60)
    plt.legend(loc='best', bbox_to_anchor=(0.3,0.5), fontsize=60)
    plt.title('alpha = '+str(default_args_dict["alpha"]), fontsize=60)
    plt.tight_layout()
    plt.savefig("{}/acc_line.png".format(file_prefix), bbox_inches='tight')
    plt.close()



    #Loss Line Plot
    fig = plt.gcf()
    fig.set_size_inches(22, 16)
    plt.plot([a for a in shifts], erm_loss, color="cornflowerblue", markersize=28, marker='X', linewidth=12, label="ERM", alpha=0.5)
    plt.plot([a for a in shifts], causal_loss, color="lightcoral", markersize=28, marker='s', linewidth=12, label="Causal", alpha=0.5)
    plt.plot([a for a in shifts], joint_loss, color="olive", markersize=28, marker='o', linewidth=12, label="Joint DRO", alpha=0.5)
    plt.plot([a for a in shifts], covariate_loss, color="mediumvioletred", markersize=28, marker='D', linewidth=12, label="Our Approach Covariate DRO", alpha=0.5)
    plt.plot([a for a in shifts], subcovariate_loss, color="brown", markersize=28, marker='*', linewidth=12, label="Our Approach Subcovariate DRO", alpha=0.5)
    # plt.plot([a for a in shifts], oracle_loss, color="orange", markersize=28, linewidth=12, label="Oracle DRO")
    plt.xlabel("Train-Test Shift", fontsize=70)
    plt.ylabel("Test Mean Squared Error", fontsize=70, labelpad=20)
    plt.gca().tick_params(labelsize=60)
    plt.legend(loc='best', fontsize=30)
    plt.title('alpha_train={}, train_prop={},\n\
                L={}, lr={}, ntrain={}, ntest={}'.format(default_args_dict["alpha"],\
                default_args_dict["trshift"], default_args_dict["lip"], default_args_dict["dlr"],\
                default_args_dict["ntrain"], default_args_dict["ntest"]), fontsize=30)
    plt.tight_layout()
    plt.savefig("{}/loss_line.png".format(file_prefix), bbox_inches='tight')
    plt.close()


    # #Accuracy Bar Plot
    # # uvdro_acc = uv_dro(shifts=[0.2])[0][0]
    # # random_uvdro_acc = random_uv_dro(shifts=[0.2], shuffle_amt=1.0)[0][0]
    # # keys = ['Baseline ERM', 'Baseline DRO', 'Covariate Shift DRO', 'Rand. Crowdsrc UV-DRO', 'Crowdsrc UV-DRO', 'Oracle UV-DRO']
    # # acc = [erm_acc[1], joint_acc[1], covariate_acc[1], random_uvdro_acc, uvdro_acc, oracle_acc[1]]
    # # keys = ['Invariant Causal Pred', 'ERM', 'Joint DRO', 'Covariate Shift DRO', 'Oracle DRO']
    # # acc = [causal_acc[1], erm_acc[1], joint_acc[1], covariate_acc[1], oracle_acc[1]]
    # keys = ['Invariant Causal Pred', 'ERM', 'Joint DRO', 'Subset DRO']
    # acc = [causal_acc[1], erm_acc[1], joint_acc[1], subcovariate_acc[1]]

    # plt.figure(figsize=(40,32))
    # # plt.bar([i*50 for i in range(len(keys))], [a/0.8 for a in acc], width=25, align='center', linewidth=3, edgecolor=['black']*len(keys), tick_label=keys, color=('lightcoral', 'cornflowerblue', 'olive', 'mediumvioletred', 'orange'))
    # plt.bar([i*50 for i in range(len(keys))], [a/0.8 for a in acc], width=25, align='center', linewidth=3, edgecolor=['black']*len(keys), tick_label=keys, color=('lightcoral', 'cornflowerblue', 'olive', 'brown'))
    # plt.xticks([-35, 20, 45, 90, 155, 210], rotation=35, fontsize=120)
    # plt.yticks(fontsize=95)
    # plt.ylabel("Target Accuracy", fontsize=100, labelpad=20)
    # # plt.ylim(0.59, 0.64)
    # plt.xlim(-35, 280)
    # plt.gca().tick_params(pad=15)
    # plt.title('alpha = '+str(default_args_dict["alpha"]), fontsize=60)
    # plt.tight_layout()
    # plt.savefig(file_prefix+"_acc_bar.png")
    # plt.close()

    #Weight Line Plot
    # plt.figure(figsize=(22,16))
    # causal_weights = [causal_w2[i]/(((causal_w1[i]**2)+(causal_w2[i]**2))**0.5) for i in range(len(causal_w1))]
    # # covariate_weights = [covariate_w2[i]/(((covariate_w1[i]**2)+(covariate_w2[i]**2))**0.5) for i in range(len(covariate_w1))]
    # subcovariate_weights = [subcovariate_w2[i]/(((subcovariate_w1[i]**2)+(subcovariate_w2[i]**2))**0.5) for i in range(len(subcovariate_w1))]
    # joint_weights = [joint_w2[i]/(((joint_w1[i]**2)+(joint_w2[i]**2))**0.5) for i in range(len(joint_w1))]
    # # oracle_weights = [oracle_w2[i]/(((oracle_w1[i]**2)+(oracle_w2[i]**2))**0.5) for i in range(len(oracle_w1))]
    # erm_weights = [erm_w2[i]/(((erm_w1[i]**2)+(erm_w2[i]**2))**0.5) for i in range(len(erm_w1))]
    # print(erm_weights)
    # plt.plot([a/0.8 for a in shifts], erm_weights, c="cornflowerblue", linewidth=12, label="ERM")
    # plt.plot([a/0.8 for a in shifts], causal_weights, c="lightcoral", linewidth=12, label="Causal")
    # # plt.plot([a/0.8 for a in shifts], covariate_weights, c="brown", linewidth=12, label="Covariate Shift DRO")
    # plt.plot([a/0.8 for a in shifts], joint_weights, c="olive", linewidth=12, label="Joint DRO")
    # plt.plot([a/0.8 for a in shifts], subcovariate_weights, c="mediumvioletred", linewidth=12, label="Our Approach")
    # # plt.plot([a/0.8 for a in shifts], oracle_weights, c="orange", linewidth=12, label="Oracle DRO")
    # plt.xlabel("Train-Test Overlap", fontsize=70)
    # plt.ylabel("X2 Weight (Normalized)", fontsize=70)
    # plt.xticks(fontsize=60)
    # plt.yticks(fontsize=60)
    # plt.gca().tick_params(axis='x', pad=15)
    # plt.legend(loc='lower right', fontsize=60)
    # plt.title('alpha = '+str(default_args_dict["alpha"]), fontsize=60)
    # plt.tight_layout()
    # plt.savefig(file_prefix+"_w2_line.png")

def replot(default_args_dict):
    shifts = default_args_dict["shifts"]
    file_prefix = "{}/{}_a{}_lip{}_lr{}_ntr{}_nte{}_trs{}_mins{}_maxs{}_p{}_it{}_int{}".format(default_args_dict["outdir"],\
                    default_args_dict["filename"],default_args_dict["alpha"],default_args_dict["lip"],\
                    default_args_dict["dlr"],default_args_dict["ntrain"],default_args_dict["ntest"],\
                    default_args_dict["trshift"],min(default_args_dict["shifts"]),max(default_args_dict["shifts"]),\
                    default_args_dict["kdual"],default_args_dict["steps"],default_args_dict["intercept"])

    with open("{}/result.p".format(file_prefix), "rb") as fr:
        (causal_acc, causal_loss, causal_params) = pickle.load(fr)
        (covariate_acc, covariate_loss, covariate_params) = pickle.load(fr)
        (subcovariate_acc, subcovariate_loss, subcovariate_params) = pickle.load(fr)
        (joint_acc, joint_loss, joint_params) = pickle.load(fr)
        # (oracle_acc, oracle_loss, oracle_params) = pickle.load(fr)
        (erm_acc, erm_loss, erm_params) = pickle.load(fr)
        shifts = pickle.load(fr)
        default_args_dict = pickle.load(fr)

    # #Accuracy Plot
    # fig = plt.gcf()
    # fig.set_size_inches(22, 16)
    # # plt.plot([a/0.8 for a in shifts], causal_acc, color="lightcoral", linewidth=12, label="Invariant Causal Pred")
    # # plt.plot([a/0.8 for a in shifts], covariate_acc, color="mediumvioletred", linewidth=12, label="Covariate Shift DRO")
    # plt.plot([a/0.8 for a in shifts], joint_acc, color="olive", linewidth=12, label="Joint DRO")
    # # plt.plot([a/0.8 for a in shifts], oracle_acc, color="orange", linewidth=12, label="Oracle DRO")
    # plt.plot([a/0.8 for a in shifts], erm_acc, color="cornflowerblue", linewidth=12, label="ERM")
    # plt.plot([a/0.8 for a in shifts], subcovariate_acc, color="mediumvioletred", linewidth=12, label="Our Approach")
    # plt.xlabel("Train-Test Overlap (alpha*)", fontsize=70)
    # plt.ylabel("Test Accuracy", fontsize=70, labelpad=20)
    # plt.gca().tick_params(labelsize=60)
    # plt.legend(loc='best', bbox_to_anchor=(0.3,0.5), fontsize=60)
    # plt.title('alpha = '+str(default_args_dict["alpha"]), fontsize=60)
    # plt.tight_layout()
    # plt.savefig(file_prefix+"_acc_line.png")
    # plt.close()



    #Loss Line Plot
    fig = plt.gcf()
    fig.set_size_inches(22, 16)
    plt.plot([a for a in shifts], erm_loss, color="cornflowerblue", markersize=28, marker='X', linewidth=12, label="ERM", alpha=0.5)
    plt.plot([a for a in shifts], causal_loss, color="lightcoral", markersize=28, marker='s', linewidth=12, label="Causal", alpha=0.5)
    plt.plot([a for a in shifts], joint_loss, color="olive", markersize=28, marker='o', linewidth=12, label="Joint DRO", alpha=0.5)
    plt.plot([a for a in shifts], covariate_loss, color="brown", markersize=28, marker='*', linewidth=12, label="Our Approach Covariate DRO", alpha=0.5)
    plt.plot([a for a in shifts], subcovariate_loss, color="mediumvioletred", markersize=28, marker='D', linewidth=12, label="Our Approach Subcovariate DRO", alpha=0.5)
    # plt.plot([a for a in shifts], oracle_loss, color="orange", markersize=28, linewidth=12, label="Oracle DRO")
    plt.xlabel("Train-Test Shift", fontsize=70)
    plt.ylabel("Test Mean Squared Error", fontsize=70)
    plt.gca().tick_params(labelsize=60)
    plt.legend(loc='best', fontsize=30)
    plt.title('alpha_train={}, train_prop={},\n\
                L={}, lr={}, ntrain={}, ntest={}'.format(default_args_dict["alpha"],\
                default_args_dict["trshift"], default_args_dict["lip"], default_args_dict["dlr"],\
                default_args_dict["ntrain"], default_args_dict["ntest"]), fontsize=30)
    plt.tight_layout()
    plt.savefig("{}/loss_line.png".format(file_prefix), bbox_inches='tight')
    plt.close()


    # #Accuracy Bar Plot
    # # uvdro_acc = uv_dro(shifts=[0.2])[0][0]
    # # random_uvdro_acc = random_uv_dro(shifts=[0.2], shuffle_amt=1.0)[0][0]
    # # keys = ['Baseline ERM', 'Baseline DRO', 'Covariate Shift DRO', 'Rand. Crowdsrc UV-DRO', 'Crowdsrc UV-DRO', 'Oracle UV-DRO']
    # # acc = [erm_acc[1], joint_acc[1], covariate_acc[1], random_uvdro_acc, uvdro_acc, oracle_acc[1]]
    # # keys = ['Invariant Causal Pred', 'ERM', 'Joint DRO', 'Covariate Shift DRO', 'Oracle DRO']
    # # acc = [causal_acc[1], erm_acc[1], joint_acc[1], covariate_acc[1], oracle_acc[1]]
    # keys = ['Causal Pred', 'ERM', 'Joint DRO', 'Subset DRO']
    # acc = [causal_acc[1], erm_acc[1], joint_acc[1], subcovariate_acc[1]]

    # plt.figure(figsize=(40,32))
    # # plt.bar([i*50 for i in range(len(keys))], [a/0.8 for a in acc], width=25, align='center', linewidth=3, edgecolor=['black']*len(keys), tick_label=keys, color=('lightcoral', 'cornflowerblue', 'olive', 'mediumvioletred', 'orange'))
    # plt.bar([i*50 for i in range(len(keys))], [a/0.8 for a in acc], width=25, align='center', linewidth=3, edgecolor=['black']*len(keys), tick_label=keys, color=('lightcoral', 'cornflowerblue', 'olive', 'brown'))
    # plt.xticks([-35, 20, 45, 90, 155, 210], rotation=35, fontsize=120)
    # plt.yticks(fontsize=95)
    # plt.ylabel("Target Accuracy", fontsize=100, labelpad=20)
    # # plt.ylim(0.59, 0.64)
    # plt.xlim(-35, 280)
    # plt.gca().tick_params(pad=15)
    # plt.title('alpha = '+str(default_args_dict["alpha"]), fontsize=60)
    # plt.tight_layout()
    # plt.savefig(file_prefix+"_acc_bar.png")
    # plt.close()

    # #Weight Line Plot
    # plt.figure(figsize=(22,16))
    # causal_weights = [causal_w2[i]/(((causal_w1[i]**2)+(causal_w2[i]**2))**0.5) for i in range(len(causal_w1))]
    # covariate_weights = [covariate_w2[i]/(((covariate_w1[i]**2)+(covariate_w2[i]**2))**0.5) for i in range(len(covariate_w1))]
    # subcovariate_weights = [subcovariate_w2[i]/(((subcovariate_w1[i]**2)+(subcovariate_w2[i]**2))**0.5) for i in range(len(subcovariate_w1))]
    # joint_weights = [joint_w2[i]/(((joint_w1[i]**2)+(joint_w2[i]**2))**0.5) for i in range(len(joint_w1))]
    # oracle_weights = [oracle_w2[i]/(((oracle_w1[i]**2)+(oracle_w2[i]**2))**0.5) for i in range(len(oracle_w1))]
    # erm_weights = [erm_w2[i]/(((erm_w1[i]**2)+(erm_w2[i]**2))**0.5) for i in range(len(erm_w1))]
    # print(erm_weights)
    # plt.plot([a/0.8 for a in shifts], causal_weights, c="lightcoral", alpha=1.0, linewidth=12, label="Invariant Causal Pred.")
    # # plt.plot([a/0.8 for a in shifts], covariate_weights, c="mediumvioletred", alpha=1.0, linewidth=12, label="Covariate Shift DRO")
    # plt.plot([a/0.8 for a in shifts], covariate_weights, c="brown", alpha=1.0, linewidth=12, label="Subset DRO")
    # plt.plot([a/0.8 for a in shifts], joint_weights, c="olive", alpha=1.0, linewidth=12, label="Joint DRO")
    # # plt.plot([a/0.8 for a in shifts], oracle_weights, c="orange", alpha=1.0, linewidth=12, label="Oracle DRO")
    # plt.plot([a/0.8 for a in shifts], erm_weights, c="cornflowerblue", alpha=1.0, linewidth=12, label="ERM")
    # plt.xlabel("Train-Test Overlap (alpha*)", fontsize=70)
    # plt.ylabel("X2 Weight (Normalized)", fontsize=70)
    # plt.xticks(fontsize=60)
    # plt.yticks(fontsize=60)
    # plt.gca().tick_params(axis='x', pad=15)
    # plt.legend(loc='lower right', fontsize=60)
    # plt.title('alpha = '+str(default_args_dict["alpha"]), fontsize=60)
    # plt.tight_layout()
    # plt.savefig(file_prefix+"_w2_line.png")

plot(default_args_dict)

# replot(shifts, default_args_dict)
    







 

