import numpy as np
import pandas as pd
import pickle
from copy import deepcopy
import scipy
from scipy import stats
from sklearn.utils import resample 
import torch
from utils import *
from model import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
torch.manual_seed(1)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def get_x_distances(train_data):
    """
    Returns distances on observed covariates x for covariate shift DRO. 
    :param train_data: train data over which distances are calculated 
    :return: square distance matrix of width/height equal to data length
    """
    new_turk_dists = np.zeros((train_data[0].shape[0], train_data[0].shape[0]))
    for i in range(new_turk_dists.shape[0]):
        for j in range(i, new_turk_dists.shape[0]):
            xi = train_data[0][i]
            xj = train_data[0][j]
            x_dist =  np.linalg.norm(xi-xj)*1.0 #feature vector distance
            new_turk_dists[i][j] = x_dist
            new_turk_dists[j][i] = x_dist
    return new_turk_dists

def get_oracle_distances(all_data):
    """
    Returns oracle distances based on location confounder. 
    :param all_data: train data over which distances are calculated 
    :return: square distance matrix of width/height equal to data length
    """
    num_x = all_data[0].shape[0]
    distances_matrix = np.zeros((num_x, num_x))
    for i in range(0, num_x):
        for j in range(i, num_x):
            if(all_data[2][i] != all_data[2][j]):
                distances_matrix[i][j] = 1.0
                distances_matrix[j][i] = 1.0
    return distances_matrix

def get_annotation_distances(train_data, shuffle_val, dists_file="data/sf/sf_dists.p"):
    """
    Returns distances using annotations over training data. 
    :param train_data: train data over which distances are calculated 
    :param shuffle_val: degree to randomly shuffle annotation distances 
    :return: square distance matrix of width/height equal to data length
    """
    turk_matrix_filename = dists_file
    with open(turk_matrix_filename,'rb') as handle:
        turk_matrix = pickle.load(handle)
    if(train_data[0].shape[0] != turk_matrix.shape[0]):
        print("***TURK DATA MATRIX MISMATCH***")
        return -1
    turk_matrix = shuffle_turk_distances(turk_matrix, shuffle_val)
    turk_matrix = smooth_distances(turk_matrix)
    return turk_matrix

def generate_data(train_env=None, target_env=None, train_size=1000, target_size=10000, train_policy=None, target_policy=None):
    """
    Generates synthetic data for simulated medical diagnosis with two causal features. 
    :param: train_env / target_env - train / target environment
    :param: train_size -  size of training data
    :param: target_size - size of groundtruth evaluation data
    :param: train_policy / target_policy - behaviour / target policy
    :return: train data features, train data labels, target data features, target data labels
    """
    train_data_x, train_data_y, train_data_a, train_data_p = [], [], [], []
    target_data_x, target_data_y = [], []
    for i in range(train_size): #Synthetic Train Data
        # np.random.seed(i)
        x = train_env.observe()
        a, p_a = train_policy.sample(torch.Tensor(x)) # action a, probability of a=1
        y = train_env.reward(x, a)
        train_data_x.append(x)
        train_data_a.append(a)
        train_data_p.append(p_a)
        train_data_y.append(y)
    for i in range(target_size): #Synthetic Shifted Target Data
        # np.random.seed(2*i+1)
        x = target_env.observe()
        a, _ = target_policy.sample(torch.Tensor(x)) # action a
        y = target_env.reward(x, a)
        target_data_x.append(x)
        target_data_y.append(y)
    return torch.FloatTensor(train_data_x), torch.FloatTensor(train_data_y), torch.FloatTensor(train_data_a), torch.FloatTensor(train_data_p), torch.FloatTensor(target_data_x), torch.FloatTensor(target_data_y)

def train_erm(train_data, valid_data, l2_reg=0.001, lr=0.01, steps=5000, log=True):
    # model = LogisticRegression(train_data[0].shape[1], 2)
    model = LinearRegression(2, 1)
    model = model.to(device)
    # loss_function = nn.NLLLoss()
    loss_function = nn.MSELoss()

    reg_function = L2_reg(l2_reg)
    # labels = torch.squeeze(torch.LongTensor(train_data[1])).to(device)
    labels = torch.squeeze(train_data[1]).to(device)
    
    # optimizer = optim.Adagrad(list(model.parameters()), lr=lr)
    optimizer = optim.ASGD(list(model.parameters())+list(loss_function.parameters()), lr=lr)

    for step in range(steps):
        optimizer.zero_grad()
        log_probs = torch.squeeze(model(torch.FloatTensor(train_data[0]).to(device)))
        model_loss = loss_function(log_probs, labels)
        loss = model_loss+reg_function(model)
        loss.backward()
        optimizer.step()
    return model

def train_dro(train_data, valid_data, args_dict):
    # model = LogisticRegression(train_data[0].shape[1], 2)
    model = LinearRegression(2, 1)
    model = model.to(device)

    b_init = torch.zeros((train_data[0].shape[0], train_data[0].shape[0]))
    train_distances = []
    if(args_dict["turk"]): #UV-DRO with Crowdsousrcing
        train_distances = get_annotation_distances(train_data, shuffle_val=args_dict["shuffle"], dists_file=args_dict["anns_dists"])
    elif(args_dict["oracle"]): #Oracle UV-DRO
        train_distances = get_oracle_distances(train_data)
    elif(args_dict["subcov"]): #Subset DRO
        train_data_sub = deepcopy(train_data)
        train_data_sub = (train_data_sub[0][:, args_dict["nonsep"]], train_data_sub[1], train_data_sub[2]) # keep non-sep covariate
        train_distances = get_x_distances(train_data_sub)
    else: #Covariate Shift DRO
        train_distances = get_x_distances(train_data)
    loss_function = Metric_loss_reg(args_dict, train_distances, b_init, device=device)
    loss_function = loss_function.to(device)
    reg_function = L2_reg(args_dict["penalty"])
    reg_function = reg_function.to(device)
    # data_target = make_target(train_data[1], 2, long_tensor=False)
    # data_target = torch.LongTensor(train_data[1])
    data_target = train_data[1].to(device)

    # optimizer = optim.Adagrad(list(model.parameters())+list(loss_function.parameters()), lr=args_dict["dlr"], initial_accumulator_value=10.0)
    optimizer = optim.ASGD(list(model.parameters())+list(loss_function.parameters()), lr=args_dict["dlr"])

    loss_function.project()
    for step in range(args_dict["steps"]):
        # labels = torch.LongTensor(train_data[1]).to(device)
        labels = train_data[1].to(device)

        optimizer.zero_grad()
        log_probs = torch.squeeze(model(torch.FloatTensor(train_data[0]).to(device)))
        model_loss = loss_function(log_probs, torch.squeeze(data_target))
        loss = model_loss+reg_function(model)
        loss.backward()
        optimizer.step()
        loss_function.project()
    return model

def train_dro_eval(train_imp_weight, train_data, valid_data, args_dict):
    # model = LogisticRegression(train_data[0].shape[1], 2)
    # model = LinearRegression(2, 1)
    # model = model.to(device)
    train_imp_weight = train_imp_weight.to(device)

    b_init = torch.zeros((train_data[0].shape[0], train_data[0].shape[0]))
    train_distances = []
    if(args_dict["turk"]): #UV-DRO with Crowdsousrcing
        train_distances = get_annotation_distances(train_data, shuffle_val=args_dict["shuffle"], dists_file=args_dict["anns_dists"])
    elif(args_dict["oracle"]): #Oracle UV-DRO
        train_distances = get_oracle_distances(train_data)
    elif(args_dict["subcov"]): #Subset DRO
        train_data_sub = deepcopy(train_data)
        train_data_sub = (train_data_sub[0][:, args_dict["nonsep"]], train_data_sub[1], train_data_sub[2]) # keep non-sep covariate
        train_distances = get_x_distances(train_data_sub)
    elif(args_dict["joint"]): #Joint DRO
        train_distances = 0
    else: #Covariate Shift DRO
        train_distances = get_x_distances(train_data)
    loss_function = Metric_loss_reg(args_dict, train_distances, b_init, device=device)
    loss_function = loss_function.to(device)
    reg_function = L2_reg(args_dict["penalty"])
    reg_function = reg_function.to(device)
    # data_target = make_target(train_data[1], 2, long_tensor=False)
    # data_target = torch.LongTensor(train_data[1])
    data_target = train_data[1].to(device)
    iw_data_target = -1 * torch.mul(train_imp_weight, data_target)
    iw_data_target = iw_data_target.detach()
    # optimizer = optim.Adagrad(list(model.parameters())+list(loss_function.parameters()), lr=args_dict["dlr"], initial_accumulator_value=10.0)
    optimizer = optim.Adagrad(list(loss_function.parameters()), lr=args_dict["dlr"], initial_accumulator_value=10.0)
    # optimizer = optim.ASGD(list(model.parameters())+list(loss_function.parameters()), lr=args_dict["dlr"])
    # optimizer = optim.ASGD(list(loss_function.parameters()), lr=args_dict["dlr"])

    loss_function.project()

    if args_dict["joint"]:
        log_probs = torch.squeeze(iw_data_target.to(device))
        loss = loss_function(log_probs, None)
    else:
        for step in range(args_dict["steps"]):
            # labels = torch.LongTensor(train_data[1]).to(device)
            # labels = train_data[1].to(device)

            optimizer.zero_grad()
            # log_probs = torch.squeeze(model(torch.FloatTensor(train_data[0]).to(device)))
            log_probs = torch.squeeze(iw_data_target.to(device))
            # model_loss = loss_function(log_probs, torch.squeeze(data_target))
            loss = loss_function(log_probs, None)
            # loss = model_loss+reg_function(model)
            loss.backward()
            optimizer.step()
            loss_function.project()
            if step%50==0:
                print("loss", step, loss.item())
    return -1*loss.item(), None

def imp_weight(train_x, train_data_a, train_data_p, target_policy):
    _, target_p = target_policy.sample(train_x)
    target_p = torch.squeeze(target_p)
    pi_e = torch.mul(train_data_a, target_p) + torch.mul(1-train_data_a, 1-target_p)
    pi_b = torch.mul(train_data_a, train_data_p) + torch.mul(1-train_data_a, 1-train_data_p)
    return torch.div(pi_e, pi_b)

def evaluate(test_data, model):
    log_probs = torch.squeeze(model(torch.FloatTensor(test_data[0]).to(device)))
    # loss_function = nn.NLLLoss()
    loss_function = nn.MSELoss()

    # labels = make_target(test_data[1], n=2, long_tensor=True)
    labels = torch.squeeze(torch.FloatTensor(test_data[1]).to(device))
    
    loss = loss_function(log_probs, labels)
    # acc = 0
    acc = loss.data.item()

    # for i in range(len(test_data[0])):
    #     predict_y = torch.max(model(torch.FloatTensor([test_data[0][i]]).to(device)), 1)[1].item()
    #     if(predict_y == int(test_data[1][i])):
    #         acc +=1
    # return acc/len(test_data[0]), loss 
    return acc, loss 

def evaluate_sample_average(target_y):
    est_value = target_y.mean().data.item()
    return est_value

def metrics(est_value, true_value):
    return (true_value - est_value)**2

def run_shift_experiment(args_dict):
    # Policies - with one layer logistic regression
    input_size = args_dict["ndim"]
    train_policy = LogisticRegressionPolicy(input_size, num_classes=1,\
                        weight_init=torch.tensor([0.1,0.1]), bias_init=torch.tensor([-1]))
    target_policy = LogisticRegressionPolicy(input_size, num_classes=1,\
                        weight_init=torch.tensor([0.1,0.1]), bias_init=torch.tensor([-0.5]))

    shift_prop = args_dict['min_p']

    y_weight_a0, y_weight_a1 = np.array([0.1,0.1]), np.array([0.1,0.5])
    # Environments
    # train_env = ToyGaussianEnv(y_weight_a0, y_weight_a1,
    #                             c_values = [1, -1], shift_probs=[1.0-shift_prop, shift_prop],
    #                             sigma_x=args_dict["sigma_x"], sigma_y=args_dict["sigma_y"])
    # target_env = ToyGaussianEnv(y_weight_a0, y_weight_a1,
    #                             c_values = [-1, 1], shift_probs=[0.8, 0.2],
    #                             sigma_x=args_dict["sigma_x"], sigma_y=args_dict["sigma_y"])
    train_env = ToyGaussianEnv(y_weight_a0, y_weight_a1,
                                c_values = [-1, 1], shift_probs=[0.1, 0.9],
                                sigma_x=args_dict["sigma_x"], sigma_y=args_dict["sigma_y"])
    target_env = ToyGaussianEnv(y_weight_a0, y_weight_a1,
                                c_values = [1, -1], shift_probs=[1.0-shift_prop, shift_prop],
                                sigma_x=args_dict["sigma_x"], sigma_y=args_dict["sigma_y"])

    # Train and ground truth evaluation data
    np.random.seed(args_dict["run_id"])
    train_x, train_y, train_data_a, train_data_p, target_x, target_y = generate_data(train_env=train_env, target_env=target_env,\
                        train_size = args_dict["ntrain"], target_size=args_dict["ntest"],\
                        train_policy=train_policy, target_policy=target_policy)
    train_data = (train_x, train_y, train_data_a, train_data_p)
    valid_data = None
    test_data = (target_x, target_y, None)

    # Importance weights - assuming same environment in test as in train
    train_imp_weight = imp_weight(train_data[0], train_data[2], train_data[3], target_policy)

    # Ground truth value
    true_value = evaluate_sample_average(test_data[1])
    print("Shift prop.: {}, True Value: {}".format(shift_prop, true_value))

    if(args_dict["model_type"] == "baseline"):
        print("*** Baseline ERM ***")
        # baseline_model = train_erm(train_data, valid_data, l2_reg=args_dict["penalty"], lr=args_dict["dlr"], steps=args_dict["steps"])
        # test_acc, test_loss = evaluate(test_data, baseline_model)
        # baseline_params = list(baseline_model.parameters())
        # print("(Baseline ERM) Shifted Test Acc: "+str(test_acc)+"(Baseline ERM) Shifted Test Loss: "+str(test_loss))
        # return test_acc, test_loss, baseline_params
    elif(args_dict["model_type"] == "oracle_eval"):
        test_value = true_value
        test_loss = metrics(test_value, true_value)
        return test_value, test_loss, None
    elif(args_dict["model_type"] == "no_transfer_eval"):
        test_value = evaluate_sample_average(train_data[1])
        test_loss = metrics(test_value, true_value)
        print("No Transfer Eval Value: "+str(test_value)+"IS Eval Loss: "+str(test_loss))
        return test_value, test_loss, None
    elif(args_dict["model_type"] == "is_eval"):
        iw_train_y = torch.mul(train_imp_weight, train_data[1])
        test_value = evaluate_sample_average(iw_train_y)
        test_loss = metrics(test_value, true_value)
        print("IS Eval Value: "+str(test_value)+"IS Eval Loss: "+str(test_loss))
        return test_value, test_loss, None
    elif(args_dict["model_type"] == "wis_eval"):
        iw_train_y = torch.mul(train_imp_weight, train_data[1])
        test_value = evaluate_sample_average(iw_train_y) * train_data[1].shape[0] / train_imp_weight.sum().data.item()
        test_loss = metrics(test_value, true_value)
        print("WIS Eval Value: "+str(test_value)+"WIS Eval Loss: "+str(test_loss))
        return test_value, test_loss, None
    elif(args_dict["model_type"] == "dro_eval"):
        test_value, dro_model = train_dro_eval(train_imp_weight, train_data, valid_data, args_dict)
        # test_value, test_loss = evaluate(test_data, dro_model)
        test_loss = metrics(test_value, true_value)
        # dro_params =  list(dro_model.parameters())
        print("(DRO) Shifted Eval Value: "+str(test_value)+"(DRO) Shifted Eval Loss: "+str(test_loss))
        return test_value, test_loss, None
    #Causal
    # elif(args_dict["model_type"] == "causal"):
    #     print("*** Invariant Causal Predictor ***")
    #     if args_dict["nonsep"] is not None:
    #         train_data[0][:, args_dict["nonsep"]] = 1 # replace non-separating column by 1 equivalent to remove
    #         if valid_data is not None:
    #             valid_data[0][:, args_dict["nonsep"]] = 1
    #         test_data[0][:, args_dict["nonsep"]] = 1
    #     causal_model = train_erm(train_data, valid_data, l2_reg=args_dict["penalty"], lr=args_dict["dlr"], steps=args_dict["steps"])
    #     test_acc, test_loss = evaluate(test_data, causal_model)
    #     causal_params = list(causal_model.parameters())
    #     print("(Invariant Causal Predictor) Shifted Test Acc: "+str(test_acc)+"(Invariant Causal Predictor) Shifted Test Loss: "+str(test_loss))
    #     return test_acc, test_loss, causal_params
    # #DRO Models 
    # elif args_dict["turk"]:
    #     print("*** UV-DRO with Crowdsourced C|X,Y")
    # elif args_dict["oracle"]:
    #     print("*** UV-DRO with Oracle C|X,Y")
    # elif args_dict["joint"]:
    #     print("*** Baseline Joint DRO")
    # elif args_dict["subcov"]:
    #     print("*** Baseline Subcovariate-Shift DRO")
    # else:
    #     print("*** Baseline Covariate-Shift DRO C|X")
    # dro_model = train_dro(train_data, valid_data, args_dict)
    # test_acc, test_loss = evaluate(test_data, dro_model)
    # dro_params =  list(dro_model.parameters())
    # print("(DRO) Shifted Test Acc: "+str(test_acc)+"(DRO) Shifted Test Loss: "+str(test_loss))
    # return test_acc, test_loss, dro_params
    

