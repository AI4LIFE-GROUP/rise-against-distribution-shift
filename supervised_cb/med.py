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
print("device", device)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class LinearRegression(torch.nn.Module): 
    def __init__(self, input_size, num_classes): 
        super(LinearRegression, self).__init__() 
        torch.manual_seed(1)
        self.linear = nn.Linear(input_size, num_classes, bias=False) 
  
    def forward(self, x): 
        out = self.linear(x) 
        return out

class Metric_loss_reg_med(torch.nn.Module):
    """
    UV-DRO Objective from [INSERT SECTION] 
        - eta, alpha, L, B are all hyperparameters
        - joint flag allows loss function to revert to baseline joint DRO for comparison  
    """
    def __init__(self, args_dict, distances, b_init, kdual=2.0):
        super(Metric_loss_reg_med, self).__init__()
        self.eta = torch.FloatTensor([0.0]).to(device) #eta hyperparemeter (worst-case threshold) 
        self.alpha = args_dict["alpha"] #alpha hyperparemeter (worst-case group size) 
        self.L = args_dict["lip"] #Lipschitz constant  
        self.b_var = torch.nn.Parameter(torch.FloatTensor(b_init).to(device)) #B (loss transport matrix) 
        self.dists = Variable(torch.FloatTensor(distances), requires_grad=False).to(device) #Distances between training points 
        self.relu = torch.nn.ReLU()
        self.kdual = args_dict["kdual"]
        self.joint = 0 if args_dict["joint"] else 1 #Set joint DRO (shifts on (x,y)) flag 

    def pick_eta(self, residual):
        llist = torch.mean(self.relu(residual.reshape(-1, 1) - residual)**self.kdual, dim=0)**(1.0/self.kdual)
        res_loss = llist/self.alpha + residual 
        return residual[torch.argmin(res_loss)].item()

    def forward(self, log_probs, target):
        # losses = -(target*log_probs).sum(dim=1)
        losses = (target - log_probs)**2
        # losses = torch.abs(target - log_probs)

        bpos = self.b_var 
        transport = torch.sum((bpos - torch.t(bpos)), 1) * self.joint #Joint DRO (shifts on (x,y)) is equivalent to not incorporating smoothness
        residual = losses.squeeze() - transport 
        pick_eta = self.pick_eta(residual)
        mean = torch.mean(self.relu(residual - pick_eta)**self.kdual)**(1.0/self.kdual) 
        if self.joint==0:
            penalty = 0
        else:
            penalty = torch.sum((bpos*self.dists)*self.L)/losses.shape[0] * self.joint #Joint DRO (shifts on (x,y)) is equivalent to not incorporating smoothness
        return mean/self.alpha + penalty + pick_eta

    def project(self):
        self.b_var.data = self.relu(self.b_var).data

# def get_x_distances(train_data):
#     """
#     Returns distances on observed covariates x for covariate shift DRO. 
#     :param train_data: train data over which distances are calculated 
#     :return: square distance matrix of width/height equal to data length
#     """
#     new_turk_dists = np.zeros((train_data[0].shape[0], train_data[0].shape[0]))
#     for i in range(new_turk_dists.shape[0]):
#         for j in range(i, new_turk_dists.shape[0]):
#             xi = train_data[0][i]
#             xj = train_data[0][j]
#             x_dist =  np.linalg.norm(xi-xj)*1.0 #feature vector distance
#             new_turk_dists[i][j] = x_dist
#             new_turk_dists[j][i] = x_dist
#     return new_turk_dists
def get_x_distances(train_data):
    """
    Returns distances on observed covariates x for covariate shift DRO. 
    :param train_data: train data over which distances are calculated 
    :return: square distance matrix of width/height equal to data length
    """
    new_turk_dists = torch.zeros((train_data[0].shape[0], train_data[0].shape[0]))
    for i in range(new_turk_dists.shape[0]):
        for j in range(i, new_turk_dists.shape[0]):
            xi = train_data[0][i]
            xj = train_data[0][j]
            x_dist =  torch.norm(xi-xj)*1.0 #feature vector distance
            # TODO change torch.norm to torch.linalg.norm
            new_turk_dists[i][j] = x_dist
            new_turk_dists[j][i] = x_dist
    return new_turk_dists
# def get_x_distances(train_data):
#     """
#     Returns distances on observed covariates x for covariate shift DRO. 
#     :param train_data: train data over which distances are calculated 
#     :return: square distance matrix of width/height equal to data length
#     """
#     np.random.seed(0)
#     x = torch.FloatTensor(train_data[0].copy().astype(float))
#     x_distances = torch.zeros((x.shape[0], x.shape[0]))
#     for xi in range(0, x.shape[0]):
#         x_distances[xi] = torch.norm((x- x[xi]).type(torch.FloatTensor),p=2, dim=1)
#     return x_distances

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

def generate_data_mede(train_size=1000, target_size=200, sigma=.4, shift_probs=[.8, .2]):
    """
    Generates synthetic data for simulated medical diagnosis. 
    :param: train_size -  size of training data
    :param: target_size - size of target data
    :param: sigma - noise amount on noisy, reliable feature x2
    :param: shift_probs - probabilities controlling train / test overlap
    :return: train data features, train data labels, target data features, target data labels
    """
    train_data_x, train_data_y, train_data_c = [], [], []
    target_data_x, target_data_y = [], []
    for i in range(train_size): #Synthetic Train Data
        np.random.seed(i)
        c1 = np.random.normal()
        c2 = np.random.choice([1, -1],p=shift_probs)
        x1 = c1*c2  #Unreliable feature 
        x2 = c1+np.random.normal(0, sigma) #Reliable, but noisy, feature
        train_data_x.append((x1, x2))
        train_data_c.append(c2)
        train_data_y.append(c1)
    for i in range(target_size): #Synthetic Shifted Target Data
        np.random.seed(2*i+1)
        c1 = np.random.normal()
        c2 = np.random.choice([-1, 1],p=[0.8, 0.2]) #80% adolescents in target data
        x1 = c1*c2
        x2 = c1+np.random.normal()*sigma
        target_data_x.append((x1, x2))
        target_data_y.append(c1)
    return torch.FloatTensor(train_data_x), torch.FloatTensor(train_data_y), torch.LongTensor(train_data_c), torch.FloatTensor(target_data_x), torch.FloatTensor(target_data_y)

def generate_data_medc(train_size=1000, target_size=200, sigma=.4, train_shift_prop=0.8, shifts=[0.2,0.8], intercept=True):
    """
    Generates synthetic data for simulated medical diagnosis with both cause and effect features. 
    :param: train_size -  size of training data
    :param: target_size - size of target data
    :param: sigma - noise amount on noisy, reliable feature x2
    :param: shift_probs - probabilities controlling train / test overlap
    :return: train data features, train data labels, target data features, target data labels
    """
    train_data_x, train_data_y, train_data_c = [], [], []
    for i in range(train_size): #Synthetic Train Data
        np.random.seed(i)
        c1 = np.random.uniform(0,1)
        c2 = np.random.choice([-1, 1],p=[1.0-train_shift_prop, train_shift_prop])
        x1 = c1*c2
        x2 = np.random.uniform(-1,1)
        y = np.abs(x1) + (x1>=0)*np.random.normal(0, sigma)
        if intercept:
            train_data_x.append((x1, x2, 1))
        else:
            train_data_x.append((x1, x2))
        train_data_c.append(c2)
        train_data_y.append(y)
    trains = (torch.FloatTensor(train_data_x), torch.FloatTensor(train_data_y), torch.LongTensor(train_data_c))

    targets = []
    for test_shift_prop in shifts:
        target_data_x, target_data_y = [], []
        shift_probs=[test_shift_prop, 1.0-test_shift_prop] # Overlap decreases with increasing test_shift_prop
        for i in range(target_size): #Synthetic Shifted Target Data
            np.random.seed(2*i+1)
            c1 = np.random.uniform(0,1)
            c2 = np.random.choice([1, -1],p=shift_probs) #80% adolescents in target data
            x1 = c1*c2
            x2 = np.random.uniform(-1,1)
            y = np.abs(x1) + (x1>=0)*np.random.normal(0, sigma)
            if intercept:
                target_data_x.append((x1, x2, 1))
            else:
                target_data_x.append((x1, x2))
            target_data_y.append(y)
        targets.append((torch.FloatTensor(target_data_x), torch.FloatTensor(target_data_y), test_shift_prop))

    sep_features, nonsep_features = [0], [1]

    return trains, targets, sep_features, nonsep_features

def generate_data_medce(train_size=1000, target_size=200, means=[-1,1], means_2=[1,1],
                        sigma_x_c=[1,1], sigma_x_c_2=[1,1], sigma_y=[.1,1],
                        sigma_x_e=1, train_shift_prop=0.8, shifts=[0.2,0.8], intercept=True):
    """
    Generates synthetic data for simulated medical diagnosis with both cause and effect features. 
    :param: train_size -  size of training data
    :param: target_size - size of target data
    :param: sigma - noise amount on noisy, reliable feature x2
    :param: shift_probs - probabilities controlling train / test overlap
    :return: train data features, train data labels, target data features, target data labels
    """
    train_data_x, train_data_y, train_data_c = [], [], []
    for i in range(train_size): #Synthetic Train Data
        np.random.seed(i)
        group = np.random.choice([0,1], p=[1.0-train_shift_prop, train_shift_prop])
        loc_x_c = (1-group)*means[0] + group*means[1]
        std_x_c = (1-group)*sigma_x_c[0] + group*sigma_x_c[1]
        x_c = np.random.normal(loc_x_c, std_x_c)
        std_y = (x_c<=0)*sigma_y[0] + (x_c>0)*sigma_y[1]
        y = 1/8*x_c**2 + np.random.normal(0, std_y)
        x_e = 1/8*y**2 + np.random.normal(0, sigma_x_e)
        if intercept:
            train_data_x.append((x_c, x_e, 1))
        else:
            train_data_x.append((x_c, x_e))
        train_data_c.append(group)
        train_data_y.append(y)
    trains = (torch.FloatTensor(train_data_x), torch.FloatTensor(train_data_y), torch.LongTensor(train_data_c))

    targets = []
    for test_shift_prop in shifts:
        target_data_x, target_data_y = [], []
        shift_probs=[test_shift_prop, 1.0-test_shift_prop] # Overlap decreases with increasing test_shift_prop
        for i in range(target_size): #Synthetic Shifted Target Data
            np.random.seed(2*i+1)
            group = np.random.choice([0,1], p=shift_probs)
            loc_x_c = (1-group)*means[0] + group*means[1]
            std_x_c = (1-group)*sigma_x_c[0] + group*sigma_x_c[1]
            x_c = np.random.normal(loc_x_c, std_x_c)
            std_y = (x_c<=0)*sigma_y[0] + (x_c>0)*sigma_y[1]
            y = 1/8*x_c**2 + np.random.normal(0, std_y)
            x_e = 1/8*y**2 + np.random.normal(0, sigma_x_e)
            if intercept:
                target_data_x.append((x_c, x_e, 1))
            else:
                target_data_x.append((x_c, x_e))
            target_data_y.append(y)
        targets.append((torch.FloatTensor(target_data_x), torch.FloatTensor(target_data_y), test_shift_prop))

    sep_features, nonsep_features = [0], [1]

    return trains, targets, sep_features, nonsep_features

def generate_data(train_size=1000, target_size=200, sigma=.4, shift_probs=[.8, .2], intercept=True):
    """
    Generates synthetic data for simulated medical diagnosis with both cause and effect features. 
    :param: train_size -  size of training data
    :param: target_size - size of target data
    :param: sigma - noise amount on noisy, reliable feature x2
    :param: shift_probs - probabilities controlling train / test overlap
    :return: train data features, train data labels, target data features, target data labels
    """
    train_data_x, train_data_y, train_data_c = [], [], []
    target_data_x, target_data_y = [], []
    for i in range(train_size): #Synthetic Train Data
        np.random.seed(i)
        x2 = np.random.normal()
        y = x2+np.random.normal(0, sigma)
        c2 = np.random.choice([1, -1],p=shift_probs)
        x1 = y*c2  #Unreliable feature 
        # x2 = c1+np.random.normal(0, sigma) #Reliable, but noisy, feature
        if intercept:
            train_data_x.append((x1, x2, 1))
        else:
            train_data_x.append((x1, x2))
        train_data_c.append(c2)
        train_data_y.append(y)
    for i in range(target_size): #Synthetic Shifted Target Data
        np.random.seed(2*i+1)
        x2 = np.random.normal()
        y = x2+np.random.normal(0, sigma)
        c2 = np.random.choice([-1, 1],p=[0.8, 0.2]) #80% adolescents in target data
        x1 = y*c2
        # x2 = c1+np.random.normal()*sigma
        if intercept:
            target_data_x.append((x1, x2, 1))
        else:
            target_data_x.append((x1, x2))
        target_data_y.append(y)
    return torch.FloatTensor(train_data_x), torch.FloatTensor(train_data_y), torch.LongTensor(train_data_c), torch.FloatTensor(target_data_x), torch.FloatTensor(target_data_y)

def train_erm(train_data, valid_data, l2_reg=0.001, lr=0.01, steps=5000, log=True):
    # model = LogisticRegression(train_data[0].shape[1], 2)
    model = LinearRegression(train_data[0].shape[1], 1)
    # model = LinearRegression(2, 1)
    model = model.to(device)
    # loss_function = nn.NLLLoss()
    loss_function = nn.MSELoss()
    # loss_function = nn.L1Loss()

    reg_function = L2_reg(l2_reg)
    # labels = torch.squeeze(torch.LongTensor(train_data[1])).to(device)
    labels = torch.squeeze(train_data[1]).to(device)
    
    # optimizer = optim.Adagrad(list(model.parameters()), lr=lr)
    optimizer = optim.Adagrad(list(model.parameters())+list(loss_function.parameters()), lr=lr, initial_accumulator_value=0.0)
    # optimizer = optim.ASGD(list(model.parameters())+list(loss_function.parameters()), lr=lr)

    for step in range(steps):
        optimizer.zero_grad()
        log_probs = torch.squeeze(model(torch.FloatTensor(train_data[0]).to(device)))
        model_loss = loss_function(log_probs, labels)
        loss = model_loss+reg_function(model)
        loss.backward()
        optimizer.step()
        # if step%1000==0:
        #     print("loss", step, loss.item())
    return model

def train_dro(train_data, valid_data, sep_features, args_dict):
    # model = LogisticRegression(train_data[0].shape[1], 2)
    model = LinearRegression(train_data[0].shape[1], 1)
    # model = LinearRegression(2, 1)
    model = model.to(device)

    b_init = torch.zeros((train_data[0].shape[0], train_data[0].shape[0]))
    train_distances = []
    if(args_dict["turk"]): #UV-DRO with Crowdsousrcing
        train_distances = get_annotation_distances(train_data, shuffle_val=args_dict["shuffle"], dists_file=args_dict["anns_dists"])
    elif(args_dict["oracle"]): #Oracle UV-DRO
        train_distances = get_oracle_distances(train_data)
    elif(args_dict["subcov"]): #Subset DRO
        train_data_sub = deepcopy(train_data)
        train_data_sub = (train_data_sub[0][:, sep_features], train_data_sub[1], train_data_sub[2]) # keep non-sep covariate
        train_distances = get_x_distances(train_data_sub)
    elif(args_dict["joint"]): #Joint DRO
        train_distances = 0
    elif(args_dict["cov"]): #Covariate Shift DRO
        train_distances = get_x_distances(train_data)
    else:
        print('***specify DRO method***')
    loss_function = Metric_loss_reg_med(args_dict, train_distances, b_init)
    loss_function = loss_function.to(device)
    reg_function = L2_reg(args_dict["penalty"])
    reg_function = reg_function.to(device)
    # data_target = make_target(train_data[1], 2, long_tensor=False)
    # data_target = torch.LongTensor(train_data[1])
    data_target = train_data[1].to(device)

    # optimizer = optim.Adagrad(list(model.parameters())+list(loss_function.parameters()), lr=args_dict["dlr"], initial_accumulator_value=10.0)
    optimizer = optim.Adagrad(list(model.parameters())+list(loss_function.parameters()), lr=args_dict["dlr"], initial_accumulator_value=0.0)
    # optimizer = optim.ASGD(list(model.parameters())+list(loss_function.parameters()), lr=args_dict["dlr"])

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
        if step%500==0:
            print("loss", step, loss.item())
    return model
        
def evaluate(test_data, model):
    log_probs = torch.squeeze(model(torch.FloatTensor(test_data[0]).to(device)))
    # loss_function = nn.NLLLoss()
    loss_function = nn.MSELoss()
    # loss_function = nn.L1Loss()

    # labels = make_target(test_data[1], n=2, long_tensor=True)
    labels = torch.squeeze(torch.FloatTensor(test_data[1]).to(device))
    
    loss = loss_function(log_probs, labels)
    # acc = 0
    acc = -1*loss.data.item()

    # for i in range(len(test_data[0])):
    #     predict_y = torch.max(model(torch.FloatTensor([test_data[0][i]]).to(device)), 1)[1].item()
    #     if(predict_y == int(test_data[1][i])):
    #         acc +=1
    # return acc/len(test_data[0]), loss 
    metrics = {"acc": acc, "loss":loss}
    return metrics


def select_causal_features(data, nonsep_features):
    new_data  = (data[0].detach().clone(), data[1], data[2])
    new_data[0][:, nonsep_features] = 0 # replace non-separating column by 0 equivalent to remove
    # new_data[0][:, nonsep_features] = 1 # model with intercept is good for medc
    return new_data

def evaluate_multiple_test(targets, model):
    metrics_m = []
    for (target_x, target_y, test_shift_prop) in targets:
        test_data = (target_x, target_y, None) # None for test_c
        test_metrics = evaluate(test_data, model)
        test_metrics["test_shift_prop"] = test_shift_prop
        metrics_m.append(test_metrics)
    return metrics_m

def run_shift_experiment(args_dict):
    # written_df, observations_descrip, cities = read_data(filename="data/sf/2014_SQF_web.csv", num_lines=90000)
    # dx, dy, dl = generate_data(written_df, observations_descrip, cities)
    # train_data, valid_data, test_data = shift_data_loc(dx, dy, dl, cities, minority_prop=args_dict['min_p'], test_size=int(.2*args_dict["ntrain"]), train_size=args_dict["ntrain"])
    # shift_prop = args_dict['min_p']

    # train_x, train_y, train_data_c, target_x, target_y = generate_data_mede(train_size=args_dict["ntrain"], target_size=args_dict["ntest"], sigma=args_dict["sigma"], shift_probs=[1.0-shift_prop, shift_prop])
    # train_x, train_y, train_data_c, target_x, target_y = generate_data(train_size=args_dict["ntrain"], target_size=args_dict["ntest"], sigma=args_dict["sigma"], shift_probs=[1.0-shift_prop, shift_prop])
    # train_data, targets, sep_features, nonsep_features = generate_data_medc(train_size=args_dict["ntrain"],\
    #                                             target_size=args_dict["ntest"],sigma=args_dict["sigma"],\
    #                                             train_shift_prop=args_dict['trshift'],shifts=args_dict["shifts"],\
    #                                             intercept=args_dict['intercept'])
    train_data, targets, sep_features, nonsep_features = generate_data_medce(train_size=args_dict["ntrain"],\
                                                target_size=args_dict["ntest"],means=[-4,4],means_2=[4,4],\
                                                sigma_x_c=[args_dict["sigma"]*np.sqrt(2),args_dict["sigma"]*np.sqrt(2)],\
                                                sigma_x_c_2=[args_dict["sigma"]*np.sqrt(1),args_dict["sigma"]*np.sqrt(1)],\
                                                sigma_y=[args_dict["sigma"]*np.sqrt(0.1),args_dict["sigma"]*np.sqrt(2)],\
                                                sigma_x_e=args_dict["sigma"]*np.sqrt(8),\
                                                train_shift_prop=args_dict['trshift'],shifts=args_dict["shifts"],\
                                                intercept=args_dict['intercept'])
    valid_data = None

    if(args_dict["model_type"] == "baseline"):
        print("*** Baseline ERM ***")
        baseline_model = train_erm(train_data, valid_data, l2_reg=args_dict["penalty"], lr=args_dict["dlr"], steps=args_dict["steps"])
        # test_acc, test_loss = evaluate(test_data, baseline_model)
        test_metrics = evaluate_multiple_test(targets, baseline_model)
        baseline_params = list(baseline_model.parameters())
        print("(Baseline ERM) Shifted Test Metrics: {}".format(test_metrics))
        return test_metrics, baseline_params
    #Causal
    elif(args_dict["model_type"] == "causal"):
        print("*** Invariant Causal Predictor ***")
        if len(nonsep_features)>0:
            train_data = select_causal_features(train_data, nonsep_features)
            if valid_data is not None:
                valid_data = select_causal_features(valid_data, nonsep_features)
            targets = [select_causal_features(test_data, nonsep_features) for test_data in targets]
        causal_model = train_erm(train_data, valid_data, l2_reg=args_dict["penalty"], lr=args_dict["dlr"], steps=args_dict["steps"])
        # test_acc, test_loss = evaluate(test_data, causal_model)
        test_metrics = evaluate_multiple_test(targets, causal_model)
        causal_params = list(causal_model.parameters())
        print("(Invariant Causal Predictor) Shifted Test Metrics: {}".format(test_metrics))
        return test_metrics, causal_params
    #DRO Models 
    elif args_dict["turk"]:
        print("*** UV-DRO with Crowdsourced C|X,Y")
    elif args_dict["oracle"]:
        print("*** UV-DRO with Oracle C|X,Y")
    elif args_dict["joint"]:
        print("*** Baseline Joint DRO")
    elif args_dict["subcov"]:
        print("*** Baseline Subcovariate-Shift DRO")
    else:
        print("*** Baseline Covariate-Shift DRO C|X")
    dro_model = train_dro(train_data, valid_data, sep_features, args_dict)
    # test_acc, test_loss = evaluate(test_data, dro_model)
    test_metrics = evaluate_multiple_test(targets, dro_model)
    dro_params =  list(dro_model.parameters())
    print("(DRO) Shifted Test Metrics: {}".format(test_metrics))
    return test_metrics, dro_params
    
if __name__ == "__main__":
    args_dict = {"steps":10000, "ntrain":2000, "alpha":"0.2", "joint":False, "lip":0.1, "kdual":2.0, "penalty":0.0,
                "model_type":"baseline", "turk":False, "oracle":False, "shuffle":0.0, "subcov":False, "cov":False,
                "outdir":"saved_results/", "ntest":2000, "sigma":1, "filename":"medce", "dlr":.01, "trshift":0.8,
                "shifts":[0.1,0.8], "intercept":False}
    # shift_prop = args_dict['min_p']
    # train_data, targets, sep_features, nonsep_features = generate_data_medc(train_size=args_dict["ntrain"],\
    #                                             target_size=args_dict["ntest"],sigma=args_dict["sigma"],\
    #                                             train_shift_prop=args_dict['trshift'],shifts=args_dict["shifts"],\
    #                                             intercept=args_dict['intercept'])
    train_data, targets, sep_features, nonsep_features = generate_data_medce(train_size=args_dict["ntrain"],\
                                                target_size=args_dict["ntest"],sigma_x_c=np.sqrt(0.5),\
                                                sigma_y=np.sqrt(0.5), sigma_x_e=np.sqrt(2.0),\
                                                train_shift_prop=args_dict['trshift'],shifts=args_dict["shifts"],\
                                                intercept=args_dict['intercept'])
    valid_data = None

    print("*** Baseline ERM ***")
    baseline_model = train_erm(train_data, valid_data, l2_reg=args_dict["penalty"], lr=args_dict["dlr"], steps=args_dict["steps"])
    # train_acc, train_loss = evaluate(train_data, baseline_model)
    # test_acc, test_loss = evaluate(test_data, baseline_model)
    train_metrics = evaluate_multiple_test([(train_data[0],train_data[1],args_dict['trshift'])], baseline_model)
    test_metrics = evaluate_multiple_test(targets, baseline_model)
    baseline_params = list(baseline_model.parameters())
    print("(Baseline ERM) Shifted Test Metrics: {}".format(test_metrics))

    print(baseline_params)

    a=(train_data[0][:,0]>=0)
    print(evaluate((train_data[0][a,:],train_data[1][a],train_data[2][a]), baseline_model))
    print(evaluate((train_data[0][~a,:],train_data[1][~a],train_data[2][~a]), baseline_model))

    a=(targets[0][0][:,0]>=0)
    print(evaluate((targets[0][0][a,:],targets[0][1][a],None), baseline_model))
    print(evaluate((targets[0][0][~a,:],targets[0][1][~a],None), baseline_model))

    print("*** Invariant Causal Predictor ***")
    if len(nonsep_features)>0:
        train_data = select_causal_features(train_data, nonsep_features)
        if valid_data is not None:
            valid_data = select_causal_features(valid_data, nonsep_features)
        targets = [select_causal_features(test_data, nonsep_features) for test_data in targets]
    causal_model = train_erm(train_data, valid_data, l2_reg=args_dict["penalty"], lr=args_dict["dlr"], steps=args_dict["steps"])
    # test_acc, test_loss = evaluate(test_data, causal_model)
    test_metrics = evaluate_multiple_test(targets, causal_model)
    causal_params = list(causal_model.parameters())
    print("(Invariant Causal Predictor) Shifted Test Metrics: {}".format(test_metrics))

    print("done")