import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.autograd import Variable

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

sigmoid_fn = nn.Sigmoid()

class Metric_loss(torch.nn.Module):
    """
    UV-DRO Objective from [INSERT SECTION] 
        - eta, alpha, L, B are all hyperparameters
        - joint flag allows loss function to revert to baseline joint DRO for comparison  
    """
    def __init__(self, args_dict, distances, b_init, kdual=2.0, device=device):
        super(Metric_loss, self).__init__()
        self.eta = torch.FloatTensor([0.0]).to(device) #eta hyperparemeter (worst-case threshold) 
        self.alpha = args_dict["alpha"] #alpha hyperparemeter (worst-case group size) 
        self.L = args_dict["lip"] #Lipschitz constant  
        self.b_var = torch.nn.Parameter(torch.FloatTensor(b_init).to(device)) #B (loss transport matrix) 
        self.dists = Variable(torch.FloatTensor(distances), requires_grad=False).to(device) #Distances between training points 
        self.relu = torch.nn.ReLU()
        self.kdual = args_dict["kdual"] # kdual 
        self.joint = 0 if args_dict["joint"] else 1 #Set joint DRO (shifts on (x,y)) flag 

    def pick_eta(self, residual):
        llist = torch.mean(self.relu(residual.reshape(-1, 1) - residual)**self.kdual, dim=0)**(1.0/self.kdual)
        res_loss = llist/self.alpha + residual 
        return residual[torch.argmin(res_loss)].item()

    def forward(self, log_probs, target):
        losses = -(target*log_probs).sum(dim=1)
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

class LogisticRegression(torch.nn.Module): 
    def __init__(self, input_size, num_classes): 
        super(LogisticRegression, self).__init__() 
        torch.manual_seed(1)
        self.linear = nn.Linear(input_size, num_classes) 
  
    def forward(self, x): 
        out = self.linear(x) 
        return F.log_softmax(out, dim=1)

class L2_reg(torch.nn.Module):
    def __init__(self, penalty):
        super(L2_reg, self).__init__()
        self.penalty = penalty 

    def forward(self, model):
        weight1 = model.linear.weight
        l2_w1 = torch.norm(weight1, p=2)**2.0
        return l2_w1*self.penalty

# class LinearRegression(torch.nn.Module): 
#     def __init__(self, input_size, num_classes): 
#         super(LinearRegression, self).__init__() 
#         torch.manual_seed(1)
#         self.linear = nn.Linear(input_size, num_classes, bias=True) 
  
#     def forward(self, x): 
#         out = self.linear(x) 
#         return out

class Metric_loss_reg(torch.nn.Module):
    """
    DRO Objective for CB from [INSERT SECTION] 
        - eta, alpha, L, B are all hyperparameters
        - joint flag allows loss function to revert to baseline joint DRO for comparison  
    """
    def __init__(self, args_dict, distances, b_init, kdual=2.0, device=device):
        super(Metric_loss_reg, self).__init__()
        self.eta = torch.FloatTensor([0.0]).to(device) #eta hyperparemeter (worst-case threshold) 
        # self.eta = torch.nn.Parameter(torch.FloatTensor([0.0]).to(device)) #eta hyperparemeter (worst-case threshold) 
        self.alpha = args_dict["alpha"] #alpha hyperparemeter (worst-case group size) 
        self.L = args_dict["lip"] #Lipschitz constant  
        self.b_var = torch.nn.Parameter(torch.FloatTensor(b_init).to(device)) #B (loss transport matrix) 
        self.dists = Variable(torch.FloatTensor(distances), requires_grad=False).to(device) #Distances between training points 
        self.relu = torch.nn.ReLU()
        # self.kdual = kdual 
        self.kdual = args_dict["kdual"]
        self.joint = 0 if args_dict["joint"] else 1 #Set joint DRO (shifts on (x,y)) flag 

    def pick_eta(self, residual):
        llist = torch.mean(self.relu(residual.reshape(-1, 1) - residual)**self.kdual, dim=0)**(1.0/self.kdual)
        res_loss = llist/self.alpha + residual 
        return residual[torch.argmin(res_loss)].item()

    def forward(self, log_probs, target):
        # losses = -(target*log_probs).sum(dim=1)
        # losses = (target - log_probs)**2
        losses = log_probs

        bpos = self.b_var 
        transport = torch.sum((bpos - torch.t(bpos)), 1) * self.joint #Joint DRO (shifts on (x,y)) is equivalent to not incorporating smoothness
        residual = losses.squeeze() - transport 
        pick_eta = self.pick_eta(residual)
        # pick_eta = self.eta
        mean = torch.mean(self.relu(residual - pick_eta)**self.kdual)**(1.0/self.kdual) 
        if self.joint==0:
            penalty = 0
        else:
            penalty = torch.sum((bpos*self.dists)*self.L)/losses.shape[0] * self.joint #Joint DRO (shifts on (x,y)) is equivalent to not incorporating smoothness
        return mean/self.alpha + penalty + pick_eta

    def project(self):
        self.b_var.data = self.relu(self.b_var).data

# Policies
class LogisticRegressionPolicy(torch.nn.Module): 
    def __init__(self, input_size, num_classes=1, weight_init=None, bias_init=None): 
        super(LogisticRegressionPolicy, self).__init__() 
        torch.manual_seed(1)
        self.linear = nn.Linear(input_size, num_classes)
        if weight_init is not None:
            with torch.no_grad():
                self.linear.weight.copy_(weight_init)
        
        if bias_init is not None:
            with torch.no_grad():
                self.linear.bias.copy_(bias_init)
    
    def forward(self, x): 
        out = self.linear(x) 
        # return F.log_softmax(out, dim=1)
        return torch.bernoulli(sigmoid_fn(out))
        # proba = F.softmax(out, dim=1)
        # return torch.bernoulli(proba[:,1])

    def sample(self, x):
        out = self.linear(x)
        prob = sigmoid_fn(out)
        return torch.bernoulli(prob), prob # TODO return prob, 1-prob

class UniformPolicy(object):
    def __init__(self, actions, prob):
        self.actions = actions
        self.prob = prob
    
    def sample(self, context):
        idx = torch.multinomial(self.prob, num_samples=1)
        return self.actions[idx], self.prob[idx]

class LinearRegressionPolicy(torch.nn.Module): 
    def __init__(self, input_size, num_classes): 
        super(LinearRegressionPolicy, self).__init__()
        torch.manual_seed(1)
        self.linear = nn.Linear(input_size, num_classes, bias=True) 
  
    def forward(self, context): 
        out = self.linear(context)
        return out

    def sample(self, context):
        pred_reward = self.forward(context)
        max_idx = torch.argmax(pred_reward, dim=1)
        return max_idx, torch.ones_like(max_idx) # action a, probability of action a
        # TODO return probabilities for all actions

# class Policy():
#     def __init__(self, weight):
#         self.weight = weight
    
#     def proba(self, x):
#         m = nn.Sigmoid()
#         return m(torch.mv(x, self.weight))
    
#     def action(self, x):
#         return torch.bernoulli(self.proba(x))

# Environments
class ToyGaussianEnv(object):
    """
    :param: sigma - noise amount on noisy, reliable feature x2
    :param: shift_probs - probabilities controlling train / test overlap
    """
    def __init__(self, y_weight_a0, y_weight_a1, c_values, shift_probs, sigma_x, sigma_y):
        self.y_weight_a0 = y_weight_a0
        self.y_weight_a1 = y_weight_a1
        self.c_values = c_values
        self.shift_probs = shift_probs
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

    def observe(self):
        c1 = 10 + np.random.normal(0, self.sigma_x)
        c2 = np.random.choice(self.c_values,p=self.shift_probs)
        x1 = c1*c2  #Unreliable feature
        x2 = 5 + np.random.normal(0, self.sigma_x) #Reliable, but noisy, feature
        return np.array([x1, x2])
        # return np.random.normal([0.5,0.5], 1)

    def reward(self, context, action):
        if action==0:
            y = np.dot(context, self.y_weight_a0) + np.random.normal(0, self.sigma_y)
        else:
            y = np.dot(context, self.y_weight_a1) + np.random.normal(0, self.sigma_y)
        return y
        # return np.dot(context, np.array([0.05,0.9])) + np.random.normal()

class SimFromData(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.step = -1
    def reset(self):
        self.step = 0
        return self.x[self.step,:]
    def observe(self):
        self.step+=1
        if self.step==self.x.shape[0]:
            self.step=0 # loop again through x
        return self.x[self.step,:]
    def reward(self, context, action):
        if self.y[self.step]==action:
            return 0
        else:
            return -1