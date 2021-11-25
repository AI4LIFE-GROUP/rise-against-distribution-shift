import numpy as np
from numpy.core.defchararray import upper
from collections import namedtuple
from scipy.optimize import minimize_scalar

MyTransition = namedtuple('Transition',('state', 'action', 'next_state', 'reward', 'done'))

# This is a trajectory set, which is used for IS
class TrajectorySet(object):
    def __init__(self):
        self.trajectories = []
        self.position = -1
        # self.max_len = args.max_length

    def new_traj(self):
        traj = []
        self.trajectories.append(traj)
        self.position += 1

    def push(self, *args):
        self.trajectories[self.position].append(MyTransition(*args))

    def __len__(self):
        return len(self.trajectories)

def cvar_obj(eta, p, v, size):
    v_t = np.maximum(v - eta, 0)
    return 1/size * np.dot(p, v_t) + eta

def kl_obj(eta, p, v, size):
    eta += 1e-12
    logmeanexp = np.log(np.dot(p, np.exp(1/eta * v)))
    return size * eta + eta*logmeanexp

class RobustValue(object):
    def __init__(self, objective, size):
        obj_fn = None
        if objective=='cvar':
            obj_fn = cvar_obj
        elif objective=='kl':
            obj_fn = kl_obj
        self.objective = objective
        self.obj_fn = obj_fn
        self.size = size

    def solve(self, p, v):
        eta = 1e-6
        if self.objective=='kl':
            fn = lambda eta: self.obj_fn(eta, p, v, self.size)
            if v.max()==0.0:
                upper_bound = 1e3
            else:
                upper_bound = max(1e-12, v.max())
            res = minimize_scalar(fn, bounds=(1e-12, upper_bound), method='bounded',
                                    options = {'disp': 0, 'xatol': 1e-03})
        else:
            fn = lambda eta: self.obj_fn(eta, p, v, 1-self.size) # use 1-size such that increasing size increases robustness
            res = minimize_scalar(fn, options = {'disp': 0})
        eta = res.x
        value = self.obj_fn(eta, p, v, self.size)

        return value