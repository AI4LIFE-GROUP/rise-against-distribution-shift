'''
Solves robust Bellman equation for Cliffwalking domain

Usage:
python basic_rl_cliff.py -e cliff -ro -rt marginal -or kl -rs 0.8 -sh 0.1 -nr 10 &
python basic_rl_cliff.py -e cliff -ro -rt joint -or kl -rs 0.8 -sh 0.1 -nr 10 &
python basic_rl_cliff.py -e cliff -a td -rt joint -sh 0.1 -nr 10 &
python basic_rl_cliff.py -e cliff -a td -rt joint -sh 0.8 -nr 10 &

python basic_rl_cliff.py -e cliff -a td -ro -rt marginal -or cvar -rs 0.8 -sh 0.1 -nr 10 &
python basic_rl_cliff.py -e cliff -a td -ro -rt joint -or cvar -rs 0.8 -sh 0.1 -nr 10 &

python basic_rl_cliff.py -e cliff -a q_learning -rt joint -sh 0.1 -nr 10 &
python basic_rl_cliff.py -e cliff -a q_learning -rt joint -sh 0.8 -nr 10 &

python basic_rl_cliff.py -e cliff -a dp -ro -rt marginal -or cvar -rs 0.8 -sh 0.1 -n 100 -nr 1 &
python basic_rl_cliff.py -e cliff -a dp -ro -rt joint -or cvar -rs 0.8 -sh 0.1 -n 100 -nr 1 &

python basic_rl_cliff.py -e cliff -a dp -rt joint -sh 0.1 -n 100
python basic_rl_cliff.py -e cliff -a dp -rt marginal -sh 0.1

'''

import argparse
from joblib import Parallel, delayed

from env_cliffwalking import CliffWalkingEnv
from utils import RobustValue, TrajectorySet

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")


parser = argparse.ArgumentParser(description='Use SARSA/Q-learning algorithm with epsilon-greedy/softmax polciy.')
parser.add_argument('-e', '--environment', default='Roulette-v0',
                    help="Name of the environment provided in the OpenAI Gym. (Default: Roulette-v0)")
parser.add_argument('-ga', '--gamma', default='0.99', type=float,
                    help="Discount rate. (Default: 0.99)")
parser.add_argument('-ro', '--robust', action='store_true', default=False,
                    help='Robust updates used for the algorithm. (Default: False)')
parser.add_argument('-rt', '--rtype', default='joint', choices=['marginal', 'joint'],
                    help="Type of robust evaluation. (Default: joint)")
parser.add_argument('-or', '--robj', default='kl', choices=['kl', 'cvar'],
                    help="Type of robust sets. (Default: kl)")
parser.add_argument('-rs', '--rsize', default='0.1', type=float,
                    help="Size of robust set. (Default: 0.1)")
parser.add_argument('-sh', '--shift', default='0.1', type=float,
                    help="Shift in test environment. (Default: 0.1)")

parser.add_argument('-ms', '--maxstep', default='100', type=int,
                    help="Maximum step allowed in a episode. (Default: 100)")
parser.add_argument('-al', '--alpha', default='0.1', type=float,
                    help="Learning rate. (Default: 0.1)")
parser.add_argument('-a', '--algorithm', default='dp', choices=['dp', 'td', 'q_learning', 'sarsa'],
                    help="Type of learning algorithm. (Default: td)")
parser.add_argument('-n', '--nepisode', default='10000', type=int,
                    help="Number of episode. (Default: 10000)")
parser.add_argument('-nr', '--nrun', default='10', type=int,
                    help="Number of repetitions. (Default: 10)")

parser.add_argument('-o', '--outdir', default='results', type=str,
                    help="Output directory for pickle and plots. (Default: results)")

parser.add_argument('-p', '--policy', default='epsilon_greedy', choices=['epsilon_greedy', 'softmax'],
                    help="Type of policy. (Default: epsilon_greedy)")
parser.add_argument('-be', '--beta', default='0.0', type=float,
                    help="Initial value of an inverse temperature. (Default: 0.0)")
parser.add_argument('-bi', '--betainc', default='0.01', type=float,
                    help="Linear increase rate of an inverse temperature. (Default: 0.01)")
parser.add_argument('-ep', '--epsilon', default='0.8', type=float,
                    help="Fraction of random exploration in the epsilon greedy. (Default: 0.8)")
parser.add_argument('-ed', '--epsilondecay', default='0.995', type=float,
                    help="Decay rate of epsilon in the epsilon greedy. (Default: 0.995)")
parser.add_argument('-ka', '--kappa', default='0.1', type=float,
                    help="Weight of the most recent cumulative reward for computing its running average. (Default: 0.01)")
parser.add_argument('-qm', '--qmean', default='0.0', type=float,
                    help="Mean of the Gaussian used for initializing Q table. (Default: 0.0)")
parser.add_argument('-qs', '--qstd', default='1.0', type=float,
                    help="Standard deviation of the Gaussian used for initializing Q table. (Default: 1.0)")
parser.add_argument('-pr', '--prob', default=0.2, type=float,
                    help="Probability of perturbation")
# args = parser.parse_args("-e cliff -ro -rt marginal -or kl -rs 0.1".split())
# args = parser.parse_args("-e cliff -rt joint -n 10".split())
args = parser.parse_args()

import gym
import numpy as np
import os

import matplotlib.pyplot as plt

def softmax(q_value, beta=1.0):
    assert beta >= 0.0
    q_tilde = q_value - np.max(q_value)
    factors = np.exp(beta * q_tilde)
    return factors / np.sum(factors)


def select_a_with_softmax(curr_s, q_value, beta=1.0):
    prob_a = softmax(q_value[curr_s, :], beta=beta)
    cumsum_a = np.cumsum(prob_a)
    return np.where(np.random.rand() < cumsum_a)[0][0]


def select_a_with_epsilon_greedy(curr_s, q_value, epsilon=0.1):
    a = np.argmax(q_value[curr_s, :])
    if np.random.rand() < epsilon:
        a = np.random.randint(q_value.shape[1])
    return a


def select_a_greedy(curr_s, q_value):
    return np.argmax(q_value[curr_s, :])


def get_value(q_value):
    # Get shape
    n_s, n_a = q_value.shape
    value = np.amax(q_value, axis=1)
    return value


def get_sigma(value, p):
    # Ball of radius r
    sigma = - p * np.linalg.norm(value)
    # sigma = - p
    return sigma


def get_robust_value(next_state_prob, v_value, size, solver):
    # value = np.dot(next_state_prob, v_value) - size * np.linalg.norm(v_value)
    value = solver.solve(next_state_prob, -1*v_value)
    return -1*value


def dp_evaluation_true(robust=False, robust_type='joint', env=None, policy=None, delta_threshold=1e-5, objective='cvar', size=1e-5):
    # env_type = args.environment
    algorithm_type = 'dp_evaluation'
    policy_type = 'random'

    # Selection of the problem
    # env = gym.envs.make(env_type)
    # env = NChainEnv(slip=0.2)
    # env = CliffWalkingEnv()

    # Constraints imposed by the environment
    # n_a = env.action_space.n
    # n_s = env.observation_space.n
    n_a = env.nA
    n_s = env.nS
    n_s_2 = env.nS_2

    print("Number of states = {}".format(n_s))
    # Meta parameters for the RL agent
    gamma = args.gamma

    # Experimental setup
    max_step = args.maxstep

    # Initialization of a Q-value table
    q_value = np.zeros([n_s, n_s_2, n_a])
    # v_value = np.zeros([n_s_2*n_s])
    v_value = np.zeros([n_s, n_s_2])

    # If qtable is not none set q_value to it
    # if qtable is not None:
    #     q_value = qtable

    # Initialization of a list for storing simulation history
    history = []

    # env.reset()

    np.set_printoptions(precision=3, suppress=True)

    # result_dir = 'results-{0}-{1}-{2}'.format(env_type, algorithm_type, policy_type)

    # Start monitoring the simulation for OpenAI Gym
    # env = gym.wrappers.Monitor(env, result_dir, force=True)
    # threshold = 1 - p_env

    robust_solver = RobustValue(objective, size)

    # Start with a random (all 0) value function
    # v_value = np.zeros(n_s)
    niter = 0
    while True:
        delta = 0
        # For each state, perform a "full backup"
        for s in range(n_s):
            for s_2 in range(n_s_2):
                v = 0
                # Look at the possible next actions
                for a, action_prob in enumerate(policy[s]): # action only depends on s
                    # For each action, look at the possible next states...
                    next_state_prob, next_state_reward, next_states = env.P[s][a]
                    next_state_prob_2, next_state_reward_2, next_states_2 = env.P_2[s_2][a]

                    if robust_type=='joint': 
                        _next_state_prob = np.outer(next_state_prob,next_state_prob_2) # rows s_1, columns s_2
                        _next_state_reward = np.add.outer(next_state_reward,next_state_reward_2) # nS * nS_2
                        _next_v_value = (_next_state_reward + gamma * v_value)
                        _next_v_value = _next_v_value.reshape(-1)
                        _next_state_prob = _next_state_prob.reshape(-1)
                    else: # marginal
                        _next_state_prob = next_state_prob # row s_1
                        _next_state_reward = np.add.outer(next_state_reward,next_state_reward_2)
                        _next_v_value = (_next_state_reward + gamma * v_value)
                        _next_v_value = np.dot(_next_v_value, next_state_prob_2) # nS * 1

                    expected_value = 0
                    if robust:
                        expected_value = get_robust_value(_next_state_prob, _next_v_value, size, robust_solver)
                    else:
                        expected_value = np.dot(_next_v_value, _next_state_prob)

                    v += action_prob * expected_value
                    # for  certainty, next_state, reward, done in next_states:
                    #     prob = next_state_prob[next_state]
                    #     # Calculate the expected value. Ref: Sutton book eq. 4.6.
                    #     v += action_prob * prob * (reward + gamma * v_value[next_state]) 
                # How much our value function changed (across any states)
                # s_12 = s + n_s*s_2
                # delta = max(delta, np.abs(v - v_value[s_12]))
                # v_value[s_12] = v
                delta = max(delta, np.abs(v - v_value[s,s_2]))
                v_value[s,s_2] = v
        niter+=1
        if niter%10==0:
            print(v_value)
        # Stop evaluating once our value function change is below a threshold
        if delta < delta_threshold:
            print('required iters', niter)
            break
    return np.array(v_value)


def dp_evaluation(robust=False, robust_type='joint', env=None, policy=None, n_episode=10000, delta_threshold=1e-5, objective='cvar', size=1e-5):
    # env_type = args.environment
    algorithm_type = 'dp_evaluation'
    policy_type = 'random'

    # Selection of the problem
    # env = gym.envs.make(env_type)
    # env = NChainEnv(slip=0.2)
    # env = CliffWalkingEnv()

    # Constraints imposed by the environment
    # n_a = env.action_space.n
    # n_s = env.observation_space.n
    n_a = env.nA
    n_s = env.nS
    n_s_2 = env.nS_2

    print("Number of states = {}".format(n_s))
    # Meta parameters for the RL agent
    gamma = args.gamma

    # Experimental setup
    max_step = args.maxstep

    # Initialization of a Q-value table
    q_value = np.zeros([n_s, n_s_2, n_a])
    # v_value = np.zeros([n_s_2*n_s])
    v_value = np.zeros([n_s, n_s_2])

    # If qtable is not none set q_value to it
    # if qtable is not None:
    #     q_value = qtable

    # Initialization of a list for storing simulation history
    history = []

    # env.reset()

    # sample trajectories and learn transition model
    traj_set = sample_trajectories(env, policy, n_episode)

    P, P_2 = learn_transition_model(env, traj_set)

    np.set_printoptions(precision=3, suppress=True)

    # result_dir = 'results-{0}-{1}-{2}'.format(env_type, algorithm_type, policy_type)

    # Start monitoring the simulation for OpenAI Gym
    # env = gym.wrappers.Monitor(env, result_dir, force=True)
    # threshold = 1 - p_env

    robust_solver = RobustValue(objective, size)

    # Start with a random (all 0) value function
    # v_value = np.zeros(n_s)
    niter = 0
    while True:
        delta = 0
        # For each state, perform a "full backup"
        for s in range(n_s):
            for s_2 in range(n_s_2):
                v = 0
                # Look at the possible next actions
                for a, action_prob in enumerate(policy[s]): # action only depends on s
                    # For each action, look at the possible next states...
                    _, next_state_reward, _ = env.P[s][a]
                    _, next_state_reward_2, _ = env.P_2[s_2][a]
                    _next_state_reward = np.add.outer(next_state_reward,next_state_reward_2) # nS * nS_2
                    _next_v_value = (_next_state_reward + gamma * v_value).copy()

                    next_state_prob = P[s][a]
                    next_state_prob_2 = P_2[s_2][a]
                    # if P[s][a].sum()==0: # no samples for state action pair in data, not req if next state prob sum to 1 by adding small noise 1e-6 and normalizing
                    #     continue

                    expected_value = 0
                    if robust:
                        if robust_type=='joint': 
                            _next_state_prob = np.outer(next_state_prob,next_state_prob_2) # rows s_1, columns s_2
                            _next_v_value = _next_v_value.reshape(-1)
                            _next_state_prob = _next_state_prob.reshape(-1)
                        else: # marginal
                            _next_state_prob = next_state_prob # row s_1
                            _next_v_value = np.dot(_next_v_value, next_state_prob_2) # nS * 1
                        expected_value = get_robust_value(_next_state_prob, _next_v_value, size, robust_solver)
                    else:
                        _next_state_prob = np.outer(next_state_prob,next_state_prob_2) # rows s_1, columns s_2
                        _next_state_prob = _next_state_prob.reshape(-1)
                        _next_v_value = _next_v_value.reshape(-1)
                        expected_value = np.dot(_next_v_value, _next_state_prob)
                    v += action_prob * expected_value
                    # for  certainty, next_state, reward, done in next_states:
                    #     prob = next_state_prob[next_state]
                    #     # Calculate the expected value. Ref: Sutton book eq. 4.6.
                    #     v += action_prob * prob * (reward + gamma * v_value[next_state]) 
                # How much our value function changed (across any states)
                # s_12 = s + n_s*s_2
                # delta = max(delta, np.abs(v - v_value[s_12]))
                # v_value[s_12] = v
                delta = max(delta, np.abs(v - v_value[s,s_2]))
                v_value[s,s_2] = v
        niter+=1
        if niter%10==0:
            print(v_value)
        # Stop evaluating once our value function change is below a threshold
        if delta < delta_threshold:
            print('required iters', niter)
            break
    return np.array(v_value)


def dp_evaluation_marginal(robust=False, env=None, policy=None, delta_threshold=1e-5, objective='cvar', size=1e-5):
    # env_type = args.environment
    algorithm_type = 'dp_evaluation_marginal'
    policy_type = 'random'

    # Selection of the problem
    # env = gym.envs.make(env_type)
    # env = NChainEnv(slip=0.2)
    # env = CliffWalkingEnv()

    # Constraints imposed by the environment
    # n_a = env.action_space.n
    # n_s = env.observation_space.n
    n_a = env.nA
    n_s = env.nS
    n_s_2 = env.nS_2

    print("Number of states = {}".format(n_s))
    # Meta parameters for the RL agent
    gamma = args.gamma

    # Experimental setup
    max_step = args.maxstep

    # Initialization of a Q-value table
    q_value = np.zeros([n_s, n_a])
    v_value = np.zeros([n_s])

    # If qtable is not none set q_value to it
    # if qtable is not None:
    #     q_value = qtable

    # Initialization of a list for storing simulation history
    history = []

    # env.reset()

    np.set_printoptions(precision=3, suppress=True)

    # result_dir = 'results-{0}-{1}-{2}'.format(env_type, algorithm_type, policy_type)

    # Start monitoring the simulation for OpenAI Gym
    # env = gym.wrappers.Monitor(env, result_dir, force=True)
    # threshold = 1 - p_env

    robust_solver = RobustValue(objective, size)

    # Start with a random (all 0) value function
    # v_value = np.zeros(n_s)
    niter = 0
    while True:
        delta = 0
        # For each state, perform a "full backup"
        for s in range(n_s):
            v = 0
            # Look at the possible next actions
            for a, action_prob in enumerate(policy[s]): # action only depends on s
                # For each action, look at the possible next states...
                _next_state_prob, _next_state_reward, next_states = env.P[s][a]
                _next_v_value = (_next_state_reward + gamma * v_value)
                expected_value = 0
                if robust:
                    expected_value = get_robust_value(_next_state_prob, _next_v_value, size, robust_solver)
                else:
                    expected_value = np.dot(_next_state_prob, _next_v_value)
                v += action_prob * expected_value
                # for  certainty, next_state, reward, done in next_states:
                #     prob = next_state_prob[next_state]
                #     # Calculate the expected value. Ref: Sutton book eq. 4.6.
                #     v += action_prob * prob * (reward + gamma * v_value[next_state]) 
            # How much our value function changed (across any states)
            delta = max(delta, np.abs(v - v_value[s]))
            v_value[s] = v
        niter+=1
        if niter%100==0:
            print(v_value)
        # Stop evaluating once our value function change is below a threshold
        if delta < delta_threshold:
            print('required iters', niter)
            break
    return np.array(v_value)


def sample_trajectories(env=None, policy=None, n_episode=10000):
    traj_set = TrajectorySet()

    env.reset()

    # Experimental setup
    max_step = args.maxstep

    for i_episode in range(n_episode):
        traj_set.new_traj()
        # Reset a cumulative reward for this episode
        # cumu_r = 0

        # Start a new episode and sample the initial state
        curr_s = env.reset()
        curr_s_1, curr_s_2 = curr_s

        action_prob = policy[curr_s_1].T.squeeze()
        curr_a = np.random.choice(np.arange(len(action_prob)), p=action_prob)

        for i_step in range(max_step):

            # Get a result of your action from the environment
            next_s, r, done, info = env.step(curr_a)
            next_s_1, next_s_2 = next_s

            traj_set.push(curr_s, curr_a, next_s, r, done)

            # Select an action
            action_prob = policy[next_s_1].T.squeeze()
            next_a = np.random.choice(np.arange(len(action_prob)), p=action_prob)
        
            curr_s = next_s
            curr_a = next_a

            if done:
                break

    return traj_set


def learn_transition_model(env, traj_set):
    n_s = env.nS
    n_s_2 = env.nS_2
    n_a = env.nA
    P = {}
    for s in range(n_s):
        P[s] = { a : np.zeros(n_s)+1e-6 for a in range(n_a) }
    P_2 = {}
    for s in range(n_s_2):
        P_2[s] = { a : np.zeros(n_s_2)+1e-6 for a in range(n_a) }

    for i_episode in range(len(traj_set)):
        # Reset a cumulative reward for this episode
        # cumu_r = 0

        for i_step in range(len(traj_set.trajectories[i_episode])):

            # Get a result of your action from the environment
            traj = traj_set.trajectories[i_episode][i_step]
            curr_s, curr_a, next_s, r, done = traj.state, traj.action, traj.next_state, traj.reward, traj.done
            curr_s_1, curr_s_2 = curr_s
            next_s_1, next_s_2 = next_s
            P[curr_s_1][curr_a][next_s_1] += 1
            P_2[curr_s_2][curr_a][next_s_2] += 1
    
    for s in range(n_s):
        for a in range(n_a):
            P[s][a] = P[s][a]/sum(P[s][a])
    for s in range(n_s_2):
        for a in range(n_a):
            P_2[s][a] = P_2[s][a]/sum(P_2[s][a])

    return P, P_2

def td_evaluation(robust=False, robust_type='joint', env=None, policy=None, n_episode=10000, objective='cvar', size=1e-5):
    # env_type = args.environment
    algorithm_type = 'td_evaluation'
    policy_type = 'random'

    # Selection of the problem
    # env = gym.envs.make(env_type)
    # env = NChainEnv(slip=0.2)
    # env = CliffWalkingEnv()

    # Constraints imposed by the environment
    # n_a = env.action_space.n
    # n_s = env.observation_space.n
    n_a = env.nA
    n_s = env.nS
    n_s_2 = env.nS_2

    print("Number of states = {}".format(n_s))
    # Meta parameters for the RL agent
    alpha = args.alpha
    gamma = args.gamma

    # Experimental setup
    max_step = args.maxstep

    # Initialization of a Q-value table
    v_value = np.zeros([n_s, n_s_2])
    # q_value = np.zeros([n_s, n_s_2, n_a])

    # If qtable is not none set q_value to it
    # if qtable is not None:
    #     q_value = qtable

    # env.reset()

    traj_set = sample_trajectories(env, policy, n_episode)

    P, P_2 = learn_transition_model(env, traj_set)

    np.set_printoptions(precision=3, suppress=True)

    # result_dir = 'results-{0}-{1}-{2}'.format(env_type, algorithm_type, policy_type)

    # Start monitoring the simulation for OpenAI Gym
    # env = gym.wrappers.Monitor(env, result_dir, force=True)
    # threshold = 1 - p_env

    robust_solver = RobustValue(objective, size)

    for i_episode in range(len(traj_set)):
        # Reset a cumulative reward for this episode
        # cumu_r = 0

        for i_step in range(len(traj_set.trajectories[i_episode])):

            # Get a result of your action from the environment
            traj = traj_set.trajectories[i_episode][i_step]
            curr_s, curr_a, next_s, r, done = traj.state, traj.action, traj.next_state, traj.reward, traj.done
            curr_s_1, curr_s_2 = curr_s
            next_s_1, next_s_2 = next_s

            # Update a cummulative reward
            # cumu_r = r + gamma * cumu_r

            if robust:
                
                next_state_prob = P[curr_s_1][curr_a]
                next_state_prob_2 = P_2[curr_s_2][curr_a]

                if robust_type=='joint':
                    _next_state_prob = np.outer(next_state_prob,next_state_prob_2) # rows s_1, columns s_2
                    _next_v_value = v_value.copy()
                    # _next_v_value = q_value[:, :, curr_a]
                    _next_v_value = _next_v_value.reshape(-1)
                    _next_state_prob = _next_state_prob.reshape(-1)
                else: # marginal
                    _next_state_prob = next_state_prob # row s_1
                    _next_v_value = v_value.copy()
                    # _next_v_value = q_value[:, :, curr_a]
                    _next_v_value = np.dot(_next_v_value, next_state_prob_2) # nS * 1
                    
                robust_value = get_robust_value(_next_state_prob, _next_v_value, size, robust_solver)
                
                delta = r + gamma * robust_value - v_value[curr_s_1, curr_s_2]
                # delta = r + gamma * robust_value - q_value[curr_s_1, curr_s_2, curr_a]

            else:
                delta = r + gamma * v_value[next_s_1, next_s_2] - v_value[curr_s_1, curr_s_2]
                # delta = r + gamma * q_value[next_s_1, next_s_2, curr_a] - q_value[curr_s_1, curr_s_2, curr_a]
            
            # Update a Q value table
            v_value[curr_s_1, curr_s_2] += alpha * delta
            # q_value[curr_s_1, curr_s_2, curr_a] += alpha * delta

            # print("adsffds", curr_s_1, r, delta)
            if done:
                # print("12adfaesf", curr_s_1, r, delta)
                break

        if i_episode%1000==0:
            print("Episode: {}, v_value: {}".format(i_episode, v_value))
    
    # print("Q_value = {0}".format(q_value))

    # v_value = np.zeros([n_s, n_s_2])
    # for s in range(n_s):
    #     v_value[s, :] = np.dot(q_value[s, :, :], policy[s].T)

    return v_value

def td_learning(robust=False, robust_type='joint', env=None, policy_train=None, n_episode=10000, objective='cvar', size=1e-5):
    # env_type = args.environment
    algorithm_type = 'td_learning'
    policy_type = 'deterministic'

    # Selection of the problem
    # env = gym.envs.make(env_type)
    # env = NChainEnv(slip=0.2)
    # env = CliffWalkingEnv()

    # Constraints imposed by the environment
    # n_a = env.action_space.n
    # n_s = env.observation_space.n
    n_a = env.nA
    n_s = env.nS
    n_s_2 = env.nS_2

    print("Number of states = {}".format(n_s))
    # Meta parameters for the RL agent
    alpha = args.alpha
    gamma = args.gamma

    # Experimental setup
    max_step = args.maxstep

    # Initialization of a Q-value table
    # v_value = np.zeros([n_s, n_s_2])
    q_value = np.zeros([n_s, n_s_2, n_a])

    # If qtable is not none set q_value to it
    # if qtable is not None:
    #     q_value = qtable
    
    print("algorithm_type: {}".format(algorithm_type))
    print("policy_type: {}".format(policy_type))

    # env.reset()

    traj_set = sample_trajectories(env, policy_train, n_episode)

    print("sampled trajectories")

    P, P_2 = learn_transition_model(env, traj_set)

    print("built transition model")

    np.set_printoptions(precision=3, suppress=True)

    # result_dir = 'results-{0}-{1}-{2}'.format(env_type, algorithm_type, policy_type)

    # Start monitoring the simulation for OpenAI Gym
    # env = gym.wrappers.Monitor(env, result_dir, force=True)
    # threshold = 1 - p_env

    robust_solver = RobustValue(objective, size)

    for i_episode in range(len(traj_set)):
        # Reset a cumulative reward for this episode
        # cumu_r = 0

        for i_step in range(len(traj_set.trajectories[i_episode])):

            # Get a result of your action from the environment
            traj = traj_set.trajectories[i_episode][i_step]
            curr_s, curr_a, next_s, r, done = traj.state, traj.action, traj.next_state, traj.reward, traj.done
            curr_s_1, curr_s_2 = curr_s
            next_s_1, next_s_2 = next_s

            # Update a cummulative reward
            # cumu_r = r + gamma * cumu_r

            if robust:
                
                next_state_prob = P[curr_s_1][curr_a]
                next_state_prob_2 = P_2[curr_s_2][curr_a]

                robust_values = np.zeros(n_a)

                for a in range(n_a):
                    if robust_type=='joint':
                        _next_state_prob = np.outer(next_state_prob,next_state_prob_2) # rows s_1, columns s_2
                        # _next_v_value = v_value.copy()
                        _next_v_value = q_value[:, :, a].copy()
                        _next_v_value = _next_v_value.reshape(-1)
                        _next_state_prob = _next_state_prob.reshape(-1)
                    else: # marginal
                        _next_state_prob = next_state_prob # row s_1
                        # _next_v_value = v_value.copy()
                        _next_v_value = q_value[:, :, a].copy()
                        _next_v_value = np.dot(_next_v_value, next_state_prob_2) # nS * 1
                        
                    robust_values[a] = get_robust_value(_next_state_prob, _next_v_value, size, robust_solver)
                
                # delta = r + gamma * robust_value - v_value[curr_s_1, curr_s_2]
                delta = r + gamma * np.max(robust_values) - q_value[curr_s_1, curr_s_2, curr_a]

            else:
                # delta = r + gamma * v_value[next_s_1, next_s_2] - v_value[curr_s_1, curr_s_2]
                delta = r + gamma * np.max(q_value[next_s_1, next_s_2, :]) - q_value[curr_s_1, curr_s_2, curr_a]
            
            # Update a Q value table
            # v_value[curr_s_1, curr_s_2] += alpha * delta
            q_value[curr_s_1, curr_s_2, curr_a] += alpha * delta

            # print("adsffds", curr_s_1, r, delta)
            if done:
                # print("12adfaesf", curr_s_1, r, delta)
                break

        if i_episode%1000==0:
            print("Episode: {}, q_value: {}".format(i_episode, q_value))
    
    # print("Q_value = {0}".format(q_value))

    v_value = np.max(q_value, axis=2)

    return q_value, v_value

def q_learning(robust, p_env, p_est, epsilon=0, qtable=None, update=True, n_episode=10000):
    env_type = args.environment
    algorithm_type = args.algorithm
    policy_type = args.policy

    # Random seed
    np.random.RandomState(42)

    # Selection of the problem
    env = gym.envs.make(env_type)

    # Constraints imposed by the environment
    n_a = env.action_space.n
    n_s = env.observation_space.n

    print("Number of states = {}".format(n_s))
    # Meta parameters for the RL agent
    alpha = args.alpha
    beta = args.beta
    beta_inc = args.betainc
    gamma = args.gamma
    epsilon_decay = args.epsilondecay
    q_mean = args.qmean
    q_std = args.qstd

    # Experimental setup
    max_step = args.maxstep

    # Running average of the cumulative reward, which is used for controlling an exploration rate
    # (This idea of controlling exploration rate by the terminal reward is suggested by JKCooper2)
    # See https://gym.openai.com/evaluations/eval_xSOlwrBsQDqUW7y6lJOevQ
    kappa = args.kappa
    ave_cumu_r = None

    # Initialization of a Q-value table
    q_value = np.zeros([n_s, n_a])

    # If qtable is not none set q_value to it
    if qtable is not None:
        q_value = qtable

    # Initialization of a list for storing simulation history
    history = []

    print("algorithm_type: {}".format(algorithm_type))
    print("policy_type: {}".format(policy_type))

    env.reset()

    np.set_printoptions(precision=3, suppress=True)

    result_dir = 'results-{0}-{1}-{2}'.format(env_type, algorithm_type, policy_type)

    # Start monitoring the simulation for OpenAI Gym
    env = gym.wrappers.Monitor(env, result_dir, force=True)
    threshold = 1 - p_env

    for i_episode in range(n_episode):
        # Reset a cumulative reward for this episode
        cumu_r = 0

        # Start a new episode and sample the initial state
        curr_s = env.reset()

        # Print q_table
        # print "qtable = {}".format(q_value)

        # Select the first action in this episode
        if policy_type == 'softmax':
            curr_a = select_a_with_softmax(curr_s, q_value, beta=beta)
        elif policy_type == 'epsilon_greedy':
            curr_a = select_a_with_epsilon_greedy(curr_s, q_value, epsilon=epsilon)
        else:
            raise ValueError("Invalid policy_type: {}".format(policy_type))

        for i_step in range(max_step):

            # Get a result of your action from the environment
            next_s, r, done, info = env.step(curr_a)

            # With some probability choose a random state
            rand = np.random.uniform(0, 1)
            if rand > threshold:
                next_s = np.random.randint(0, n_s)

            # Update a cummulative reward
            cumu_r = r + gamma * cumu_r

            # Select an action
            if policy_type == 'softmax':
                next_a = select_a_with_softmax(next_s, q_value, beta=beta)
            elif policy_type == 'epsilon_greedy':
                next_a = select_a_with_epsilon_greedy(next_s, q_value, epsilon=epsilon)
            else:
                raise ValueError("Invalid policy_type: {}".format(policy_type))

            # Calculation of TD error
            if update:
                # Only update table if update set to true
                noise = 0
                if robust:
                    value = get_value(q_value)
                    noise = get_sigma(value, p_est)
                if algorithm_type == 'sarsa':
                    delta = r + gamma * noise + gamma * q_value[next_s, next_a] - q_value[curr_s, curr_a]
                elif algorithm_type == 'q_learning':
                    delta = r + gamma * noise + gamma * np.max(q_value[next_s, :]) - q_value[curr_s, curr_a]
                else:
                    raise ValueError("Invalid algorithm_type: {}".format(algorithm_type))

                # Update a Q value table
                q_value[curr_s, curr_a] += alpha * delta
            curr_s = next_s
            curr_a = next_a
            if done:

                # Running average of the terminal reward, which is used for controlling an exploration rate
                # (This idea of controlling exploration rate by the terminal reward is suggested by JKCooper2)
                # See https://gym.openai.com/evaluations/eval_xSOlwrBsQDqUW7y6lJOevQ
                kappa = 0.01
                if ave_cumu_r == None:
                    ave_cumu_r = cumu_r
                else:
                    ave_cumu_r = kappa * cumu_r + (1 - kappa) * ave_cumu_r

                if cumu_r > ave_cumu_r:
                    # Bias the current policy toward exploitation

                    if policy_type == 'epsilon_greedy':
                        # epsilon is decayed expolentially
                        epsilon = epsilon * epsilon_decay
                    elif policy_type == 'softmax':
                        # beta is increased linearly
                        beta = beta + beta_inc

                if policy_type == 'softmax':
                    print("Episode: {0}\t Steps: {1:>4}\tCumuR: {2:>5.2f}\tTermR: {3}\tAveCumuR: {4:.3f}\tBeta: {5:.3f}".format(
                        i_episode, i_step, cumu_r, r, ave_cumu_r, beta))
                    history.append([i_episode, i_step, cumu_r, r, ave_cumu_r, beta])
                elif policy_type == 'epsilon_greedy':
                    print("Episode: {0}\t Steps: {1:>4}\tCumuR: {2:>5.2f}\tTermR: {3}\tAveCumuR: {4:.3f}\tEpsilon: {5:.3f}".format(
                        i_episode, i_step, cumu_r, r, ave_cumu_r, epsilon))
                    history.append([i_episode, i_step, cumu_r, r, ave_cumu_r, epsilon])
                else:
                    raise ValueError("Invalid policy_type: {}".format(policy_type))

                break

    # Stop monitoring the simulation for OpenAI Gym
    # env.monitor.close()
    history = np.array(history)

    print("Q_value = {0}".format(q_value))

    if policy_type == 'softmax':
        print("Action selection probability:")
        print(np.array([softmax(q, beta=beta) for q in q_value]))
    elif policy_type == 'epsilon_greedy':
        print("Greedy action")
        greedy_action = np.zeros([n_s, n_a])
        greedy_action[np.arange(n_s), np.argmax(q_value, axis=1)] = 1
        print(greedy_action)

    return history, q_value


def plot_cumulative_reward(robust_history, nominal_history, file_name):
    fig = plt.figure()
    plt.xlabel('Episode')
    plt.ylabel('Average cumulative reward')
    plt.plot(nominal_history[:, 0], nominal_history[:, 4], ls="-", label='Nominal')
    plt.plot(robust_history[:, 0], robust_history[:, 4], ls="-", label='Robust')
    plt.legend(loc='best', fontsize=14)
    fig.savefig(file_name)


def plot_tail_distribution(robust_history, nominal_history, file_name):
    fig = plt.figure()
    plt.xlabel('a')
    plt.ylabel('Pr[r > a]')

    num_bins = 5000
    values, base = np.histogram(nominal_history[:, 4], bins=num_bins)
    cumulative = np.cumsum(values[::-1])[::-1]
    cumulative = np.array(cumulative, dtype=np.float64)
    cumulative /= np.max(cumulative)

    robust_values, robust_base = np.histogram(robust_history[:, 4], bins=num_bins)
    robust_cumulative = np.cumsum(robust_values[::-1])[::-1]
    robust_cumulative = np.array(robust_cumulative, dtype=np.float64)
    robust_cumulative /= np.max(robust_cumulative)
    plt.plot(base[:-1], cumulative, ls='-', label='Nominal')
    plt.plot(robust_base[:-1], robust_cumulative, ls='-', label='Robust')
    plt.legend(loc='best', fontsize=14)
    fig.savefig(file_name)


def cross_validate(folds=5):
    p_env = args.prob
    epsilon = args.epsilon
    n_episode = args.nepisode
    p_est = 1e-9
    nu = 10
    rewards = []
    p_est_list = []
    num_runs = 1
    for _ in range(folds):
        r = 0
        nominal_r = 0
        for _ in range(num_runs):
            robust_learning_history, _ = q_learning(robust=True, p_env=p_env, p_est=p_est,
                                                    epsilon=epsilon, qtable=None,
                                                    update=True, n_episode=n_episode)
            r += robust_learning_history[:, 4][-1]
        r /= num_runs
        nominal_r /= num_runs
        p_est_list.append(p_est)
        rewards.append(r)
        p_est *= nu
    nominal_history, _ = q_learning(robust=True, p_env=p_env, p_est=0,
                                    epsilon=epsilon, qtable=None,
                                    update=True, n_episode=n_episode)
    nominal_rewards = [nominal_history[:, 4][-1]] * len(rewards)
    # Plot stuff
    cv_file_name = 'cross_validation_{0}_{1}.png'.format(args.environment, format_e(p_env))
    fig = plt.figure()
    p_est_list = np.array(p_est_list, dtype=np.float64)
    rewards = np.array(rewards, dtype=np.float64)
    p_est_list_log = np.log10(p_est_list)
    plt.xlabel('Estimated log probability')
    plt.ylabel('Average Reward')
    plt.plot(p_est_list_log, nominal_rewards, ls="-", label="Nominal")
    plt.plot(p_est_list_log, rewards, ls="-", label="Robust")
    plt.legend(loc='best', fontsize=14)
    fig.savefig(cv_file_name)
    print(("rewards = {}".format(rewards)))
    print(("p_est = {}".format(p_est_list)))

    # Return max p_est
    max_idx = np.argmax(rewards)
    return p_est_list[max_idx]


def compare_nominal(p_est):
    p_env = args.prob
    epsilon = args.epsilon
    n_episode = args.nepisode
    robust_learning_history, robust_qtable = q_learning(robust=True, p_env=p_env, p_est=p_est,
                                                        epsilon=epsilon, qtable=None,
                                                        update=True, n_episode=n_episode)
    learning_history, nominal_qtable = q_learning(robust=False, p_env=p_env, p_est=p_est,
                                                  epsilon=epsilon, qtable=None,
                                                  update=True, n_episode=n_episode)

    history, _ = q_learning(robust=True, p_env=0, p_est=0, epsilon=0, qtable=nominal_qtable,
                            update=False, n_episode=n_episode)
    robust_history, _ = q_learning(robust=True, p_env=0, p_est=0, epsilon=0, qtable=robust_qtable,
                                   update=False, n_episode=n_episode)

    # Plot cumulative reward of learning phase
    learning_file_name = 'cum_rewards_learning_{0}_{1}.png'.format(args.environment, format_e(p_env))
    plot_cumulative_reward(robust_learning_history, learning_history, learning_file_name)

    cum_file_name = 'cum_rewards_{0}_{1}.png'.format(args.environment, format_e(p_env))
    plot_cumulative_reward(robust_history, history, cum_file_name)

    # Plot tail distribution
    tail_file_name = 'cdf_comparison_{0}_{1}.png'.format(args.environment, format_e(p_env))
    plot_tail_distribution(robust_history, history, tail_file_name)


def format_e(n):
    a = '%e' % n
    return a.split('e')[0].rstrip('0').rstrip('.') + 'e' + a.split('e')[1]


def run_experiment(run_id, num_runs, args):
    env_name = args.environment # 'cliff'
    robust = args.robust # 'marginal'
    robust_type = args.rtype # 'joint'
    objective = args.robj # 'kl'
    size = args.rsize # 0.9
    alg = args.algorithm # 'td'
    outdir = args.outdir
    # num_runs = args.nrun # 10

    slip = 1
    slip_prob = args.shift # 0.9
    delta_threshold = 1e-2

    # td evaluation
    n_episode = args.nepisode

    if env_name=='cliff':
        env = CliffWalkingEnv(slip, slip_prob, seed=run_id) # random seed = 0,1,2,...,10 
    else:
        env = None

    # Policy
    random_policy = np.ones([env.nS, env.nA]) / env.nA

    if alg=='dp':
        v = dp_evaluation(robust=robust, robust_type=robust_type, env=env, policy=random_policy, n_episode=n_episode, delta_threshold=delta_threshold,
                        objective=objective, size=size)

        dirname = '{}/{}_v_value_alg{}_r{}_t{}_u{}_a{}_s{}_e{}_runs{}'.format(outdir, env_name, alg,
                                    robust, robust_type, objective, size, slip_prob, n_episode, num_runs)
        
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        print("Saving files to", dirname)

        np.save('{}/value_run{}.pickle'.format(dirname, run_id), v)
        
        print("Value Function:")
        print(v)
        print("")

        v_grids = v
        v_grid = v_grids[:,0].reshape(env.shape)
        print("Reshaped Grid Value Function:")
        print(v_grid)
        print("")

        ax = env.plot(v_grid, '{}/value_run{}_g1.png'.format(dirname, run_id))

        v_grid = v_grids[:,1].reshape(env.shape)
        print("Reshaped Grid Value Function:")
        print(v_grid)
        print("")

        ax = env.plot(v_grid, '{}/value_run{}_g2.png'.format(dirname, run_id))
    elif alg=='td':
        v = td_evaluation(robust=robust, robust_type=robust_type, env=env, policy=random_policy, n_episode=n_episode, objective=objective, size=size)

        dirname = '{}/{}_v_value_alg{}_r{}_t{}_u{}_a{}_s{}_e{}_runs{}'.format(outdir, env_name, alg,
                                    robust, robust_type, objective, size, slip_prob, n_episode, num_runs)
        
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        print("Saving files to", dirname)

        np.save('{}/value_run{}.pickle'.format(dirname, run_id), v)
        
        print("Value Function:")
        print(v)
        print("")

        v_grids = v
        v_grid = v_grids[:,0].reshape(env.shape)
        print("Reshaped Grid Value Function:")
        print(v_grid)
        print("")

        ax = env.plot(v_grid, '{}/value_run{}_g1.png'.format(dirname, run_id))

        v_grid = v_grids[:,1].reshape(env.shape)
        print("Reshaped Grid Value Function:")
        print(v_grid)
        print("")

        ax = env.plot(v_grid, '{}/value_run{}_g2.png'.format(dirname, run_id))
    elif alg=='q_learning':
        q, v = td_learning(robust=robust, robust_type=robust_type, env=env, policy_train=random_policy, n_episode=n_episode, objective=objective, size=size)

        p = np.argmax(q, axis=2)
        p = p.astype(str)
        p[p=="0"] = "U"
        p[p=="1"] = "R"
        p[p=="2"] = "D"
        p[p=="3"] = "L"

        dirname = '{}/{}_v_value_alg{}_r{}_t{}_u{}_a{}_s{}_e{}_runs{}'.format(outdir, env_name, alg,
                                    robust, robust_type, objective, size, slip_prob, n_episode, num_runs)
        
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        print("Saving files to", dirname)

        np.save('{}/value_run{}.pickle'.format(dirname, run_id), v)
        
        print("Value Function:")
        print(v)
        print("")

        v_grids = v
        p_grids = p
        v_grid = v_grids[:,0].reshape(env.shape)
        p_grid = p_grids[:,0].reshape(env.shape)
        print("Reshaped Grid Value Function:")
        print(v_grid)
        print("")

        ax = env.plot_actions(v_grid, p_grid, '{}/value_run{}_g1.png'.format(dirname, run_id))

        v_grid = v_grids[:,1].reshape(env.shape)
        p_grid = p_grids[:,1].reshape(env.shape)
        print("Reshaped Grid Value Function:")
        print(v_grid)
        print("")

        ax = env.plot_actions(v_grid, p_grid, '{}/value_runs{}_g2.png'.format(dirname, run_id))


def run_tdlearning(args):    
    env_name = args.environment # 'cliff'
    robust = args.robust # 'marginal'
    robust_type = args.rtype # 'joint'
    objective = args.robj # 'kl'
    size = args.rsize # 0.9
    alg = args.algorithm # 'td'
    outdir = args.outdir
    num_runs = args.nrun # 10

    slip = 2
    slip_prob = args.shift # 0.9
    delta_threshold = 1e-2

    # td evaluation
    n_episode = args.nepisode

    # env = GridworldEnv()
    if env_name=='cliff':
        env = CliffWalkingEnv(slip, slip_prob, seed=0)
    else:
        env = None

    # Policy
    random_policy = np.ones([env.nS, env.nA]) / env.nA

    # Evaluation
    # v = td_evaluation(robust=robust, robust_type=robust_type, env=env, policy=random_policy, n_episode=n_episode, objective=objective, size=size)

    # v_dp = dp_evaluation(robust=robust, robust_type=robust_type, env=env, policy=random_policy, delta_threshold=delta_threshold, objective=objective, size=size)
    
    # print("check equal td, dp", np.allclose(v, v_dp))

    q, v = td_learning(robust=robust, robust_type=robust_type, env=env, policy_train=random_policy, n_episode=n_episode, objective=objective, size=size)

    p = np.argmax(q, axis=2)
    p = p.astype(str)
    p[p=="0"] = "U"
    p[p=="1"] = "R"
    p[p=="2"] = "D"
    p[p=="3"] = "L"

    filename = 'cliffwalking_v_value_alg{}_learning_r{}_t{}_u{}_a{}_s{}'.format(alg, robust, robust_type, objective, size, slip_prob)
    print("Saving files to", filename)

    np.save('{}/{}.pickle'.format(outdir, filename), v)
    
    print("Value Function:")
    print(v)
    print("")

    v_grids = v
    p_grids = p
    v_grid = v_grids[:,0].reshape(env.shape)
    p_grid = p_grids[:,0].reshape(env.shape)
    print("Reshaped Grid Value Function:")
    print(v_grid)
    print("")

    # ax = env.plot(v_grid, '{}/{}_g1.png'.format(outdir, filename))
    ax = env.plot_actions(v_grid, p_grid, '{}/{}_g1.png'.format(outdir, filename))

    v_grid = v_grids[:,1].reshape(env.shape)
    p_grid = p_grids[:,1].reshape(env.shape)
    print("Reshaped Grid Value Function:")
    print(v_grid)
    print("")

    # ax = env.plot(v_grid, '{}/{}_g2.png'.format(outdir, filename))
    ax = env.plot_actions(v_grid, p_grid, '{}/{}_g2.png'.format(outdir, filename))


    # filename = 'cliffwalking_v_value_alg{}_r{}_t{}_u{}_a{}_s{}'.format('dp', robust, robust_type, objective, size, slip_prob)
    # print("Saving files to", filename)

    # np.save('{}/{}.pickle'.format(outdir, filename), v_dp)
    
    # # print("Value Function:")
    # # print(v_dp)
    # # print("")

    # v_grids = v_dp
    # v_grid = v_grids[:,0].reshape(env.shape)
    # print("Reshaped Grid Value Function:")
    # print(v_grid)
    # print("")

    # ax = env.plot(v_grid, '{}/{}_g1.png'.format(outdir, filename))

    # v_grid = v_grids[:,1].reshape(env.shape)
    # print("Reshaped Grid Value Function:")
    # print(v_grid)
    # print("")

    # ax = env.plot(v_grid, '{}/{}_g2.png'.format(outdir, filename))

    # if robust_type=='marginal':
    #     v_grid = v_1.reshape(env.shape)
    #     print("Reshaped Grid Value Function:")
    #     print(v_grid)
    #     print("")

    #     ax = env.plot(v_grid, '{}_exact_dp.png'.format(filename))

if __name__ == "__main__":
    # # Cross validate
    # p_est = cross_validate(folds=10)
    # # First compare nominal
    # compare_nominal(p_est=p_est)
    # print(("p_est = {}".format(p_est)))

    print('arguments', args)

    num_runs = args.nrun # 10

    # for run_id in range(num_runs):
    #     run_experiment(run_id, num_runs, args)

    results = Parallel(n_jobs=min(30,num_runs))(delayed(run_experiment)(\
                                    run_id, num_runs, args) for run_id in range(num_runs))