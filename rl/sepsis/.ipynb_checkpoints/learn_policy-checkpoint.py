'''
Learn optimal policy from sepsis simulator, simulate trajectories and generate transition, reward models
'''

import numpy as np
from .cf import counterfactual as cf
from .cf import utils
import pandas as pd
import pickle
import itertools as it
from tqdm.notebook import tqdm
from scipy.linalg import block_diag

# Sepsis Simulator code
from .sepsisSimDiabetes.State import State
from .sepsisSimDiabetes.Action import Action
from .sepsisSimDiabetes.DataGenerator import DataGenerator
from .sepsisSimDiabetes import MDP as simulator 

from .mdptoolboxSrc import mdp as mdptools

def get_physician_policy(config, SEED):
    np.random.seed(SEED)
    NSIMSAMPS = 1000  # Samples to draw from the simulator
    NSTEPS = config['max_horizon'] # 20  # Max length of each trajectory
    NCFSAMPS = 1  # Counterfactual Samples per observed sample
    DISCOUNT_Pol = config['discount'] # 0.99 # Used for computing optimal policies
    DISCOUNT = config['discount'] # 1 # Used for computing actual reward
    PHYS_EPSILON = 0.05 # Used for sampling using physician pol as eps greedy

    PROB_DIAB = config['p_diabetes'] # 0.2

    # Number of iterations to get error bars
    N_REPEAT_SAMPLING = 100
    NHELDOUT = 1000 # Heldout samples for WIS

    # These are properties of the simulator, do not change
    n_actions = Action.NUM_ACTIONS_TOTAL
    n_components = 2
    n_glucose_levels = 5

    # These are added as absorbing states
    # NOTE: Change to full states
    n_states_abs = State.NUM_FULL_STATES + 2
    discStateIdx = n_states_abs - 1
    deadStateIdx = n_states_abs - 2

    # Get the transition and reward matrix from file
    with open("./data/diab_txr_mats-replication.pkl", "rb") as f:
        mdict = pickle.load(f)

    tx_mat = mdict["tx_mat"]
    r_mat = mdict["r_mat"]
    p_mixture = np.array([1 - PROB_DIAB, PROB_DIAB])

    from scipy.linalg import block_diag

    tx_mat_full = np.zeros((n_actions, State.NUM_FULL_STATES, State.NUM_FULL_STATES))
    r_mat_full = np.zeros((n_actions, State.NUM_FULL_STATES, State.NUM_FULL_STATES))

    for a in range(n_actions):
        tx_mat_full[a, ...] = block_diag(tx_mat[0, a, ...], tx_mat[1, a,...])
        r_mat_full[a, ...] = block_diag(r_mat[0, a, ...], r_mat[1, a, ...])

    fullMDP = cf.MatrixMDP(tx_mat_full, r_mat_full)
    fullPol = fullMDP.policyIteration(discount=DISCOUNT_Pol, eval_type=1)

    physPolSoft = np.copy(fullPol)
    physPolSoft[physPolSoft == 1] = 1 - PHYS_EPSILON
    physPolSoft[physPolSoft == 0] = PHYS_EPSILON / (n_actions - 1)

    with open('./data/physician_policy_st.pkl', 'wb') as fw:
        pickle.dump(physPolSoft, fw)


    ###### Learn RL policy from simulated data using physPolSoft

    # Construct the projection matrix for obs->proj states

    # In this case, this is an identity matrix
    n_proj_states = n_states_abs
    proj_matrix = np.eye(n_states_abs)

    proj_matrix = proj_matrix.astype(int)

    proj_lookup = proj_matrix.argmax(axis=-1)

    dgen = DataGenerator()
    states, actions, lengths, rewards, diab, emp_tx_totals, emp_r_totals = dgen.simulate(
    NSIMSAMPS, NSTEPS, policy=physPolSoft, policy_idx_type='full', 
    output_state_idx_type='full', 
    p_diabetes=PROB_DIAB, use_tqdm=False) #True, tqdm_desc='Behaviour Policy Simulation')

    obs_samps = utils.format_dgen_samps(
    states, actions, rewards, diab, NSTEPS, NSIMSAMPS)

    emp_tx_mat_full = np.copy(emp_tx_totals)
    emp_r_mat_full = np.copy(emp_r_totals)

    # (2) Add new aborbing states, and a new est_tx_mat with Absorbing states
    death_states = (emp_r_mat_full.sum(axis=0).sum(axis=0) < 0)
    disch_states = (emp_r_mat_full.sum(axis=0).sum(axis=0) > 0)

    est_tx_cts_abs = np.zeros((n_actions, n_states_abs, n_states_abs))
    est_tx_cts_abs[:, :-2, :-2] = np.copy(emp_tx_mat_full)

    death_states = np.concatenate([death_states, np.array([True, False])])
    disch_states = np.concatenate([disch_states, np.array([False, True])])
    assert est_tx_cts_abs[:, death_states, :].sum() == 0
    assert est_tx_cts_abs[:, disch_states, :].sum() == 0

    est_tx_cts_abs[:, death_states, deadStateIdx] = 1
    est_tx_cts_abs[:, disch_states, discStateIdx] = 1

    # (3) Project the new est_tx_cts_abs to the reduced state space
    # PASS IN THIS CASE
    proj_tx_cts = np.copy(est_tx_cts_abs)
    proj_tx_mat = np.zeros_like(proj_tx_cts)

    # Normalize
    nonzero_idx = proj_tx_cts.sum(axis=-1) != 0
    proj_tx_mat[nonzero_idx] = proj_tx_cts[nonzero_idx]

    proj_tx_mat[nonzero_idx] /= proj_tx_mat[nonzero_idx].sum(axis=-1, keepdims=True)

    ############ Construct the reward matrix, which is known ##################
    proj_r_mat = np.zeros((n_actions, n_proj_states, n_proj_states))
    proj_r_mat[..., -2] = -1
    proj_r_mat[..., -1] = 1

    proj_r_mat[..., -2, -2] = 0 # No reward once in aborbing state
    proj_r_mat[..., -1, -1] = 0

    ############ Construct the empirical prior on the initial state ##################
    initial_state_arr = np.copy(states[:, 0, 0])
    initial_state_counts = np.zeros((n_states_abs,1))
    for i in range(initial_state_arr.shape[0]):
        initial_state_counts[initial_state_arr[i]] += 1

    # Project initial state counts to new states
    proj_state_counts = proj_matrix.T.dot(initial_state_counts).T
    proj_p_initial_state = proj_state_counts / proj_state_counts.sum()

    # Check projection is still identity
    assert np.all(proj_state_counts.T == initial_state_counts)

    # Because some SA pairs are never observed, assume they cause instant death
    zero_sa_pairs = proj_tx_mat.sum(axis=-1) == 0
    proj_tx_mat[zero_sa_pairs, -2] = 1  # Always insta-death if you take a never-taken action

    # Construct an extra axis for the mixture component, of which there is only one
    projMDP = cf.MatrixMDP(proj_tx_mat, proj_r_mat, 
                        p_initial_state=proj_p_initial_state)
    try:
        RlPol = projMDP.policyIteration(discount=DISCOUNT_Pol)
    except:
        assert np.allclose(proj_tx_mat.sum(axis=-1), 1)
        RlPol = projMDP.policyIteration(discount=DISCOUNT_Pol, skip_check=True)

    # Get the true RL reward as a sanity check
    # Note that the RL policy includes actions for "death" and "discharge" absorbing states, which we ignore by taking [:-2, :]
    NSIMSAMPS_RL = NSIMSAMPS
    states_rl, actions_rl, lengths_rl, rewards_rl, diab_rl, _, _ = dgen.simulate(
    NSIMSAMPS_RL, NSTEPS, policy=RlPol[:-2, :], 
    policy_idx_type='full', # Note the difference 
    p_diabetes=PROB_DIAB, use_tqdm=False) #True, tqdm_desc='RL Policy Simulation')

    obs_samps_rlpol = utils.format_dgen_samps(
    states_rl, actions_rl, rewards_rl, diab_rl, NSTEPS, NSIMSAMPS_RL)

    this_true_rl_reward = cf.eval_on_policy(
        obs_samps_rlpol, discount=DISCOUNT, 
        bootstrap=False)  # Need a second axis to concat later

    with open('./data/rl_policy.pkl', 'wb') as fw:
        pickle.dump(RlPol, fw)

def sample_trajectories(config, RlPol, SEED):
    np.random.seed(SEED)
    NSIMSAMPS = config['num_iters'] # 1000  # Samples to draw from the simulator
    NSTEPS = config['max_horizon'] # 20  # Max length of each trajectory
    NCFSAMPS = 1  # Counterfactual Samples per observed sample
    DISCOUNT_Pol = config['discount'] # 0.99 # Used for computing optimal policies
    DISCOUNT = config['discount'] # 1 # Used for computing actual reward
    PHYS_EPSILON = 0.05 # Used for sampling using physician pol as eps greedy

    PROB_DIAB = config['p_diabetes'] # 0.2

    # Number of iterations to get error bars
    N_REPEAT_SAMPLING = 100
    NHELDOUT = 1000 # Heldout samples for WIS

    # These are properties of the simulator, do not change
    n_actions = Action.NUM_ACTIONS_TOTAL
    n_components = 2
    n_glucose_levels = 5

    # These are added as absorbing states
    # NOTE: Change to full states
    n_states_abs = State.NUM_FULL_STATES + 2
    discStateIdx = n_states_abs - 1
    deadStateIdx = n_states_abs - 2

    # Construct the projection matrix for obs->proj states

    # In this case, this is an identity matrix
    n_proj_states = n_states_abs
    proj_matrix = np.eye(n_states_abs)

    proj_matrix = proj_matrix.astype(int)

    proj_lookup = proj_matrix.argmax(axis=-1)

    ###### Simulate data using RLPol

    dgen = DataGenerator()
    states, actions, lengths, rewards, diab, emp_tx_totals, emp_r_totals = dgen.simulate(
    NSIMSAMPS, NSTEPS, policy=RlPol[:-2, :], policy_idx_type='full', 
    output_state_idx_type='full', 
    p_diabetes=PROB_DIAB, use_tqdm=False) #True, tqdm_desc='Behaviour Policy Simulation')

    obs_samps = utils.format_dgen_samps(
    states, actions, rewards, diab, NSTEPS, NSIMSAMPS)

    emp_tx_mat_full = np.copy(emp_tx_totals)
    emp_r_mat_full = np.copy(emp_r_totals)

    # (2) Add new aborbing states, and a new est_tx_mat with Absorbing states
    death_states = (emp_r_mat_full.sum(axis=0).sum(axis=0) < 0)
    disch_states = (emp_r_mat_full.sum(axis=0).sum(axis=0) > 0)

    proj_tx_cts = np.copy(emp_tx_mat_full)
    proj_tx_mat = np.zeros_like(proj_tx_cts)

    # Normalize
    nonzero_idx = proj_tx_cts.sum(axis=-1) != 0
    proj_tx_mat[nonzero_idx] = proj_tx_cts[nonzero_idx]

    proj_tx_mat[nonzero_idx] /= proj_tx_mat[nonzero_idx].sum(axis=-1, keepdims=True)

    proj_r_cts = np.copy(emp_r_mat_full)
    proj_r_mat = np.zeros_like(proj_r_cts)

    proj_r_mat[:,:,death_states] = -1
    proj_r_mat[:,:,disch_states] = 1

    # with open('./sepsis/data/tx_tr_obs_rlpol_seed{}.pkl'.format(SEED), 'wb') as fw:
    #     pickle.dump((proj_tx_mat, proj_r_mat, obs_samps), fw)

    return (proj_tx_mat, proj_r_mat, obs_samps)

if __name__ == "__main__":
    config = {
        'p_diabetes': 0.2, 'max_horizon': 20,
        'num_iters': 1000, 'discount': 0.99
    }
    get_physician_policy(config)