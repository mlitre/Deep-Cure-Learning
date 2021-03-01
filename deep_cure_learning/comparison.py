from q_table_agent import greedy_policy, discretize
from deep_q_agent import logistic_regression
from envs.deep_cure_env import DeepCure, ForeignCountry, random_base_infect_rate, random_lifetime

import numpy as np

def constant_action(action,rate, lifetime):
    state = env.reset(rate, lifetime)
    end = False
    while not end :
        state, reward, end, _ = env.step(action)
    return sum(env.hist_reward)

def deep_q(env, theta, rate, lifetime):
    obs = env.reset(rate)
    done = False
    while not done:
        probs = logistic_regression(obs, theta)
        actions = probs >= 0.5
        obs, reward, done, _ = env.step(actions)
    return sum(env.hist_reward)

def q_table(env, table, stepsize, num_states, rate, lifetime):
    state = discretize(env.reset(rate), stepsize, num_states)
    end = False
    t = 0
    while not end :
        action = greedy_policy(state, table)
        state, reward, end, _ = env.step(action)
        state = discretize(state, stepsize, num_states)
    return sum(env.hist_reward)

def compare(env,theta, q_tables, n = 100):
    cnts = [0] * (5 + len(q_tables))
    rewards = [0] * (5 + len(q_tables))
    reward_hist = list()
    for i in range(n):
        rate = random_base_infect_rate()
        lifetime = random_lifetime()
        rewards[0] = constant_action([False,False,False], rate, lifetime)
        rewards[1] = constant_action([True,False,False], rate, lifetime)
        rewards[2] = constant_action([False,True,False], rate, lifetime)
        rewards[3] = constant_action([True,True,False], rate, lifetime)
        rewards[4] = deep_q(env, theta, rate, lifetime)
        for i,(table,stepsize,num_states) in enumerate(q_tables):
            rewards[5+i] = q_table(env, table, stepsize, num_states, rate, lifetime)

        reward_hist.append(list(rewards))
        cnts[np.argmax(rewards)] += 1

    reward_hist = np.array(reward_hist)
    print(f'Action Nothing : {cnts[0]}')
    print(f'Action Masks : {cnts[1]}')
    print(f'Action Curfew : {cnts[2]}')
    print(f'Action All : {cnts[3]}')
    print(f'Action deep-q agent: {cnts[4]}')
    for i,(_,stepsize,_) in enumerate(q_tables):
        print(f'Action q_table {stepsize}: {cnts[5+i]}')
    print()
    print(f'Action nothing\t\t{np.mean(reward_hist[:,0])} ({np.std(reward_hist[:,0])})')
    print(f'Action masks\t\t{np.mean(reward_hist[:,1])} ({np.std(reward_hist[:,1])})')
    print(f'Action curfew\t\t{np.mean(reward_hist[:,2])} ({np.std(reward_hist[:,2])})')
    print(f'Action all\t\t{np.mean(reward_hist[:,3])} ({np.std(reward_hist[:,3])})')
    print(f'Action agent\t\t{np.mean(reward_hist[:,4])} ({np.std(reward_hist[:,4])})')
    for i,(_,stepsize,_) in enumerate(q_tables):
        print(f'Action q_table {stepsize}:\t\t{np.mean(reward_hist[:,5+i])} ({np.std(reward_hist[:,5+i])}')

SEED = 21
np.random.seed(SEED)
env = DeepCure(foreign_countries = [ForeignCountry(500,100,100_000, save_history=True)], save_history=True, seed=SEED)
env.reset()


theta = np.array([[-3.69626016e-04, -4.26768075e-04,  3.24879127e-04, -1.00518075e-03],
 [ 3.95053506e-04,  7.77401462e-04, -3.50329017e-04, -3.46842164e-03],
 [-5.65304779e-04, -1.83368648e-03, -1.03550497e-05,  1.65541727e-03]])

q_tables = [
    (np.load('qtable-100-10.npy'), 100, np.minimum((env.observation_space.high - env.observation_space.low)/10, 10).astype(int)),
    (np.load('qtable-1000-100.npy'), 1000, np.minimum((env.observation_space.high - env.observation_space.low)/100, 100).astype(int))
]

compare(env, theta, q_tables)
