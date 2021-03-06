from envs.deep_cure_env import DeepCure, ForeignCountry, random_base_infect_rate
from plotting import plot
import gym
import numpy as np
import math
import matplotlib.pyplot as plt

def discretize(state, stepsize, num_states):
    return np.minimum(state / stepsize, num_states - 1).astype(int)

def index(state, action = None):
    if action is None:
        # corresponds to [:]
        action = slice(None,None)
    else:
        action = sum([b * (2 ** i) for i,b in enumerate(action)])
    index = tuple((*state,action))
    return index

def greedy_policy(state, q_array):
    action_index = np.argmax(q_array[index(state)])
    action = np.array([int((action_index / (2 ** i)) % 2) for i in range(int(math.log2(q_array.shape[-1])))])
    # if p:
    #     print(f'state = {state}: actions = {q_array[index(q_array, state)]}, pick {action} ({action_index})')
    return action

def epsilon_greedy_policy(env, state, q_array, epsilon):

    ### BEGIN SOLUTION
    if np.random.rand() < epsilon:
        action = env.action_space.sample()
    else:
        action = greedy_policy(state, q_array)
    ### END SOLUTION

    return action

def q_learning(environment, alpha=0.1, alpha_factor=0.9995, gamma=0.99, epsilon=0.5, num_episodes=10000, rate = None, stepsize = 20, max_steps = 100):
    q_array_history = [0]
    last_q_array = None
    alpha_history = []
    num_states = np.minimum((environment.observation_space.high - environment.observation_space.low)/stepsize, max_steps).astype(int)
    num_actions = 2**environment.action_space.n
    q_array = np.zeros(list(num_states) + [num_actions])   # Initial Q table
    print(f'QTable = {q_array.shape}')

    for episode_index in range(num_episodes):
        alpha_history.append(alpha)

        # Update alpha
        if alpha_factor is not None:
            alpha = alpha * alpha_factor

        ### BEGIN SOLUTION ###

        is_final_state = False
        state = discretize(environment.reset(rate), stepsize, num_states)

        while not is_final_state:
            action = epsilon_greedy_policy(environment, state, q_array, epsilon)
            new_state, reward, is_final_state, info = environment.step(action)

            new_state = discretize(new_state, stepsize, num_states)
            new_action = greedy_policy(new_state, q_array)
            q_array[index(state, action)] = q_array[index(state, action)] + alpha * (reward + gamma * q_array[index(new_state, new_action)] - q_array[index(state, action)])

            state = new_state

        if last_q_array is not None:
            q_array_history.append(np.max(np.absolute(q_array - last_q_array)))
        last_q_array = q_array.copy()

        ### END SOLUTION ###

    return q_array, q_array_history, alpha_history

if __name__ == "__main__":
    SEED = 42

    np.random.seed(SEED)

    env = DeepCure(foreign_countries = [ForeignCountry(0.1,100,100_000, save_history=True)], save_history=True, seed = SEED)

    stepsize = 100
    max_steps = 10

    q_table, q_array_history, alpha_history = q_learning(env, epsilon=0.5, stepsize = stepsize, max_steps = max_steps)
    np.save(f'qtable-{stepsize}-{max_steps}.npy', q_table)
#
# fig = plt.figure()
# ax0 = fig.add_subplot(2,1,1)
# ax0.set_title('Q Table Convergence')
# ax0.set_xlabel('iterations')
# ax0.set_ylabel('absolute difference')
# ax0.plot(range(len(q_array_history)), q_array_history)
#
# ax1 = fig.add_subplot(2,1,2)
# ax1.set_title('$\\alpha$')
# ax1.set_xlabel('iterations')
# ax1.set_ylabel('$\\alpha$')
# ax1.plot(range(len(alpha_history)), alpha_history)
#
# fig.tight_layout()
#
# plt.show()

# q_table = np.load('qtable-1000-100.npy')
# print(f'Explored : {np.sum(q_table != 0) / q_table.size}')

# # q_table = np.load('qtable.npy')
# rate = 2.5 #random_base_infect_rate()
#
# def constant_action(action):
#     state = env.reset(rate)
#     end = False
#     t = 0
#     while not end :
#         state, reward, end, _ = env.step(action)
#         t += 1
#     print(f'Action = {action}')
#     print(f'Total reward: {sum(env.hist_reward)}')
#     print(f'Total number of dead: {env.number_of_dead}')
#     print(f'Total number of severe: {env.number_of_severe}')
#
# constant_action([0,0,0])
# constant_action([1,0,0])
# constant_action([0,1,0])
# constant_action([1,1,0])
#
# state = discretize(env.reset(rate), stepsize, np.array([10,10,10]))
# end = False
# t = 0
# while not end :
#     action = greedy_policy(state, q_table)
#     # if t > 4:
#     #     action = 2
#     state, reward, end, _ = env.step(action)
#     print(f'Before = {state}')
#     state = discretize(state, stepsize, np.array([10,10,10]))
#     print(f'After = {state}')
#     t += 1
#
# print('')
# print(f'Total number of dead: {env.number_of_dead}')
# print(f'Total number of severe: {env.number_of_severe}')
# print(f'Total reward = {sum(env.hist_reward)}')
# plot(env)
