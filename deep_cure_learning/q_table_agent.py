from envs.deep_cure_env import DeepCure, random_base_infect_rate
from plotting import plot
import gym
import numpy as np

env = DeepCure(foreign_countries = [], stepsize = 20, save_history=True)
env.reset()

def greedy_policy(state, q_array):

    ### BEGIN SOLUTION
    action = np.argmax(q_array[state, :])
    ### END SOLUTION

    return action


def epsilon_greedy_policy(state, q_array, epsilon):

    ### BEGIN SOLUTION
    if np.random.rand() < epsilon:
        action = np.random.randint(q_array.shape[1])
    else:
        action = greedy_policy(state, q_array)
    ### END SOLUTION

    return action

def q_learning(environment, alpha=0.1, alpha_factor=0.9995, gamma=0.99, epsilon=0.5, num_episodes=10000, display=False):
    q_array_history = []
    alpha_history = []
    num_states = environment.state_space_size()
    num_actions = 4
    q_array = np.zeros([num_states, num_actions])   # Initial Q table

    for episode_index in range(num_episodes):
        if display and episode_index % DISPLAY_EVERY_N_EPISODES == 0:
            qtable_display(q_array, title="Q table", cbar=True)
        else:
            print('.', end="")
        q_array_history.append(q_array.copy())
        alpha_history.append(alpha)

        # Update alpha
        if alpha_factor is not None:
            alpha = alpha * alpha_factor

        ### BEGIN SOLUTION ###

        is_final_state = False
        state = environment.reset()

        while not is_final_state:
            action = epsilon_greedy_policy(state, q_array, epsilon)
            new_state, reward, is_final_state, info = environment.step(action)

            #if reward > 0:
            #    print(reward)

            #if is_final_state:
            #    print(state)

            new_action = greedy_policy(new_state, q_array)
            q_array[state, action] = q_array[state, action] + alpha * (reward + gamma * q_array[new_state, new_action] - q_array[state, action])

            state = new_state

        ### END SOLUTION ###

    return q_array, q_array_history, alpha_history

q_table, _, _ = q_learning(env)

rate = random_base_infect_rate()

def constant_action(action):
    state = env.reset(rate)
    end = False
    t = 0
    while not end :
        state, reward, end, _ = env.step(action)
        t += 1
    print(f'Action = {action}')
    print(f'Total reward: {sum(env.hist_reward)}')
    print(f'Total number of dead: {env.number_of_dead}')
    print(f'Total number of severe: {env.number_of_severe}')

constant_action(0)
constant_action(1)
constant_action(2)
constant_action(3)

state = env.reset(rate)
end = False
t = 0
while not end :
    action = greedy_policy(state, q_table)
    state, reward, end, _ = env.step(action)
    t += 1

print(f'Total number of dead: {env.number_of_dead}')
print(f'Total number of severe: {env.number_of_severe}')
print(f'Total reward = {sum(env.hist_reward)}')
plot(env)
