from envs.deep_cure_env import DeepCure, random_base_infect_rate, random_lifetime, ForeignCountry
from plotting import plot
import gym
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def logistic_regression(s, theta):
    # s: state space
    # theta action space x state space +1
    # result action space
    a = np.zeros((len(s)+1,1))
    a[:len(s),0] = s
    a[len(s),0] = 1
    prob_active = sigmoid(theta @ a) #s.reshape(len(s),1))
    return prob_active.reshape(-1)

def draw_action(s, theta, epsilon = 0.5):
    # if np.random.rand() < epsilon:
    #     rand_action = [np.random.rand() < 0.5 for i in range(theta.shape[0])]
    #     return rand_action
    # theta : dim action space x state space
    probs = logistic_regression(s, theta)
    # print(probs)

    actions = np.random.rand(probs.shape[0]) < probs
    return actions

# Generate an episode
def play_one_episode(env, theta, rate = None, lifetime = None):
    s_t = env.reset(rate=rate, lifetime=lifetime)

    episode_states = []
    episode_actions = []
    episode_rewards = []
    episode_states.append(s_t)

    done = False
    while not done:
        a_t = draw_action(s_t, theta)
        s_t, r_t, done, info = env.step(a_t)
        # print(f'Action = {a_t}')

        episode_states.append(s_t)
        episode_actions.append(a_t)
        episode_rewards.append(r_t)

    return episode_states, episode_actions, episode_rewards, env.v_base_infect_rate

def play_baseline(env, rate):
    def play(action):
        s_t = env.reset(rate)
        episode_rewards = []

        done = False
        while not done:
            a_t = action
            s_t, r_t, done, info = env.step(a_t)
            episode_rewards.append(r_t)
        return episode_rewards
    rewards_false = play([False])
    rewards_true = play([True])
    return [max(a,b) for a,b in zip(rewards_false, rewards_true)]

def score_on_multiple_episodes(env, theta, rate = None, lifetime = None, num_episodes=10):
    average_return = 0
    for episode_index in range(num_episodes):
        _, _, episode_rewards, _ = play_one_episode(env, theta, rate, lifetime)
        total_rewards = sum(episode_rewards)
        average_return += (1.0 / num_episodes) * total_rewards
        # print("Test Episode {0}: Total Reward = {1}".format(episode_index,total_rewards))
    return average_return

def compute_policy_gradient(episode_states, episode_actions, episode_rewards, theta, rate):
    ### BEGIN SOLUTION ###
    H = len(episode_rewards)
    PG = np.zeros_like(theta)
    # baseline = play_baseline(env, rate)
    for t in range(H):
        probs = logistic_regression(episode_states[t], theta)
        a_t = episode_actions[t]
        # print(probs)
        R_t = sum(episode_rewards[t::])
        baseline_t = 0#sum(baseline[t::])
        # episode_states[t] : observations at time t [previous new infected, new infected, 1]
        states = np.ones(len(episode_states[t])+1)
        states[:len(episode_states[t])] = episode_states[t]
        for i,prob in enumerate(probs):
            if not a_t[i]:
                g_theta_log_pi = -(1-prob) * states * (R_t - baseline_t)
            else:
                g_theta_log_pi = prob * states * (R_t - baseline_t)
            # print(f'prob = {prob}, states = {states}, R_t = {R_t}')
            PG[i] += g_theta_log_pi
    ### END SOLUTION ###
    return PG

def train(env, theta_init, alpha_init = 0.001, iterations = 2000, rate = None, lifetime = None):

    theta = theta_init
    average_returns = []

    R = score_on_multiple_episodes(env, theta, rate, lifetime)
    average_returns.append(R)

    # Train until success
    for i in range(iterations):
        episode_states, episode_actions, episode_rewards, played_rate = play_one_episode(env, theta, rate, lifetime)
        alpha = alpha_init /np.sqrt(1+i)
        PG = compute_policy_gradient(episode_states, episode_actions, episode_rewards, theta, played_rate)
        old_theta = np.copy(theta)
        theta = theta + alpha * PG
        # print(f'PG = {PG}')
        # if i < 10:
        #     print(f'theta = {theta}')
        R = score_on_multiple_episodes(env, theta, rate, lifetime)
        average_returns.append(R)

    return theta, average_returns

env = DeepCure(foreign_countries = [ForeignCountry(500,100,100_000, save_history=True)], save_history=True)
env.reset()

theta = np.zeros((3,4))#np.random.randn(1,5)
# sigmoid(b * w2) >= 0.5

theta, average_returns = train(env, theta)

# theta = np.array([[-1, 1, 0,0,0]])
# average_returns = []
# theta = np.array([[-0.49629676,  3.13567235, 0]])

def constant_action(action,rate, lifetime):
    state = env.reset(rate, lifetime)
    end = False
    t = 0
    while not end :
        state, reward, end, _ = env.step(action)
        t += 1
    return sum(env.hist_reward)

def compare(env,theta, n = 100):
    cnts = [0] * 5
    rewards = [0] * 5
    reward_hist = list()
    for i in range(n):
        rate = random_base_infect_rate()
        lifetime = random_lifetime()
        rewards[0] = constant_action([False,False,False], rate, lifetime)
        rewards[1] = constant_action([True,False,False], rate, lifetime)
        # if i +1 >= n:
        #     plot(env, average_returns)
        rewards[2] = constant_action([False,True,False], rate, lifetime)
        rewards[3] = constant_action([True,True,False], rate, lifetime)

        obs = env.reset(rate)
        done = False
        while not done:
            probs = logistic_regression(obs, theta)
            actions = probs >= 0.5
            obs, reward, done, _ = env.step(actions)
        rewards[4] = sum(env.hist_reward)

        reward_hist.append(list(rewards))
        cnts[np.argmax(rewards)] += 1

    reward_hist = np.array(reward_hist)
    print(f'Action Nothing : {cnts[0]}')
    print(f'Action Masks : {cnts[1]}')
    print(f'Action Curfew : {cnts[2]}')
    print(f'Action All : {cnts[3]}')
    print(f'Action agent: {cnts[4]}')
    print()
    print(f'Action nothing\t\t{np.mean(reward_hist[:,0])} ({np.std(reward_hist[:,0])})')
    print(f'Action masks\t\t{np.mean(reward_hist[:,1])} ({np.std(reward_hist[:,1])})')
    print(f'Action curfew\t\t{np.mean(reward_hist[:,2])} ({np.std(reward_hist[:,2])})')
    print(f'Action all\t\t{np.mean(reward_hist[:,3])} ({np.std(reward_hist[:,3])})')
    print(f'Action agent\t\t{np.mean(reward_hist[:,4])} ({np.std(reward_hist[:,4])})')

compare(env, theta)

# import matplotlib.pyplot as plt
# plt.figure()
# plt.plot(range(len(average_returns)), average_returns)
# plt.show()

# constant_action([False, False],rate)
# constant_action([False,True], rate)
# constant_action([True, False],rate)
# constant_action([True,True], rate)
#
# obs = env.reset(rate)
# done = False
# print(f'Theta = {theta}')
# while not done:
#     # actions = draw_action(obs, theta)
#     probs = logistic_regression(obs, theta)
#     actions = probs >= 0.5
#     # print(f'Policy = {actions}')
#     # print(actions)
#     # for prob in probs:
#     #     actions.append(prob >= 0.5)
#     # actions = [False, False]
#     obs, reward, done, _ = env.step(actions)
# print('Agent')
# print(f'Total number of dead: {env.number_of_dead}')
# print(f'Total number of severe: {env.number_of_severe}')
# print(f'Total reward = {sum(env.hist_reward)}')
# # print(f'Reward baseline = {sum(play_baseline(env,rate))}')

plot(env, average_returns)
