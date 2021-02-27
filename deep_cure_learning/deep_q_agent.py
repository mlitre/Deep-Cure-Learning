from envs.deep_cure_env import DeepCure, random_base_infect_rate
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

def draw_action(s, theta):
    # theta : dim action space x state space
    probs = logistic_regression(s, theta)
    # print(probs)

    actions = np.random.rand(probs.shape[0]) < probs
    return actions

# Generate an episode
def play_one_episode(env, theta, rate = None):
    if rate is not None:
        s_t = env.reset(rate)
    else:
        s_t = env.reset()

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

    return episode_states, episode_actions, episode_rewards

def score_on_multiple_episodes(env, theta, rate = None, num_episodes=10):
    average_return = 0
    for episode_index in range(num_episodes):
        _, _, episode_rewards = play_one_episode(env, theta, rate)
        total_rewards = sum(episode_rewards)
        average_return += (1.0 / num_episodes) * total_rewards
        # print("Test Episode {0}: Total Reward = {1}".format(episode_index,total_rewards))
    return average_return

def compute_policy_gradient(episode_states, episode_actions, episode_rewards, theta):
    ### BEGIN SOLUTION ###
    H = len(episode_rewards)
    PG = np.zeros_like(theta)
    for t in range(H):
        probs = logistic_regression(episode_states[t], theta)
        a_t = episode_actions[t]
        print(probs)
        R_t = sum(episode_rewards[t::])
        states = np.ones(len(episode_states[t])+1)
        states[:len(episode_states[t])] = episode_states[t]
        for i,prob in enumerate(probs):
            if not a_t[i]:
                g_theta_log_pi = - prob * states * R_t
            else:
                g_theta_log_pi = (1-prob) * states * R_t
            # print(f'prob = {prob}, states = {states}, R_t = {R_t}')
            PG[i] += g_theta_log_pi
    ### END SOLUTION ###
    return PG

def train(env, theta_init, alpha_init = 0.0001, iterations = 1000, rate = None):

    theta = theta_init
    average_returns = []

    R = score_on_multiple_episodes(env, theta, rate)
    average_returns.append(R)

    # Train until success
    for i in range(iterations):
        episode_states, episode_actions, episode_rewards = play_one_episode(env, theta, rate)
        alpha = alpha_init /(1+i)
        PG = compute_policy_gradient(episode_states, episode_actions, episode_rewards, theta)
        theta = theta + alpha * PG
        R = score_on_multiple_episodes(env, theta, rate)
        average_returns.append(R)

    return theta, average_returns

env = DeepCure(foreign_countries = [], save_history=True)
env.reset()

theta = np.random.randn(1,3)
# sigmoid(b * w2) >= 0.5

theta, average_returns = train(env, theta)
print(theta)

obs = env.reset(2)
done = False
while not done:
    probs = logistic_regression(obs, theta)
    actions = probs >= 0.5
    # for prob in probs:
    #     actions.append(prob >= 0.5)
    # actions = [False, False]
    obs, reward, done, _ = env.step(actions)

print(f'Total number of dead: {env.number_of_dead}')
print(f'Total number of severe: {env.number_of_severe}')
print(f'Total reward = {sum(env.hist_reward)}')
plot(env)
