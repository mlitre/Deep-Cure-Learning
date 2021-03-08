from envs.deep_cure_env import DeepCure, random_base_infect_rate, random_lifetime, ForeignCountry
from plotting import plot
import gym
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt

def relu(x):
    x_and_zeros = np.array([x, np.zeros(x.shape)])
    return np.max(x_and_zeros, axis=0)

def sigmoid(x):
    return 1/(1+np.exp(-x))

class NeuralNetworkPolicy:

    def __init__(self, env, h_size=16, one_layer=False):   # h_size = number of neurons on the hidden layer

        if one_layer:
            self.activation_functions = (sigmoid,)
            weights = (np.zeros([env.observation_space.shape[0] + 1, env.action_space.shape[0]]),)
        else:
            self.activation_functions = (relu, sigmoid)
            # Make a neural network with 1 hidden layer of `h_size` units
            weights = (np.zeros([env.observation_space.shape[0] + 1, h_size]),
                       np.zeros([h_size + 1, env.action_space.shape[0]]))

        self.shape_list = weights_shape(weights)
        self.num_params = len(flatten_weights(weights))


    def __call__(self, state, theta):
        weights = unflatten_weights(theta, self.shape_list)

        return feed_forward(inputs=state,
                            weights=weights,
                            activation_functions=self.activation_functions)


def feed_forward(inputs, weights, activation_functions, verbose=False):
    x = inputs.copy()
    for layer_weights, layer_activation_fn in zip(weights, activation_functions):
        y = np.dot(x, layer_weights[1:])
        y += layer_weights[0]
        layer_output = layer_activation_fn(y)
        x = layer_output
    return layer_output


def weights_shape(weights):
    return [weights_array.shape for weights_array in weights]


def flatten_weights(weights):
    """Convert weight parameters to a 1 dimension array (more convenient for optimization algorithms)"""
    nested_list = [weights_2d_array.flatten().tolist() for weights_2d_array in weights]
    flat_list = list(itertools.chain(*nested_list))
    return flat_list


def unflatten_weights(flat_list, shape_list):
    """The reverse function of `flatten_weights`"""
    length_list = [shape[0] * shape[1] for shape in shape_list]

    nested_list = []
    start_index = 0

    for length, shape in zip(length_list, shape_list):
        nested_list.append(np.array(flat_list[start_index:start_index+length]).reshape(shape))
        start_index += length

    return nested_list

class ObjectiveFunction:

    def __init__(self, env, policy, num_episodes=1, max_time_steps=float('inf'), minimization_solver=True):
        self.ndim = policy.num_params  # Number of dimensions of the parameter (weights) space
        self.env = env
        self.policy = policy
        self.num_episodes = num_episodes
        self.max_time_steps = max_time_steps
        self.minimization_solver = minimization_solver

        self.num_evals = 0


    def eval(self, policy_params, num_episodes=None):
        """Evaluate a policy"""

        self.num_evals += 1

        if num_episodes is None:
            num_episodes = self.num_episodes

        average_total_rewards = 0

        for i_episode in range(num_episodes):

            total_rewards = 0.
            state = self.env.reset()
            done = False
            while not done:
                action = self.policy(state, policy_params)
                action = action >= 0.5
                state, reward, done, info = self.env.step(action)
                total_rewards += reward

            average_total_rewards += float(total_rewards) / num_episodes

        if self.minimization_solver:
            average_total_rewards *= -1.

        return average_total_rewards   # Optimizers do minimization by default...


    def __call__(self, policy_params, num_episodes=None):
        return self.eval(policy_params, num_episodes)


def saes(objective_function,
             x_array,
             sigma_array,
             max_iterations=500,
             tau=None,
             hist_dict=None):
    """
    x_array : shape (n,)
    sigma_array: shape (n,)
    """

    if tau is None:
        # Self-adaptation learning rate
        tau = 1./(2.* len(x_array))

    fx = objective_function(x_array)
    for i in range(max_iterations):
        sigma_array_ = sigma_array * np.exp(tau*np.random.normal(0,1,size=sigma_array.shape))
        x_array_ = x_array + sigma_array_ * np.random.normal(0,1,size=x_array.shape)
        fx_ = objective_function(x_array_)
        if fx_ < fx:
            fx = fx_
            x_array = x_array_
            sigma_array = sigma_array_
        if hist_dict is not None:
            hist_dict[i] = [fx] + x_array.tolist() + sigma_array.tolist()

    return x_array

if __name__ == "__main__":
    SEED = 42

    np.random.seed(SEED)

    env = DeepCure(foreign_countries = [ForeignCountry(0.1,100,100_000, save_history=True)], save_history=True, seed=SEED)
    env.reset()

    nn_policy = NeuralNetworkPolicy(env, one_layer=True)
    objective_function = ObjectiveFunction(env=env, policy=nn_policy, num_episodes=25)

    hist_dict = {}

    initial_solution_array = np.random.random(nn_policy.num_params)
    initial_sigma_array = np.ones(nn_policy.num_params) * 1.

    theta = saes(objective_function=objective_function,
                     x_array=initial_solution_array,
                     sigma_array=initial_sigma_array,
                     max_iterations=1000,
                     hist_dict=hist_dict)

    np.save('saes-theta.npy', theta)
    print(theta)

    np.random.seed(SEED)

    nn_policy2 = NeuralNetworkPolicy(env, h_size=10, one_layer=False)
    objective_function2 = ObjectiveFunction(env=env, policy=nn_policy2, num_episodes=25)

    hist_dict2 = {}

    initial_solution_array = np.random.random(nn_policy2.num_params)
    initial_sigma_array = np.ones(nn_policy2.num_params) * 1.

    theta2 = saes(objective_function=objective_function2,
                     x_array=initial_solution_array,
                     sigma_array=initial_sigma_array,
                     max_iterations=1000,
                     hist_dict=hist_dict2)

    np.save('saes-theta2.npy', theta2)
    print(theta2)

    rewards = pd.DataFrame.from_dict(hist_dict, orient='index').iloc[:,0].to_numpy()
    rewards2 = pd.DataFrame.from_dict(hist_dict2, orient='index').iloc[:,0].to_numpy()

    plt.figure()
    plt.plot(range(len(rewards)), rewards, label='1 layer')
    plt.plot(range(len(rewards2)), rewards2, label='2 layer')
    plt.xlabel("Training Steps")
    plt.ylabel("Reward")
    plt.legend()
    plt.show()
