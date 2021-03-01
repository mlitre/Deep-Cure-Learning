import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import math

def infection_curve(rate, population, number_of_infected):
    """
    Computes the new number of infected given current rate, total population and
    current number of infected
    """
    return population/(1+(population/number_of_infected-1)/np.exp(rate))

class ForeignCountry:
    """
    Environment simulation of a country not under the control of the agent
    """
    def __init__(self, border_traffic, number_of_infected, population, save_history=False):
        """
        border_traffic: number of people that cross over to the agent's country
        number_of_infected: number of infected citizens of this foreign country
        population: total number of citizens of this foreign country
        save_history: If true, a history (for plotting) is saved
        """
        self.border_traffic = border_traffic
        self.initial_number_of_infected = number_of_infected
        self.population = population
        self.save_history = save_history
        self.reset()


    def step(self, infection_rate):
        """
        Simulate a time step. Computes the new number of infected in this foreign country
        infection_rate: the R-factor of the virus in this country
        """
        old = self.number_of_infected
        self.number_of_infected += self.new_number_of_infected * infection_rate * ((self.population-self.number_of_infected)/self.population)
        self.new_number_of_infected = self.number_of_infected - old
        if self.save_history:
            self.hist_infected.append(self.number_of_infected)

    def reset(self):
        self.number_of_infected = self.initial_number_of_infected
        self.new_number_of_infected = self.number_of_infected
        if self.save_history:
            self.hist_infected = []

class Measure:
    """
    A measure that (if active) reduces the R-factor by infection_rate_impact but
    decreases the score by score_impact
    """
    def __init__(self, name, infection_rate_impact, score_impact):
        """
        name: The name of the measure
        infection_rate_impact: subtracted from the R-factor if the measure is active
        score_impact: subtracted from the score if the measure is active (i.e. a penalty)
        """
        self.name = name
        self.infection_rate_impact = infection_rate_impact
        self.score_impact = score_impact

class Observation:
    def __init__(self, old_new_number_of_infected, new_number_of_infected):
        self.state0 = old_new_number_of_infected
        self.state1 = new_number_of_infected
        # self.state = min(int(new_infected / stepsize), env.state_space_size() - 1) # state 5 = 100 < n_infected < 120
        # self.state_prev = min(int(old_infected / stepsize), env.state_space_size() - 1)

def random_base_infect_rate():
    return 1 + 2 * np.random.rand()

def random_lifetime():
    return np.random.randint(30,101)

class DeepCure(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, foreign_countries, v_base_infect_rate = None, v_lifetime = None, save_history=False):
        self.save_history = save_history
        self.f_countries = foreign_countries
        self.reset(v_base_infect_rate, v_lifetime)

    def state_space_size(self):
        return int(self.population / self.stepsize)

    def step(self, action):
        """

        Parameters
        ----------
        action : [mask: boolean, curfew: boolean, border_open: list of boolean]
        Indicate the action to take

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        # apply action
        old_infected = self.number_of_infected
        old_severe = self.number_of_severe
        old_dead = self.number_of_dead

        self.take_action(action)
        # simulate environment
        for fcountry in self.f_countries:
            fcountry.step(1)

        # compute real R factor taking measures into account
        self.intern_infect_rate = max(0,self.v_base_infect_rate - sum([m.infection_rate_impact for i,m in enumerate(self.available_measures) if self.measures_active[i]]))

        # compute infections caused by border traffic
        infected_from_foreign_countries = sum([(1 + np.sin(self.t*np.pi/20)) * c.border_traffic for i,c in enumerate(self.f_countries) if self.borders_open[i]])

        # compute system state for new time step
        potential_infected_ratio = ((self.population-self.number_of_infected)/self.population) * (self.curfew_impact if self.is_curfew_active else 1)
        self.number_of_infected += self.new_number_of_infected * self.intern_infect_rate * potential_infected_ratio
        self.number_of_infected = min(self.number_of_infected + infected_from_foreign_countries, self.population)

        # min(self.population, infection_curve(self.intern_infect_rate, self.population, self.number_of_infected)+ infected_from_foreign_countries)
        self.number_of_dead += self.v_lethality * max(0, self.new_number_of_severe - self.hospital_capacity) + 0.1 * self.v_lethality * min(self.new_number_of_severe, self.hospital_capacity)
        self.number_of_severe = self.v_severity * old_infected#self.number_of_infected

        self.new_number_of_infected = self.number_of_infected - old_infected
        self.new_number_of_severe = self.number_of_severe - old_severe
        self.new_number_of_dead = self.number_of_dead - old_dead

        self.reward = self.get_reward()

        if self.save_history:
            self.hist_internal_infection_rate.append(self.intern_infect_rate)
            self.hist_infected.append(self.number_of_infected)
            self.hist_severe.append(self.number_of_severe)
            self.hist_dead.append(self.number_of_dead)
            self.hist_reward.append(self.reward)
            self.hist_border.append(infected_from_foreign_countries)
            self.hist_new_infected.append(self.new_number_of_infected)
            self.hist_new_severe.append(self.new_number_of_severe)
            self.hist_new_dead.append(self.new_number_of_dead)
            self.hist_action.append(action)

        self.t += 1

        # ob = Observation(old_new_number_of_infected, self.new_number_of_infected)
        return np.array([self.new_number_of_severe, self.new_number_of_dead, infected_from_foreign_countries]), self.reward, self.is_episode_over(), {}

    def reset(self, rate = None, lifetime = None):
        # the total number of citizens
        self.population = 100_000
        # the number of infected citizens at the current time step
        self.number_of_infected = 100
        self.new_number_of_infected = self.number_of_infected
        # the number of dead citizens at the current time step
        self.number_of_dead = 0
        self.new_number_of_dead = 0
        # the number of severe cases of the illness
        self.number_of_severe = 0
        self.new_number_of_severe = 0
        # the capacity of hospitals. Severe cases that aren't treated in the hospital may lead to death
        self.hospital_capacity = 0.005 * self.population
        # the true R-factor in the agent's country
        self.intern_infect_rate = 0

        # Environment:
        # List of foreign countries
        for fcountry in self.f_countries:
            fcountry.reset()
        # self.f_countries = self.init_foreign_countries

        # the virus's base R-factor
        if rate is None:
            self.v_base_infect_rate = random_base_infect_rate()
        else:
            self.v_base_infect_rate = rate
        # the percentage of severe cases in case of infection
        # E.g. if v_severity=0.5, then 50% of all infections are severe cases
        self.v_severity = 0.2
        # the percentage of lethal outcome in case of severe case without hospitalization
        # E.g. if v_severity=0.5, then 50% of non-hospitalized severe cases lead to death
        self.v_lethality = 0.5

        if lifetime is None:
            self.v_lifetime = random_lifetime()
        else:
            self.v_lifetime = lifetime

        # current time step
        self.t = 0
        # the current score
        self.reward = 0
        # score weighting for the number of dead citizens
        self.weight_dead = 10
        # score weighting for the number of severe cases
        self.weight_severe = 2

        self.weight_infected = 0.1

        # Measures:
        # If borders_open[i] is true, then border traffic from foreign country i is allowed
        self.borders_open = [True] * len(self.f_countries)
        # A list of available measures
        self.available_measures = [Measure('Masks', infection_rate_impact = 0.25, score_impact = 0.002 * self.population)] #, Measure('Curfew', 1, )]
        self.curfew_impact = 0.2
        self.is_curfew_active = False
        self.curfew_reward = 0.015 * self.population
        # If measures_active[i] is true, then measure available_measures[i] is active
        self.measures_active = [False] * len(self.available_measures)

        if self.save_history:
            self.hist_internal_infection_rate = []
            self.hist_infected = []
            self.hist_severe = []
            self.hist_dead = []
            self.hist_reward = []
            self.hist_border = []
            self.hist_new_infected = []
            self.hist_new_severe = []
            self.hist_new_dead = []
            self.hist_action = []

        # return Observation(0, self.new_number_of_infected)
        return np.array([0, 0, 0])

    def render(self, mode='human', close=False):
        pass

    def take_action(self, action):
        """
        action: [ masks, curfew, borders]
        """
        # print(f'Action taken = {action}')
        self.measures_active[0] = action[0] #action[0]
        self.is_curfew_active = action[1]
        self.borders_open[0] = action[2]
        # if action == 0:
        #     self.measures_active[0] = False
        #     self.is_curfew_active = False
        # elif action == 1:
        #     self.measures_active[0] = True
        #     self.is_curfew_active = False
        # elif action == 2:
        #     self.measures_active[0] = False
        #     self.is_curfew_active = True
        # elif action == 3:
        #     self.measures_active[0] = True
        #     self.is_curfew_active = True
        # self.measures_active[0] = action[0]
        # self.measures_active[1] = action[1]
        # assert(len(action[2]) == len(self.f_countries))

    def get_reward(self):
        """ Penality for active measures, number of severe cases and number of dead """
        reward = -sum(m.score_impact for i,m in enumerate(self.available_measures) if self.measures_active[i])
        reward -= self.weight_severe * self.new_number_of_severe
        reward -= self.weight_dead * self.new_number_of_dead
        if self.is_curfew_active:
            reward -= self.curfew_reward
        # reward -= self.weight_infected * (self.new_number_of_infected - self.new_number_of_severe)
        return reward / self.population / self.v_lifetime

    def is_episode_over(self):
        """
        The episode is over after 15 timesteps
        """
        return self.t >= self.v_lifetime
