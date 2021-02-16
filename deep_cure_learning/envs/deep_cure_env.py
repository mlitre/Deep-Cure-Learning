import gym
from gym import error, spaces, utils
from gym.utils import seeding

class ForeignCountry:
    """
    Environment simulation of a country not under the control of the agent
    """
    def __init__(self, border_traffic, number_of_infected, population, save_history=False):
        """
        border_traffic: percentage of infected that cross over to the agent's country
        number_of_infected: number of infected citizens of this foreign country
        population: total number of citizens of this foreign country
        save_history: If true, a history (for plotting) is saved
        """
        self.border_traffic = border_traffic
        self.number_of_infected = number_of_infected
        self.population = population
        self.save_history = save_history
        if self.save_history:
            self.hist_infected = []

    def step(self, infection_rate):
        """
        Simulate a time step. Computes the new number of infected in this foreign country
        infection_rate: the R-factor of the virus in this country
        """
        self.number_of_infected = min(self.population, infection_rate * self.number_of_infected)
        if self.save_history:
            self.hist_infected.append(self.number_of_infected)

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
    def __init__(self):
        pass

class DeepCure(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, save_history=False):
        self.save_history = save_history
        self.reset()

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
        self.take_action(action)
        # simulate environment
        for fcountry in self.f_countries:
            fcountry.step(self.v_base_infect_rate)

        # compute real R factor taking measures into account
        self.intern_infect_rate = self.v_base_infect_rate - sum([m.infection_rate_impact for i,m in enumerate(self.available_measures) if self.measures_active[i]])

        # compute infections caused by border traffic
        infected_from_foreign_countries = sum([c.border_traffic * c.number_of_infected for i,c in enumerate(self.f_countries) if self.borders_open[i]])

        # compute system state for new time step
        self.number_of_infected = min(self.population, self.intern_infect_rate * self.number_of_infected + infected_from_foreign_countries)
        self.number_of_dead = self.v_lethality * max(0, self.number_of_severe - self.hospital_capacity)
        self.number_of_severe = self.v_severity * self.number_of_infected

        self.reward = self.get_reward()

        if self.save_history:
            self.hist_internal_infection_rate.append(self.intern_infect_rate)
            self.hist_infected.append(self.number_of_infected)
            self.hist_severe.append(self.number_of_severe)
            self.hist_dead.append(self.number_of_dead)
            self.hist_reward.append(self.reward)
            self.hist_border.append(infected_from_foreign_countries)

        self.t += 1

        ob = Observation()
        return ob, self.reward, self.is_episode_over(), {}

    def reset(self):
        # the total number of citizens
        self.population = 1000
        # the number of infected citizens at the current time step
        self.number_of_infected = 1
        # the number of dead citizens at the current time step
        self.number_of_dead = 0
        # the number of severe cases of the illness
        self.number_of_severe = 0
        # the capacity of hospitals. Severe cases that aren't treated in the hospital may lead to death
        self.hospital_capacity = 100
        # the true R-factor in the agent's country
        self.intern_infect_rate = 0

        # Environment:
        # List of foreign countries
        self.f_countries = [ForeignCountry(0.01,10,1000, self.save_history)]

        # the virus's base R-factor
        self.v_base_infect_rate = 1.5
        # the percentage of severe cases in case of infection
        # E.g. if v_severity=0.5, then 50% of all infections are severe cases
        self.v_severity = 0.5
        # the percentage of lethal outcome in case of severe case without hospitalization
        # E.g. if v_severity=0.5, then 50% of non-hospitalized severe cases lead to death
        self.v_lethality = 0.5

        # current time step
        self.t = 0
        # the current score
        self.reward = 0
        # score weighting for the number of dead citizens
        self.weight_dead = 1
        # score weighting for the number of severe cases
        self.weight_severe = 0.5

        # Measures:
        # If borders_open[i] is true, then border traffic from foreign country i is allowed
        self.borders_open = [True] * len(self.f_countries)
        # A list of available measures
        self.available_measures = [Measure('Masks', 0.1, 0.1), Measure('Curfew', 0.5, 0.5)]
        # If measures_active[i] is true, then measure available_measures[i] is active
        self.measures_active = [False] * len(self.available_measures)

        if self.save_history:
            self.hist_internal_infection_rate = []
            self.hist_infected = []
            self.hist_severe = []
            self.hist_dead = []
            self.hist_reward = []
            self.hist_border = []

    def render(self, mode='human', close=False):
        pass

    def take_action(self, action):
        """
        action: [ masks, curfew, borders]
        """
        self.measures_active[0] = action[0]
        self.measures_active[1] = action[1]
        assert(len(action[2]) == len(self.f_countries))
        self.borders_open = action[2]

    def get_reward(self):
        """ Penality for active measures, number of severe cases and number of dead """
        return -sum(m.score_impact for i,m in enumerate(self.available_measures) if self.measures_active[i]) - self.weight_severe * self.number_of_severe - self.weight_dead * self.number_of_dead

    def is_episode_over(self):
        """
        The episode is over if
        - the whole population dies
        """
        return self.number_of_dead >= self.population
