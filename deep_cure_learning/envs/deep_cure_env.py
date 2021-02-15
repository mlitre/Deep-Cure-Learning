import gym 
from gym import error, spaces, utils
from gym.utils import seeding

class DeepCure(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.intern_infect_rate = 0
        self.hospital_count = 0
        self.morale_score = 0
        self.v_base_infect_rate = 0
        self.v_severity = 0
        self.v_lethality = 0
        self.masks = 0
        self.curfew = 0
        self.borders_open = 0
        self.impact_masks = 0
        self.impact_curfew = 0
        self.impact_borders_open = 0
		self.counter = 0
		self.reward = 0
        self.f_countries = []

    def _step(self, action):
        """

        Parameters
        ----------
        action :

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
        self._take_action(action)
        for i, (traffic, nb_infected) in enumerate(self.f_countries):
            self.f_countries[i][1] = nb_infected * self.v_base_infect_rate

        self.intern_infect_rate = self.v_base_infect_rate - self.impact_borders_open
        internal infection rate = base infection rate - measures
        number of infected = internal infection rate * old number of infected + (border traffic * percentage infected people in the foreign country if border is not closed)
        number hospitalizations = number infected * severity
        number dead = (number hospitalizations - number available places) + remaining * lethality

    score =  measures + dead + hospitalizations

        reward = self._get_reward()
        ob = self.env.getState()
        episode_over = self.status != hfo_py.IN_GAME
        return ob, reward, episode_over, {}

    def _reset(self):
        pass

    def _render(self, mode='human', close=False):
        pass

    def _take_action(self, action):
        """
        action: [ masks, curfew, borders]
        """
        self.masks = action[0]
        self.curfew = action[1]
        self.borders_open = action[2]

    def _get_reward(self):
        """ Reward is given for XY. """
        if self.status == FOOBAR:
            return 1
        elif self.status == ABC:
            return self.somestate ** 2
        else:
            return 0