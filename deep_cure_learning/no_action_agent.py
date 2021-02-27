from envs.deep_cure_env import DeepCure
from plotting import plot
import gym

env = DeepCure(save_history=True)
env.reset()

end = False
t = 0
while not end and t <= 20:
    action = [False, False, [True] * 1]
    if t >= 5:
        action[0] = True
        action[1] = True
    obs, reward, end, _ = env.step(action)
    t += 1

plot(env)
