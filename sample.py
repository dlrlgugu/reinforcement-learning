import gym

env = gym.make("Taxi-v2")
state = env.reset()
for _ in range(1000):
    env.render()
    action = env.action_space.sample()#take random action now
    state , reward , done , info = env.step(action)
