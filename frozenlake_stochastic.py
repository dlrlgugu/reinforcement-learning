import gym
from gym.envs.registration import register

import readchar

LEFT=0
DOWN=1
RIGHT=2
UP=3

arrow_keys={
    '\x1b[A':UP,
    '\x1b[B':DOWN,
    '\x1b[C':RIGHT,
    'x1b[0':LEFT
    }

env=gym.make('FrozenLake-v0')
env.render()

while True:
    state=env.reset()
    key=readchar.readkey()
    if key not in arrow_keys.keys():
        print("nah")
        break

    action = arrow_keys[key]
    new_state , reward , done , info = env.step(action)
    env.render()#render after action.
    print("state: ",new_state,"Action: ",action,"Reward: ",reward,
            "Info: ",info)

    if done:
        print("done. reward is : ",reward)
        break
    
            



















