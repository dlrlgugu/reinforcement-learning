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


register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name':'4x4','is_slippery':False}
    )

env=gym.make('FrozenLake-v3')
env.render()

while True:
    key=readchar.readkey()
    if key not in arrow_keys.keys():
        print("nah")
        break

    action = arrow_keys[key]
    state , reward , done , info = env.step(action)
    env.render()#render after action.
    print("state: ",state,"Action: ",action,"Reward: ",reward,
            "Info: ",info)

    if done:
        print("done. reward is : ",reward)
        break
    
            



















