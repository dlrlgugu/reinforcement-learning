


Q=np.zeros([env.observation_spcae.n,env.action_space.n])

episodes=2000

rList=[]

for i in range(episodes):
    state = env.reset()
    rAll=0
    done=False

    while not done:
        action = rargmax(Q[state,:])
        new_state , reward,done,_=env.step(action)
        Q[state,action]=reward+np.max(Q[new_state,:])
        #update Q
        state=new_state


                                
