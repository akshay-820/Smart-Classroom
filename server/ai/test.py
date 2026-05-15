from sim.classroom_env import SmartClassroomEnv

env = SmartClassroomEnv()

state = env.reset()

print("Initial State:", state)

for i in range(5):

    action = i % 4

    next_state, reward, done = env.step(action)

    print(next_state, reward)