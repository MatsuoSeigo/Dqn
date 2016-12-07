import parser
from agent import Agent
from enviroment import Enviroment

NUM_EPISODES = 10000

def main():
    args = parser.parse()

    env = Enviroment(args.environment, args.display)
    agent = Agent(env_name=args.environment, num_actions=env.gym_env.action_space.n)

    if args.test: #TEST MODE
        agent.restore_network()
        for episode in range(NUM_EPISODES):
            state = env.reset()
            episode_end = False

            while not episode_end:
                action = agent.select_action_test(state)
                state, _, _ = env.step(action)

    else: #TRAIN MODE
        for episode in range(NUM_EPISODES):
            state = env.reset()
            episode_end = False

            while not episode_end:
                action = agent.select_action(state)
                state, reward, episode_end = env.step(action)
                agent.set(state, action, reward, episode_end)

if __name__ == '__main__':
    main()
