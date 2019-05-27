import random
from environment import Environment


# エージェントを操作
class Agent():

    def __init__(self, env):
        self.actions = env.actions

     # 戦略(Day1では行動をランダムに選ばせる)
    def policy(self, state):
        return random.choice(self.actions)


def main():
    # Make grid environment.
    grid = [
        [0, 0, 0, 1],
        [0, 9, 0, -1],
        [0, 0, 0, 0]
    ]
    env = Environment(grid)
    agent = Agent(env)

    # Try 10 game.
    for i in range(10):
        # Initialize position of agent.
        
        # 位置の初期化
        state = env.reset()
        
        # その他必要項目初期化 
        total_reward = 0
        done = False

        while not done:
            action = agent.policy(state)
            next_state, reward, done = env.step(action)
            total_reward += reward
            state = next_state

        print("Episode {}: Agent gets {} reward.".format(i, total_reward))


if __name__ == "__main__":
    main()
