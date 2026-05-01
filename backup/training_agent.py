from drl_agent import DRLAgent
from drl_agent import RDPEnv

class TrainingAgent:

    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

    def train(self, episodes=1000):
        for episode in range(episodes):
            state = self.env.reset()
            done = False

            while not done:
                action = self.agent.get_action(state)
                next_state, reward, done = self.env.step(action)
                self.agent.update_q_value(state, action, reward, next_state)
                state = next_state

if __name__ == "__main__":
    servers = [
        {"cpu": 20, "active_sessions": 1, "max_sessions": 10},
        {"cpu": 30, "active_sessions": 2, "max_sessions": 15},
        {"cpu": 25, "active_sessions": 3, "max_sessions": 12},
    ]
    env = RDPEnv(servers)
    agent = DRLAgent(action_size=len(servers))
    trainer = TrainingAgent(env, agent)
    trainer.train(episodes=500)