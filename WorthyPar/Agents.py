import torch
import torch.nn as nn
import numpy as np
import os
from matplotlib import pyplot as plt
from WorthyPar.Env import ENV

class DQN():
    def __init__(self,
                 index,
                 state_num,
                 action_num,
                 info,
                 batch_size=32,
                 learning_rate=0.05,
                 epsilon=1.0,
                 epsilon_min=0.02,
                 down_epsilon=0.99,
                 gamma=0.9,
                 target_replace_iter=60,
                 memory_size=500, ):
        self.index = index
        self.state_num = state_num
        self.action_num = action_num
        self.eval_net =  self.build_net()
        self.target_net = self.build_net()
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.down_epsilon = down_epsilon
        self.gamma = gamma
        self.target_replace_iter = target_replace_iter
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory_size = memory_size
        self.memory = self.build_memory()
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.learning_rate)
        self.loss_func = nn.MSELoss()
        self.temp_env = ENV(info)
        self.temp_env.reset()

    def build_net(self):
        return torch.nn.Sequential(
            torch.nn.Linear(self.state_num + self.action_num, 250),
            torch.nn.ReLU(),
            torch.nn.Linear(250, 1),
        )

    def build_memory(self):
        return np.zeros((self.memory_size, self.state_num * 2 + self.action_num + 1))

    def choose_action(self, state, actions):
        # 1-greedy  select actions via evalNET
        # greedy    random selection action
        if np.random.uniform() > self.epsilon:
            actions_value = []
            for action in actions:
                temp = state.copy()
                temp.extend(action)
                x = torch.unsqueeze(torch.FloatTensor(temp), 0)
                actions_value.append(self.eval_net.forward(x).detach().numpy()[0])
            i = np.argmax(actions_value)
        else:
            i = np.random.randint(0, len(actions))
        self.epsilon = max(self.epsilon_min, self.epsilon * self.down_epsilon)
        return actions[i]

    def store_transition(self, state, action, reward, state_next):
        transition = np.hstack((state, action, reward, state_next))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % self.target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            self.epsilon = self.epsilon * self.down_epsilon
        self.learn_step_counter += 1

        data_size = self.memory_size if self.memory_counter > self.memory_size else self.memory_counter
        sample_index = np.random.choice(data_size, self.batch_size)

        # memory_item = [state, action, reward, state_next]
        b_memory = self.memory[sample_index, :]
        b_s_a = torch.FloatTensor(b_memory[:, :self.state_num + self.action_num])
        b_r = torch.FloatTensor(b_memory[:,self.state_num + self.action_num:
                                           self.state_num + self.action_num + 1])
        s_ = b_memory[:, -self.state_num:]
        # 获取 q_eval, q_target
        q_eval = self.eval_net(b_s_a)
        q_next = []
        for i in range(self.batch_size):
            self.temp_env.set_state(self.index, list(s_[i]))
            actions = self.temp_env.get_actions(self.index)
            action_value = []
            for action in actions:
                temp = list(s_[i]).copy()
                temp.extend(action)
                x = torch.unsqueeze(torch.FloatTensor(temp), 0)
                action_value.append(self.target_net.forward(x).detach().numpy()[0])
            q_next.append(max(action_value))
        q_next = torch.FloatTensor(np.array(q_next))
        q_target = b_r + self.gamma * q_next.view(self.batch_size, 1)
        loss = self.loss_func(q_eval, q_target)
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.cpu().detach().numpy()

    def save_model(self, path):
        torch.save(self.eval_net.state_dict(), path)

    def load_model(self, path):
        self.eval_net.load_state_dict(torch.load(path))



def train(info):
    env = ENV(info)
    agents = []
    best_reward = -1
    root_path = os.getcwd()
    relative_path = "model"
    for i in range(len(info['db_schema'])):
        agents.append(DQN(i, env.get_state_num()[i], env.get_action_num()[i], info))
    episode_list = []
    ep_r_list = []
    loss_list = []
    for i_episode in range(500):
        state = env.reset()
        ep_r = 0
        loss = 0
        action_list = [[] for _ in range(len(info['db_schema']))]
        while True:
            for i in range(len(info['db_schema'])):
                actions = env.get_actions(i)
                action = agents[i].choose_action(state[i], actions)
                action_list[i].append(action)
                state_next, reward, done = env.step(i, action)
                agents[i].store_transition(state[i], action, reward, state_next)
                state[i] = state_next
                ep_r += reward
            for i in range(len(info['db_schema'])):
                if agents[i].memory_counter > agents[i].memory_size:
                    loss += agents[i].learn()
            if done:
                if i_episode % 10 == 0:
                    print('Ep: ', i_episode + 1, '| Ep_r: ', round(ep_r, 2))
                episode_list.append(i_episode + 1)
                ep_r_list.append(round(ep_r, 2))
                loss_list.append(loss)
                if ep_r > best_reward:
                    best_reward = ep_r
                    for agent in agents:
                        model_name = "model" + str(agent.index) + ".pt"
                        path = os.path.join(root_path, relative_path, model_name)
                        agent.save_model(path)
                break
    plt.plot(episode_list, ep_r_list)
    plt.savefig('Training_result_reward')


def generate_result(info):
    env = ENV(info)
    state = env.reset()
    action_list = [[] for _ in range(len(info['db_schema']))]
    agents = []
    root_path = os.path.dirname(os.getcwd())
    relative_path = "model"
    for i in range(len(info['db_schema'])):
        model_name = "model" + str(i) + ".pt"
        path = os.path.join(root_path, relative_path, model_name)
        agent = DQN(i, env.get_state_num()[i], env.get_action_num()[i], info, epsilon=0, epsilon_min=0)
        agent.load_model(path)
        agents.append(agent)

    while True:
        for i in range(len(info['db_schema'])):
            actions = env.get_actions(i)
            action = agents[i].choose_action(state[i], actions)
            action_list[i].append(action)
            state_next, reward, done = env.step(i, action)
            state[i] = state_next
        if done:
            for i in range(len(action_list)):
                print("table", i)
                for j in range(len(action_list[i])):
                    print(action_list[i][j])
            break
    return env.show_result()

def get_result(Info):
    num = float(len(Info['db_query']))
    for query_name, query_code in Info['db_query'].items():
        query_code.append(1 / num)
    return generate_result(Info)

