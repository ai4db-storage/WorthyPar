# encoding: utf-8
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
import copy
from env_singledb import ENV

predicted_query1 = [[[[7, 11], [7, 17], [7, 16]], [1, 2, 3, 5, 6, 7], [[1, [2, [1, 2, 4, 5, 7, 8, 9, 10]]], [1, [9, [1, 2, 4, 5, 7, 8, 9, 10]]], [1, [11, [1, 2, 4, 5, 7, 8, 9, 10]]], [2, [5, [1, 4, 5, 6, 7, 8, 9, 10]]], [2, [7, [1, 4, 5, 6, 7, 8, 9, 10]]], [2, [8, [1, 4, 5, 6, 7, 8, 9, 10]]], [3, [1, [1, 2, 4, 6, 7, 8, 9]]], [3, [3, [1, 2, 4, 6, 7, 8, 9]]], [5, [11, [4]]], [6, [2, [6]]], [6, [11, [6]]], [7, [1, [1, 3, 5, 8, 9, 10]]], [7, [5, [1, 3, 5, 8, 9, 10]]], [7, [6, [1, 3, 5, 8, 9, 10]]], [7, [7, [1, 3, 5, 8, 9, 10]]], [7, [8, [1, 3, 5, 8, 9, 10]]], [7, [14, [1, 3, 5, 8, 9, 10]]], [7, [23, [1, 3, 5, 8, 9, 10]]]]], 0.4]
predicted_query2 = [[[[7, 11]], [1, 2, 5, 6, 7], [[1, [2, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]], [1, [9, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]], [1, [11, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]], [2, [5, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]], [2, [7, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]], [2, [8, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]], [5, [11, [4]]], [6, [2, [5]]], [6, [11, [5]]], [7, [1, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]], [7, [5, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]], [7, [7, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]], [7, [8, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]], [7, [14, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]], [7, [23, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]]]], 0.3]
predicted_query3 = [[[], [3, 4, 5, 7], [[3, [1, [1, 3, 5, 6, 8, 10]]], [3, [3, [1, 3, 5, 6, 8, 10]]], [4, [1, [4]]], [4, [3, [4]]], [4, [4, [4]]], [5, [11, [4]]], [5, [24, [4]]], [7, [2, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]], [7, [6, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]], [7, [8, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]]]], 0.3]
list_db = [[10, 13], [10, 9], [10, 5], [10, 10], [10, 29], [10, 28], [10, 23]]
actua_db = [383000, 1920800, 7200, 86400, 144, 73049, 144004764]
predicted_qurry = [predicted_query1, predicted_query2, predicted_query3]

temp_env = ENV(list_db, predicted_qurry, actua_db)
temp_env.reset()


# Fixed Q-target 网络模型
class DQN():
    def __init__(self,
                 num,
                 dim_state,
                 n_actions,
                 col_num,
                 batch_size=32,
                 learning_rate=0.04,
                 epsilon=0.5,
                 down_epsilon=0.5,
                 gamma=0.9,
                 target_replace_iter=30,
                 memory_size=500, ):
        self.num = num
        self.eval_net, self.target_net = self.bulid_Net(dim_state, n_actions), self.bulid_Net(dim_state, n_actions)
        self.list_db = list_db
        self.dim_state = dim_state  # state dimension
        self.n_actions = n_actions  # action dimension
        self.col_num = col_num      # horizontal partition dimension(10)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon            # greed coefficient
        self.down_epsilon = down_epsilon  # descent rate of greed coefficient
        self.gamma = gamma                # decay rate of reward
        self.memory_size = memory_size
        self.taget_replace_iter = target_replace_iter  # targetNET Updated interval steps
        self.learn_step_counter = 0  # Used to calculate updates after n steps
        self.memory_counter = 0  # Used to calculate memory index
        self.memory = np.zeros((self.memory_size, self.dim_state * 2 + self.n_actions + 1 + self.col_num - 10 ))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.learning_rate)
        self.loss_func = nn.MSELoss()

    def choose_action(self, s, actua_state, actions):
        # 1-greedy  Select actions via evalNET
        # greedy    Select actions randomly
        if np.random.uniform() > self.epsilon:
            actions_value = []
            for action in actions:
                temp = s.copy()
                temp.extend(actua_state)
                temp.extend(action)
                x = torch.unsqueeze(torch.FloatTensor(temp), 0)
                actions_value.append(self.eval_net.forward(x).detach().numpy()[0])
            i = np.argmax(actions_value)
        else:
            i = np.random.randint(0, len(actions))
        return actions[i]

    def learn(self):
        # update targetNET
        if self.learn_step_counter % self.taget_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            self.epsilon = self.epsilon * self.down_epsilon
        self.learn_step_counter += 1

        # Retrieves part of the data in the memory of the memory bank
        # data_size = MIN (memory_counter, batch_size)
        data_size = self.memory_size if self.memory_counter > self.memory_size else self.memory_counter
        sample_index = np.random.choice(data_size, self.batch_size)

        # memory_item = [state, actua_state1, action, actual_state2, reward, next_state]
        b_memory = self.memory[sample_index, :]
        b_s_a = torch.FloatTensor(b_memory[:, :self.dim_state + self.n_actions])
        b_a_s = torch.FloatTensor(
            b_memory[:, self.dim_state + self.n_actions:self.dim_state + self.n_actions + self.col_num])
        b_r = torch.FloatTensor(b_memory[:,
                                self.dim_state + self.n_actions + self.col_num:self.dim_state + self.n_actions + self.col_num + 1])
        s_ = b_memory[:, -(self.dim_state - 10):]
        torch.FloatTensor(s_)

        q_eval = self.eval_net(b_s_a)
        q_next = []
        for i in range(self.batch_size):
            temp_env.set_state(self.num, list(s_[i]))
            temp_env.set_actua_state(self.num, list(b_a_s[i]))
            actions = temp_env.get_actions(self.num)
            action_value = []
            for action in actions:
                temp = list(s_[i]).copy()
                temp.extend(list(b_a_s[i]))
                temp.extend(action)
                x = torch.unsqueeze(torch.FloatTensor(temp), 0)
                action_value.append(self.target_net.forward(x).detach().numpy()[0])
            q_next.append(max(action_value))
        q_next = torch.FloatTensor(np.array(q_next))
        q_target = b_r + self.gamma * q_next.view(self.batch_size, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.cpu().detach().numpy()

    def store_transition(self, s, a, r, s_, a_s, a_s_):
        temp = s.copy()
        temp.extend(a_s)
        temp.extend(a)
        transition = np.hstack((temp, a_s_, r, s_))
        # store memory (overwrite the memory if the first round is full)
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def bulid_Net(self, dim_state, n_actions):
        return torch.nn.Sequential(
            torch.nn.Linear(dim_state + n_actions, 150),
            torch.nn.ReLU(),
            torch.nn.Linear(150, 1),
        )

    def save_model(self, path):
        torch.save(self.eval_net.state_dict(), path)


if __name__ == '__main__':
    env = ENV(list_db, predicted_qurry, actua_db)
    dqn = []
    best_dqn = []
    for i in range(len(list_db)):
        dqn.append(DQN(i + 1, env.get_n_state(i + 1), env.get_n_action(i + 1), list_db[i][0]))
        best_dqn.append(DQN(i + 1, env.get_n_state(i + 1), env.get_n_action(i + 1), list_db[i][0]))
    best_reward = 0
    # Draw a line chart (x: i_episode y: ep_r)
    i_episode_list = []
    ep_r_list = []
    loss_list = []
    print('Collecting experience...')
    for i_episode in range(200):
        s, a_s = env.reset()
        ep_r = 0
        loss = 0
        temp = True
        while True:
            done_all = False
            for i in range(len(list_db)):
                actions = env.get_actions(i + 1)            # 生成可能的动作列表
                a = dqn[i].choose_action(s[i], a_s[i], actions)     # 选择动作
                s_, r, done = env.step(i + 1, a)            # 根据选择的动作改变状态，获取下一状态s_，以及该动作奖赏r，以及是否完成标志done
                a_s_ = env.get_actua_state(i + 1)            # 获取当前表的实际水平分区状态(即水平分区的布局情况)
                if temp:
                    temp = False
                else:
                    dqn[i].store_transition(s[i], a, r, s_, a_s[i], a_s_)  # store information
                s[i] = s_
                a_s[i] = a_s_
                ep_r += r
                if done:
                    done_all = True
            for i in range(len(list_db)):
                if dqn[i].memory_counter > dqn[i].memory_size:
                    loss += dqn[i].learn()
                    loss_list.append(loss)
            if done_all:
                if i_episode % 10 == 0:
                    print('Ep: ', i_episode + 1, '| Ep_r: ', round(ep_r, 2))
                i_episode_list.append(i_episode + 1)
                ep_r_list.append(round(ep_r, 2))
                if ep_r > best_reward:
                    best_reward = ep_r
                    for i in range(len(list_db)):
                        best_dqn[i] = copy.deepcopy(dqn[i])
                break
    print('Training over')
    plt.plot(i_episode_list, ep_r_list)
    plt.show()
    for i in range(len(ep_r_list)):
        print('Ep: ', i_episode_list[i], '| Ep_r: ', round(ep_r_list[i], 2))
    # Test
    print('Testing . . .')
    s, a_s = env.reset()
    ep_r = 0
    temp_actions = []
    for i in range(len(list_db)):
        temp_actions.append([])
    while True:
        done_all = False
        for i in range(len(list_db)):
            actions = env.get_actions(i + 1)
            a = best_dqn[i].choose_action(s[i], a_s[i], actions)
            temp_actions[i].append(a)
            s_, r, done = env.step(i + 1, a)
            a_s_ = env.get_actua_state(i + 1)
            ep_r += r
            s[i] = s_
            a_s[i] = a_s_
            if done:
                done_all = True
        if done_all:
            print(str(ep_r) + "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            for i in range(len(temp_actions)):
                print(i)
                for j in range(len(temp_actions[i])):
                    print(temp_actions[i][j])
            break

