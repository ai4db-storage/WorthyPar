import torch
import torch.nn as nn
import numpy as np


class DRL_H():
    def __init__(self,info,index):
        self.index = index
        self.state_num = info['db_schema'][index][1]
        self.action_num = info['db_schema'][index][1]
        self.eval_net =  self.build_net()
        self.target_net = self.build_net()
        self.batch_size = 32
        self.learning_rate = 0.05
        self.epsilon = 1.0
        self.epsilon_min = 0.02
        self.down_epsilon = 0.99
        self.gamma = 0.9
        self.target_replace_iter = 60
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory_size = 500
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
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.cpu().detach().numpy()

    def save_model(self, path):
        torch.save(self.eval_net.state_dict(), path)

    def load_model(self, path):
        self.eval_net.load_state_dict(torch.load(path))


class ENV(object):
    def __init__(self, info):
        self.max_step = 10
        self.partition_num = 3
        self.db_schema = info['db_schema']
        self.db_size = info['db_size']
        self.db_query = info['db_query']
        self.state = None
        self.state_num = self.get_state_num()
        self.action_num = self.get_action_num()
        self.done = False
        self.step_num = 0
        self.MAX_STEP = self.max_step * len(self.db_schema)
        self.cost = []
        self.attribute = []

    def set_state(self, label, state):
        self.state[label-1] = state

    def get_state_num(self):
        state_num = []
        for i in range(len(self.db_schema)):
            state_num.append(self.get_attribute_num(i))
        return state_num

    def get_action_num(self):
        action_num = []
        for i in range(len(self.db_schema)):
            action_num.append(self.get_attribute_num(i))
        return action_num

    def get_attribute_num(self, index):
        return self.db_schema[index][1]

    def reset(self):
        self.step_num = 0
        self.done = False
        self.cost = [i for i in self.db_size]
        self.row_block = [[] for _ in range(len(self.db_size))]
        state = [[] for _ in range(len(self.db_schema))]
        for i in range(len(self.db_schema)):
            db = self.db_schema[i]
            for j in range(db[1]):
                state[i].append(0)
            for query in self.db_query.values():
                state[i].append(query[3])
        self.state = state
        return state

    def get_actions(self, index):
        actions = []
        attribute_num = self.get_attribute_num(index)
        action = [0 for _ in range(attribute_num)]
        if self.step_num >= self.MAX_STEP*0.2:
            actions.append(action.copy())
        for i in range(attribute_num):
            action[i] = 1
            actions.append(action.copy())
            action[i] = 0
        return actions

    def step(self, index, action):
        state_next = self.take_action(index, action)
        if sum(action) == 0:
            reward = 0
        else:
            self.state[index] = state_next
            reward = self.reward(index)
        self.step_num += 1
        if self.step_num == self.MAX_STEP:
            self.done = True
        return state_next, reward, self.done

    def take_action(self, index, action):
        state_next = self.state[index]
        if sum(action) != 0:
            i = action.index(1)
            state_next[i] = 1
            self.attribute.append(i)
        return state_next

    def reward(self, index):
        cost = self.calculate_cost(index)
        if cost == 0:
            reward = 0
        else:
            reward = -1 + self.cost[index] / cost
            self.cost[index] = cost
        return reward

    def calculate_cost(self, index):
        cost = 0
        for query in self.db_query.values():
            if index in query[1]:
                loc = []
                for column in query[2]:
                    if column[0] == index:
                        loc = column[2]
                q_loc = loc[self.state[index][:self.get_attribute_num(index)].index(1)][1]
                partitions = [[] for _ in range(self.partition_num)]
                for i in range(1, self.db_schema[index][0]+1):
                    partitions[i%self.partition_num].append(i)
                block = []
                for partition in partitions:
                    for i in q_loc:
                        if i in partition:
                            block.extend(partition)
                            break
                q_rate = query[3]
                size = self.db_size[index] * len(block) / self.db_schema[index][0]
                cost += q_rate * size
        return cost

    def generate_partition(self, tables, path):
        table_name = list(tables.keys())[self.index]
        attribute = []
        for idx in self.attribute:
            attribute.append(tables[table_name][3][idx])
        partition_name = table_name + "_p0"
        print(f"CREATE TABLE {partition_name} AS ("
              f"SELECT * FROM {table_name} WHERE 1=0) temp"
              f"partition by HASH ({','.join(self.attribute)});")
        for i in range(self.partition_num):
            sub_partition_name = partition_name + str(i)
            print(
                f"create table {sub_partition_name} partition of {partition_name} for values with (modulus {self.partition_num}, remainder {i});")
        print(f"copy {partition_name} from '{path}/{table_name}.dat' with delimiter as '|' NULL '';")

