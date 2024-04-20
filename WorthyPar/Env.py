class ENV(object):
    def __init__(self, info):
        self.max_step = 10                                   # The maximum number of steps performed on each table
        self.db_schema = info['db_schema']                   # The number of horizontal blocks and attribute columns for each table in the database
        self.db_size = info['db_size']                       # The size of each table in the database
        self.db_query = info['db_query']                     # Information queried in the workload
        self.state = None                                    # Status coding
        self.state_num = self.get_state_num()                # Status code length
        self.action_num = self.get_action_num()              # Action code length
        self.done = False                                    # Whether to complete the partition operation
        self.step_num = 0                                    # The number of steps currently executed
        self.MAX_STEP = self.max_step * len(self.db_schema)  # Maximum number of executed steps
        self.cost = []                                       # cost
        self.block_size = []
        self.block_num = []
        self.partition_num = []
        self.q_target = []
        self.q_loc = []
        self.row_block = []
        self.col_block = []
        self.order_key = []

    def set_state(self, label, state):
        self.state[label-1] = state

    def get_state_num(self):
        state_num = []
        for i in range(len(self.db_schema)):
            state_num.append(self.get_state_attribute_num(i) + self.get_state_line_h_num(i) + self.get_state_line_v_num(i) + len(self.db_query))
        return state_num

    def get_action_num(self):
        action_num = []
        for i in range(len(self.db_schema)):
            action_num.append(3 + self.get_state_attribute_num(i) + self.get_state_line_h_num(i) + self.get_state_line_v_num(i))
        return action_num

    def get_state_attribute_num(self, index):
        return self.db_schema[index][1]

    def get_state_line_h_num(self, index):
        return self.db_schema[index][0]-1

    def get_state_line_v_num(self, index):
        return self.db_schema[index][1]-1

    def get_block_size(self, index):
        return self.block_size[index]

    def get_block_num(self, index):
        return self.block_num[index]

    def get_partition_num(self, index):
        return self.partition_num[index]

    def reset(self):
        self.step_num = 0
        self.done = False
        self.cost = [i for i in self.db_size]
        self.block_size = [i for i in self.db_size]
        self.block_num = [1 for _ in range(len(self.db_size))]
        self.partition_num = [1 for _ in range(len(self.db_size))]
        self.q_target = [[] for _ in range(len(self.db_size))]
        self.q_loc = [[] for _ in range(len(self.db_size))]
        self.row_block = [[] for _ in range(len(self.db_size))]
        self.col_block = [[] for _ in range(len(self.db_size))]
        self.order_key = [0 for _ in range(len(self.db_size))]
        state = [[] for _ in range(len(self.db_schema))]
        for i in range(len(self.db_schema)):
            db = self.db_schema[i]
            for j in range(db[1]):
                state[i].append(0)
            for j in range(db[0]-1):
                state[i].append(0)
            for j in range(db[1]-1):
                state[i].append(0)
            for query in self.db_query.values():
                state[i].append(query[3])
        self.state = state
        return state

    def get_actions(self, index):
        actions = []
        attribute_num = self.get_state_attribute_num(index)
        if sum(self.state[index][:attribute_num]) == 0:
            actions.extend(self.get_action_list(index, 1))
        else:
            actions.extend(self.get_action_list(index, 2))
            actions.extend(self.get_action_list(index, 3))
        return actions

    def get_action_list(self, index, type):
        action_list = []
        action = [0 for _ in range(self.get_action_num()[index])]
        if self.step_num >= self.MAX_STEP*0.2:
            action_list.append(action.copy())
        if type == 1:
            action[0] = 1
            start_index = 3
            state_attribute_num = self.get_state_attribute_num(index)
            for i in range(state_attribute_num):
                action[start_index + i] = 1
                action_list.append(action.copy())
                action[start_index + i] = 0

        elif type == 2:
            action[1] = 1
            start_index = self.get_state_attribute_num(index) + 3
            state_line_h_num = self.get_state_line_h_num(index)
            for i in range(state_line_h_num):
                if self.state[index][start_index - 3 + i] == 0:
                    action[start_index + i] = 1
                    action_list.append(action.copy())
                    action[start_index + i] = 0
        elif type == 3:
            action[2] = 1
            start_index = self.get_action_num()[index] - self.get_state_line_v_num(index)
            state_line_v_num = self.get_state_line_v_num(index)
            for i in range(state_line_v_num):
                if self.state[index][start_index - 3 + i] == 0:
                    action[start_index + i] = 1
                    action_list.append(action.copy())
                    action[start_index + i] = 0
        return action_list

    def step(self, index, action):
        state_next = self.take_action(index, action)
        if sum(action[:3]) == 0:
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
        if action[0] == 1:
            self.order_key[index] = action[3:].index(1)
        if sum(action[:3]) != 0:
            i = action[3:].index(1)
            state_next[i] = 1
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
        block_size = 0
        block_num = 0
        partition_num = 0
        for query in self.db_query.values():
            if index in query[1]:
                # 提取查询信息
                q_target = []
                loc = []
                for column in query[0]:
                    if column[0] == index:
                        q_target.append(column[1])
                for column in query[2]:
                    if column[0] == index:
                        q_target.append(column[1])
                        loc = column[2]
                q_target = list(set(q_target))
                q_loc = loc[self.state[index][:self.get_state_attribute_num(index)].index(1)][1]
                # 进行混合分区划分
                row_state = self.state[index][self.get_state_attribute_num(index):
                                              self.get_state_attribute_num(index) + self.get_state_line_h_num(index)]
                col_state = self.state[index][self.get_state_attribute_num(index) + self.get_state_line_h_num(index):]
                row_partition = []
                temp = []
                for i in range(len(row_state) + 1):
                    temp.append(i+1)
                    if i < len(row_state) and row_state[i] == 1:
                        row_partition.append(temp)
                        temp = []
                    elif i == len(row_state):
                        row_partition.append(temp)
                col_partition = []
                temp = []
                for i in range(len(col_state) + 1):
                    temp.append(i)
                    if i < len(col_state) and col_state[i] == 1:
                        col_partition.append(temp)
                        temp = []
                    elif i == len(col_state):
                        col_partition.append(temp)
                # 计算查询所在分区块大小
                row_num = 0
                col_num = 0
                row_block_num = 0
                col_block_num = 0
                for partition in row_partition:
                    for i in q_loc:
                        if i in partition:
                            row_num += len(partition)
                            row_block_num += 1
                            break
                for partition in col_partition:
                    for i in q_target:
                        if i in partition:
                            col_num += len(partition)
                            col_block_num += 1
                            break
                q_rate = query[3]

                size = self.db_size[index] * (row_num * col_num) / (
                            self.db_schema[index][0] * self.db_schema[index][1])
                block_size += q_rate * size
                block_num += q_rate * row_block_num * col_block_num
                partition_num = len(row_partition) * len(col_partition)
                cost += q_rate * size
        self.block_size = block_size
        self.block_num = block_num
        self.partition_num = partition_num
        return cost

    def calculate_result(self, index, query):
        q_target = []
        loc = []
        for column in query[0]:
            if column[0] == index:
                q_target.append(column[1])
        for column in query[2]:
            if column[0] == index:
                q_target.append(column[1])
                loc = column[2]
        q_target = list(set(q_target))
        q_loc = loc[self.state[index][:self.get_state_attribute_num(index)].index(1)][1]

        row_state = self.state[index][self.get_state_attribute_num(index):
                                      self.get_state_attribute_num(index) + self.get_state_line_h_num(index)]
        col_state = self.state[index][self.get_state_attribute_num(index) + self.get_state_line_h_num(index):
                                      self.get_state_attribute_num(index) + self.get_state_line_h_num(index)
                                      +self.get_state_line_v_num(index)]
        # print('row_state', len(row_state), row_state)
        # print('col_state', len(col_state), col_state)
        row_partition = []
        temp = []
        for i in range(len(row_state) + 1):
            temp.append(i + 1)
            if i < len(row_state) and row_state[i] == 1:
                row_partition.append(temp)
                temp = []
            elif i == len(row_state):
                row_partition.append(temp)
        col_partition = []
        temp = []
        for i in range(len(col_state) + 1):
            temp.append(i)
            if i < len(col_state) and col_state[i] == 1:
                col_partition.append(temp)
                temp = []
            elif i == len(col_state):
                col_partition.append(temp)

        row_block = []
        col_block = []
        for partition in row_partition:
            for i in q_loc:
                if i in partition:
                    row_block.extend(partition)
                    break
        for partition in col_partition:
            for i in q_target:
                if i in partition:
                    col_block.extend(partition)
                    break
        return q_target, q_loc, row_block, col_block

    def show_result(self):
        Result = dict()
        for query_name, query in self.db_query.items():
            result = []
            for index in query[1]:
                q_target, q_loc, row_block, col_block = self.calculate_result(index, query)
                result.append([index, self.order_key[index], row_block, col_block])
            Result[query_name] = result
        return Result, self.block_size, self.block_num, self.partition_num
