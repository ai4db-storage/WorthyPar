# encoding: utf-8
class ENV(object):
    def __init__(self, db_list, predicted_query, actua_db):
        self.db_list = db_list
        self.temp_list = []
        for i in range(len(self.db_list)):
            self.temp_list.append(sum(self.db_list[i]))
        self.predicted_query = predicted_query
        self.actua_db = actua_db
        self.optimization_rate = 0.5
        self.n_states = sum(self.temp_list) + len(self.predicted_query)
        self.n_actions = len(self.db_list) + 3 + sum(self.temp_list)
        self.state = None
        self.actua_state = None
        self.step_num = 0
        self.MAX_STEP = 10 * len(self.db_list)
        self.last_query_num = []
        self.last_query_block_num = 0
        self.done = None

    def set_state(self, num, state):
        self.state[num - 1] = state

    def set_actua_state(self, num, actua_state):
        self.actua_state[num - 1] = actua_state

    def get_n_state(self, num):
        return self.temp_list[num - 1] + len(self.predicted_query) + 10

    def get_n_action(self, num):
        return 3 + self.temp_list[num - 1]

    def get_actua_state(self, num):
        return self.actua_state[num - 1]

    def reset(self):
        self.step_num = 0
        s = []
        actua_state = []
        for i in range(len(self.db_list)):
            actua_state.append([])
            s.append([])
            self.last_query_num.append(self.actua_db[i] * self.db_list[i][1])
        for i in range(len(self.db_list)):
            db = self.db_list[i]
            for j in range(db[0]):
                s[i].append(0)
                actua_state[i].append(j + 1)
            for j in range(db[1]):
                s[i].append(0)
            for query in self.predicted_query:
                s[i].append(query[1])

        self.state = s
        self.actua_state = actua_state
        self.done = False

        return s, actua_state

    def get_new_state(self, num):
        return self.state[num - 1]

    def get_start(self, a):
        for i in range(len(a) - 1, 0, -1):
            if a[i] < a[i - 1]:
                return i
        return 0

    def get_actions_on_table_and_action(self, table_num, action_num):
        actions = []
        # When self.step_num >= self.MAX_STEP*0.2, add No Operation to the actions
        temp_action = [0, 0, 0]
        for i in range(self.temp_list[table_num - 1]):
            temp_action.append(0)
        if self.step_num >= self.MAX_STEP * 0.2:
            actions.append(temp_action.copy())
        if action_num == 1:
            temp_action[0] = 1
            need_num = self.db_list[table_num - 1][0]
            need_select = 0
            if need_num > 2:
                for i in range(need_num):
                    if i >= need_num - 2:
                        break
                    else:
                        temp_action[need_select + i + 3] = 1
                        for j in range(i + 2, need_num):
                            temp_action[need_select + j + 3] = 1
                            actions.append(temp_action.copy())
                            temp_action[need_select + j + 3] = 0
                        temp_action[need_select + i + 3] = 0
        if action_num == 2:
            temp_action[1] = 1
            need_num = self.db_list[table_num - 1][0]
            for i in range(need_num - 1):
                if self.state[table_num - 1][i] == 0:
                    temp_action[i + 3] = 1
                    actions.append(temp_action.copy())
                    temp_action[i + 3] = 0
        if action_num == 3:
            temp_action[2] = 1
            need_select = self.db_list[table_num - 1][0]
            need_num = self.db_list[table_num - 1][1]
            for i in range(need_num - 1):
                if self.state[table_num - 1][need_select + i] == 0:
                    temp_action[need_select + i + 3] = 1
                    actions.append(temp_action.copy())
                    temp_action[need_select + i + 3] = 0
        return actions

    def get_actions(self, num):
        # Gets the actions that can be performed in this table
        actions = []
        idx_num = self.temp_list[num - 1]
        if sum(self.state[num - 1][:idx_num]) == 0 and self.step_num <= 3 * len(self.db_list):
            actions.extend(self.get_actions_on_table_and_action(num, 1))  # change layout
            actions.extend(self.get_actions_on_table_and_action(num, 2))  # horizontal partition
            actions.extend(self.get_actions_on_table_and_action(num, 3))  # vertical partition

        else:
            actions.extend(self.get_actions_on_table_and_action(num, 2))  # horizontal partition
            actions.extend(self.get_actions_on_table_and_action(num, 3))  # vertical partition

        return actions

    def step(self, num, a):
        s_ = self.take_action(num, a)
        if self.state[num - 1] == s_ or a[0] == 1:
            r = self.reward(num)
        else:
            self.state[num - 1] = s_
            r = self.reward(num)
        self.step_num += 1
        if self.step_num == self.MAX_STEP:
            self.done = True
        return s_, r, self.done

    def take_action(self, num, a):
        s_ = []
        if sum(a[:3]) == 0:
            s_ = self.state[num - 1]
        else:
            action_num = a[:3].index(1) + 1

            if action_num == 1:
                s_ = self.state[num - 1].copy()
                indexs = []
                need_num = self.db_list[num - 1][0]
                for i in range(need_num):
                    if a[i + 3] == 1:
                        indexs.append(i)
                assert len(indexs) == 2
                next_nun = self.actua_state[num - 1][indexs[1]]
                self.actua_state[num - 1].pop(indexs[1])
                self.actua_state[num - 1].insert(indexs[0], next_nun)
            if action_num == 2:
                s_ = self.state[num - 1].copy()
                need_num = self.db_list[num - 1][0]
                for i in range(need_num - 1):
                    if a[i + 3] == 1:
                        s_[i] = 1
            if action_num == 3:
                s_ = self.state[num - 1].copy()
                need_num = self.db_list[num - 1][1]
                need_select = self.db_list[num - 1][0]
                for i in range(need_num - 1):
                    if a[need_select + i + 3] == 1:
                        s_[need_select + i] = 1

        return s_

    def reward_function(self, table_num, query_num):
        if query_num == 0:
            return query_num
        return -1 + self.last_query_num[table_num - 1] / query_num

    def query_num_function(self, num):
        # query_num: cost(P,Q) = SUM[io(sizeof(P)×access(P,q))]
        query_cost = 0
        query_cost_table = []
        query_cost_rate = []
        for query in self.predicted_query:
            radio = query[1]  # Extract the frequency of the query in the workload
            query = query[0]  # Extract the information involved in the query
            query_target = query[0]
            query_table = query[1]
            query_where = query[2]

            if num not in query_table:
                continue

            table_num = num
            row_state = self.state[table_num - 1][:self.db_list[table_num - 1][0]]
            col_state = self.state[table_num - 1][self.db_list[table_num - 1][0]:
                                                  self.db_list[table_num - 1][0] + self.db_list[table_num - 1][1]]
            temp_where = []
            for where in query_where:
                if where[0] == table_num:
                    temp_where.append(where[1])
            temp_target = []
            for target in query_target:
                if target[0] == table_num:
                    temp_target.append(target[1])

            # horizontal partition
            row_index = []
            for i in range(len(row_state)):
                if row_state[i] == 1:
                    row_index.append(i + 1)

            row_partition = []
            if len(row_index) != 0:
                j = 0
                temp_partition = []
                for i in range(len(self.actua_state[table_num - 1])):
                    if j <= len(row_index) - 1 and i == row_index[j]:
                        row_partition.append(temp_partition)
                        temp_partition = [self.actua_state[table_num - 1][i]]
                        j += 1
                    else:
                        temp_partition.append(self.actua_state[table_num - 1][i])
                row_partition.append(temp_partition)
            else:
                temp_partition = self.actua_state[table_num - 1]
                row_partition.append(temp_partition)

            # vertical partition
            col_index = []
            for i in range(len(col_state)):
                if col_state[i] == 1:
                    col_index.append(i + 1)

            col_partition = []
            if len(col_index) != 0:
                j = 0
                temp_partition = []
                for i in range(len(col_state)):
                    if j <= len(col_index) - 1 and i == col_index[j]:
                        col_partition.append(temp_partition)
                        temp_partition = [i + 1]
                        j += 1
                    else:
                        temp_partition.append(i + 1)
                col_partition.append(temp_partition)
            else:
                temp_partition = [i + 1 for i in range(len(col_state))]
                col_partition.append(temp_partition)

            temp_where_set = set()
            for where in temp_where:
                temp_target.append(where[0])
                for i in where[1]:
                    temp_where_set.add(i)
            temp_target = list(set(temp_target))

            row_num = 0
            col_num = 0
            for partition in row_partition:
                for i in temp_where_set:
                    if i in partition:
                        row_num += len(partition)
                        break
            for partition in col_partition:
                for i in temp_target:
                    if i in partition:
                        col_num += len(partition)
                        break
            query_cost_rate.append(radio)
            query_cost_table.append((row_num / 10) * col_num * self.actua_db[table_num - 1])
        for i in range(len(query_cost_rate)):
            query_cost += query_cost_rate[i] / sum(query_cost_rate) * query_cost_table[i]
        return query_cost

    def reward(self, num):
        query_num = self.query_num_function(num)
        reward = self.reward_function(num, query_num)
        self.last_query_num[num - 1] = query_num
        return reward
