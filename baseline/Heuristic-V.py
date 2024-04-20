import numpy as np
from sklearn.cluster import AgglomerativeClustering


class Heuristic_V():
    def __init__(self, info, index):
        self.info = info
        self.index = index
        self.n = info['db_schema'][self.index][1]
        self.clusters = None
        self.n_clusters = 3

    def affinity_attribute(self):
        affinity_matrix = np.zeros((self.n, self.n))
        for query in self.info['db_query'].values():
            p = [0 for _ in range(self.n)]
            if self.index in query[1]:
                q_rate = query[3]
                for column in query[0]:
                    if column[0] == self.index:
                        p[column[1]] = q_rate
                for column in query[2]:
                    if column[0] == self.index and p[column[1]] != 0:
                        p[column[1]] = 1
                    elif column[0] == self.index:
                        p[column[1]] = q_rate
                for i in range(self.n):
                    for j in range(i, self.n):
                        if p[i] != 0 or p[j] != 0:
                            affinity_matrix[i][j] += q_rate * (p[i] + p[j]) * 0.5 / max(p[i], p[j])
                            affinity_matrix[j][i] += q_rate * (p[i] + p[j]) * 0.5 / max(p[i], p[j])
        return affinity_matrix

    def learn(self):
        x = [i for i in range(self.n)]
        clustering = AgglomerativeClustering(n_clusters=self.n_clusters, affinity=self.affinity_attribute(), linkage='average')
        clustering.fit(x)
        cluster_labels = clustering.labels_
        clusters = [[] for _ in range(self.n_clusters)]
        for i, label in enumerate(cluster_labels):
            clusters[label].append(i)
        self.clusters = clusters

    def calculate_result(self, query):
        col_block = []
        if self.index in query[1]:
            q_target = []
            for column in query[0]:
                if column[0] == self.index:
                    q_target.append(column[1])
            for column in query[2]:
                if column[0] == self.index:
                    q_target.append(column[1])
            for cluster in self.clusters:
                for i in q_target:
                    if i in cluster:
                        col_block.extend(cluster)
                        break
        return col_block

    def generate_partition_heuristic_v(tables, query_name, index, col_block):
        table_name = list(tables.keys())[index]
        partition_name = table_name + "_" + query_name + "_p2"

        print(f"CREATE TABLE {partition_name} AS ("
              f"SELECT * FROM {table_name});")
        cols = set(range(tables[table_name][2])).difference(col_block)
        # print(cols)
        for col in cols:
            print(f"ALTER TABLE {partition_name} DROP COLUMN {tables[table_name][3][col]};")
        print(f"VACUUM FULL {partition_name} ;")

