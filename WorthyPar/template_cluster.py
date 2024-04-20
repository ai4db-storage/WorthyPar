import copy
import random
import matplotlib.pyplot as plt
import numpy as np
from fastdtw import fastdtw
import math
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors

USE_KNN = False
KNN_ALG = "kd_tree"


def DTW(x, y):
    x = np.array(x).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)
    x = normalize(x, axis=0)
    y = normalize(y, axis=0)
    dist_algorithm = lambda a, b: abs(a - b)
    return fastdtw(x, y, dist=dist_algorithm)[0]


class DBSCAN(object):
    def __init__(self, rho, min_Pts):
        self.rho = rho
        self.min_Pts = min_Pts

    def find_neighbors(self, target, templates):
        neighbors = set()
        for t in templates.keys():
            target_feature = templates[target].feature
            # plt.plot(target_feature)
            # plt.show()
            if target != t:
                t_feature = templates[t].feature
                distance = DTW(target_feature, t_feature)
                plt.plot(normalize(np.array(t_feature).reshape(-1, 1), axis=0))
                plt.plot(normalize(np.array(target_feature).reshape(-1, 1), axis=0))
                plt.title(distance)
                plt.show()
                if distance < self.rho:
                    neighbors.add(t)
        return neighbors

    def fit(self, templates):
        k = -1
        neighbor_lists = dict()
        cores = []
        candidates = set()  # 初始时将所有点标记为未访问
        clusters = dict()
        for t in templates.keys():
            neighbor_lists[t] = self.find_neighbors(t, templates)
            # print(neighbor_lists[t])
            candidates.add(t)
            if len(neighbor_lists[t]) >= self.min_Pts:
                cores.append(t)
        cores = set(cores)
        while len(cores) > 0:
            pre_candidates = copy.deepcopy(candidates)
            candidate = random.choice(list(cores))
            k = k + 1
            queue = list()
            queue.append(candidate)
            candidates.remove(candidate)
            while len(queue) > 0:
                q = queue[0]
                queue.remove(q)
                if len(neighbor_lists[q]) >= self.min_Pts:
                    for item in neighbor_lists[q]:
                        if item in candidates:
                            queue.append(item)
                            candidates.remove(item)
            cluster_k = pre_candidates - candidates
            for i in range(len(cluster_k)):
                clusters[list(cluster_k)[i]] = k
            cores = cores - cluster_k
        """
        将各个离群点作为单独的聚类处置
        """
        for item in candidates:
            k = k + 1
            clusters[item] = k
        return clusters, k+1


class Center(object):
    def __init__(self, num_samples):
        self.state = 'add'
        self.feature = []
        self.feat_constraint = num_samples

    def change_state(self):
        if self.state == 'add':
            self.state = 'update'
        elif self.state == 'update':
            self.state = 'add'
        else:
            raise NotImplementedError

    def update_feature(self, vals, pos):
        if self.state == 'add':
            assert pos
            self.feature = self.feature + vals
            self.change_state()
        else:
            if pos:
                for idx in range(len(vals)):
                    self.feature[-idx - 1] += vals[idx]
            else:
                for idx in range(len(vals)):
                    self.feature[-idx - 1] -= vals[idx]

    def get_feature(self, padding=False):
        if len(self.feature) < self.feat_constraint:
            paddings = [0 for _ in range(self.feat_constraint - len(self.feature))]
            if padding:
                return paddings + self.feature
            else:
                return self.feature
        return self.feature[-self.feat_constraint:]


class Cluster(object):
    """
    cluster_gap : int
    templates whose feature length is smaller than feature_size_constraint is filtered before
    num_sample : int
    represents the dim of center features
    bound : int
    templates whose feature dim  is lower than bound will be filtered
    which means that we cannot assign such template to an appropriate cluster due to lack of information
    """

    def __init__(self, args, cluster_gap, rho, num_sample, bound=None):
        self.args = args
        self.cluster_gap = cluster_gap
        self.rho = rho
        self.centers = dict()
        self.cluster_sizes = dict()
        self.assignments = dict()
        self.sample_constraint = num_sample
        self.state = 'initialize'
        if bound is None:
            self.bound = self.sample_constraint // 2
        else:
            self.bound = bound

        self.centers_log = dict()
        self.assignments_log = dict()
    """
    调用DBSCAN算法初始化聚类
    """
    # templates : Template_Extractor.templates (dict)
    def cluster_generate(self, templates, feature_dims):
        assert self.state == 'initialize'
        filtered_templates = dict()
        dbscan = DBSCAN(rho=self.rho, min_Pts=1)
        for t in templates.keys():
            # 过滤掉到达率特征不充分的模板
            if len(templates[t].feature) < self.bound:
                self.assignments[t] = -1
            else:
                filtered_templates[t] = templates[t]
        clusters, num_clusters = dbscan.fit(filtered_templates)
        for t in clusters.keys():
            self.assignments[t] = clusters[t]

        cluster_sizes = {idx: 0 for idx in range(num_clusters)}
        for t in clusters.keys():
            cluster_sizes[clusters[t]] += 1

        self.cluster_sizes = cluster_sizes
        self.state = 'adjust'

        self.centers = {idx: Center(self.sample_constraint) for idx in range(num_clusters)}

        for cluster in self.centers.keys():
            self.centers[cluster].feature = np.zeros(shape=feature_dims)
            for template in self.assignments.keys():
                if self.assignments[template] == cluster:
                    template_feature = templates[template].feature
                    feature_dim = len(template_feature)
                    if feature_dim < feature_dims:
                        template_feature = [0 for _ in range(feature_dims - feature_dim)] + template_feature
                    if feature_dim > feature_dims:
                        template_feature = template_feature[-feature_dims:]
                    template_feature = np.array(template_feature)
                    self.centers[cluster].feature += template_feature
            self.centers[cluster].feature = list(self.centers[cluster].feature)

        self.centers_log[0] = self.centers
        self.assignments_log[0] = self.assignments

    """
    不断更新各聚类的center，但不会根据center的变化重新聚类
    """
    def center_trace(self, templates):
        for cluster in self.centers.keys():
            self.centers[cluster].state = 'add'
            for template in self.assignments.keys():
                if self.assignments[template] == cluster:
                    if self.state == 'add':
                        self.centers[cluster].feature.append(templates[template].feature[-1])
                        self.centers[cluster].change_state()
                    else:
                        self.centers[cluster].feature[-1] += templates[template].feature[-1]

    """
    调整聚类，并相应的重新训练部署在其上的预测模型
    """
    def cluster_adjust(self, templates):
        assert self.state == 'adjust'
        # for cluster in self.centers.keys():
        #     self.centers[cluster].state = 'add'

        next_cluster = len(self.centers)

        # 将新到达的模板加入self.assignments
        # self.assignments用于表示各个模板分别属于哪个聚类
        for t in templates.keys():
            if t not in self.assignments.keys():
                self.assignments[t] = -1

        adjust_assignments = self.assignments.copy()

        # Update cluster centers with new data in the last gap
        # This function is replaced by self.center_trace
        # for cluster in self.centers.keys():
        #     for template in self.assignments.keys():
        #         if self.assignments[template] == cluster:
        #             self.add2center(center=self.centers[cluster], data=templates[template].feature)

        # Use kdtree for single point assignment
        if USE_KNN:
            clusters = sorted(self.centers.keys())

            samples = list()
            for cluster in clusters:
                sample = self.centers[cluster].get_feature(padding=True)
                samples.append(sample)

            if len(samples) == 0:
                nbrs = None
            else:
                normalized_samples = normalize(np.array(samples), copy=False)
                nbrs = NearestNeighbors(n_neighbors=1, algorithm=KNN_ALG, metric='l2')
                nbrs.fit(normalized_samples)

        template_count = 0
        for t in sorted(self.assignments.keys()):
            template_count += 1
            # Test whether this template still belongs to the original cluster
            if adjust_assignments[t] != -1:
                center = self.centers[adjust_assignments[t]]
                if self.cluster_sizes[adjust_assignments[t]] == 1 or \
                        DTW(center.get_feature(), templates[t].get_feature(self.sample_constraint)) < self.rho:
                    continue

            # the template is eliminated from the original cluster
            if adjust_assignments[t] != -1:
                cluster = adjust_assignments[t]
                self.cluster_sizes[cluster] -= 1

                center_feature = np.array(self.centers[cluster].feature)
                template_feature = templates[t].feature
                if len(template_feature) < len(center_feature):
                    template_feature = [0 for _ in range(len(center_feature) - len(template_feature))] \
                                       + template_feature
                else:
                    template_feature = template_feature[-len(center_feature):]
                template_feature = np.array(template_feature)
                center_feature = center_feature - template_feature
                self.centers[cluster].feature = list(center_feature)

            # Whether this template has "arrived" yet?
            if adjust_assignments[t] == -1 and len(templates[t].feature) < self.bound:
                continue

            # whether this template is similar to the center of an existing cluster
            new_cluster = None
            if not USE_KNN or nbrs is None:
                for cluster in self.centers.keys():
                    center = self.centers[cluster]
                    if DTW(center.get_feature(), templates[t].get_feature(self.sample_constraint)) < self.rho:
                        new_cluster = cluster
                        break
            else:
                nbr = nbrs.kneighbors(normalize(np.array(templates[t].get_feature(self.sample_constraint)).reshape(-1, 1), axis=0), return_distance=False)[0][0]
                if DTW(templates[t].get_feature(self.sample_constraint), self.centers[clusters[nbr]].get_feature()) < self.rho:
                    new_cluster = clusters[nbr]

            if new_cluster is not None:
                if adjust_assignments[t] == -1:
                    adjust_assignments[t] = new_cluster

                    center_feature = np.array(self.centers[new_cluster].feature)
                    template_feature = templates[t].feature
                    if len(template_feature) < len(center_feature):
                        template_feature = [0 for _ in range(len(center_feature) - len(template_feature))] \
                                           + template_feature
                    else:
                        template_feature = template_feature[-len(center_feature):]
                    template_feature = np.array(template_feature)
                    center_feature = center_feature + template_feature
                    self.centers[new_cluster].feature = list(center_feature)

                    self.cluster_sizes[new_cluster] += 1
                    continue

            # construct a new cluster for newly arrived template
            adjust_assignments[t] = next_cluster
            self.centers[next_cluster] = Center(self.sample_constraint)

            self.centers[new_cluster].feature = templates[t].feature

            self.cluster_sizes[next_cluster] = 1
            next_cluster += 1

        clusters = sorted(self.centers.keys())
        # a union-find set to track the root cluster for clusters that have been merged
        root = [-1] * len(clusters)

        if USE_KNN:
            samples = list()
            for cluster in clusters:
                sample = self.centers[cluster].get_feature(padding=True)
                samples.append(sample)

            if len(samples) == 0:
                nbrs = None
            else:
                normalized_samples = normalize(np.array(samples), copy=False)
                nbrs = NearestNeighbors(n_neighbors=2, algorithm=KNN_ALG, metric='l2')
                nbrs.fit(normalized_samples)

        for i in range(len(clusters)):
            c = None
            c1 = clusters[i]

            if not USE_KNN or nbrs is None:
                for j in range(i+1, len(clusters)):
                    c2 = clusters[j]
                    if DTW(self.centers[c1].get_feature(), self.centers[c2].get_feature()) < self.rho:
                        c = c2
                        break
            else:
                nbr = nbrs.kneighbors(np.array(self.centers[c1].get_feature()).reshape(-1, 1), return_distance=False)[0]
                if clusters[nbr[0]] == c1:
                    nbr = nbr[1]
                else:
                    nbr = nbr[0]

                while root[nbr] != -1:
                    nbr = root[nbr]

                if c1 != clusters[nbr] and DTW(self.centers[c1].get_feature(), self.centers[clusters[nbr]].get_feature()) < self.rho:
                    c = clusters[nbr]

            if c is not None:
                center_feature = np.array(self.centers[c].feature)
                template_feature = self.centers[c1].feature
                if len(template_feature) < len(center_feature):
                    template_feature = [0 for _ in range(len(center_feature) - len(template_feature))] \
                                       + template_feature
                else:
                    template_feature = template_feature[-len(center_feature):]
                template_feature = np.array(template_feature)
                center_feature = center_feature + template_feature
                self.centers[c].feature = list(center_feature)

                self.cluster_sizes[c] += self.cluster_sizes[c1]
                del self.centers[c1]
                del self.cluster_sizes[c1]

                if USE_KNN and nbrs is not None:
                    root[i] = nbr

                for t in templates.keys():
                    if adjust_assignments[t] == c1:
                        adjust_assignments[t] = c

        self.assignments = adjust_assignments

        self.centers_log[len(self.centers_log)] = self.centers
        self.assignments_log[len(self.assignments_log)] = self.assignments

        print('adjust cluster successfully')

    def add2center(self, center, data, pos=True):
        if len(data) < self.cluster_gap:
            vals = [0 for i in range(self.cluster_gap - len(data))]
            vals = vals + data
        else:
            vals = data[-self.cluster_gap:]
        center.update_feature(vals, pos)
