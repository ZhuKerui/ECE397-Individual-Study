
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt
import sys

def CalculateDistMatrix(dataset):
    # Calculate the distance matrix
    return cosine_distances(dataset)

def get_avg_dist(dist_matrix, k):
    sorted_dist_matrix = np.sort(dist_matrix)
    first_k_dist = sorted_dist_matrix[:, 1:k+1] # The first point must be itself, so we start from the second
    return np.sort(first_k_dist.sum(axis=1) / k)

def plot_avg_dist(avg_dist):
    plt.plot(range(len(avg_dist)), avg_dist)
    plt.show()

def get_slope(avg_dist, interval=1):
    slope = []
    for i in range(int((len(avg_dist) - 1) / interval)):
        slope.append((avg_dist[(i + 1) * interval] - avg_dist[i * interval]) / interval)
    return slope

def get_minpts(dist_matrix, eps):
    find_neighbor_num = (dist_matrix <= eps).sum(axis=1)
    return find_neighbor_num.sum() / (find_neighbor_num > 0).sum()

def generate_paras(dist_matrix, slope, avg_dist, interval):
    Eps = []
    Minpts = []
    for idx in range(len(slope) - 1):
        slopeDiff = slope[idx + 1] - slope[idx]
        temp_Minpts = get_minpts(dist_matrix, avg_dist[idx * interval])
        if (slopeDiff >= 0.1 * slope[idx] or slopeDiff <= 0.2 * slope[idx]) and temp_Minpts >= 2:
            Eps.append(avg_dist[idx * interval])
            Minpts.append(temp_Minpts)
    return Eps, Minpts

def do_cluster(dataset, k):
    interval = 4
    data_num = len(dataset)
    dist_matrix = CalculateDistMatrix(dataset)
    avg_dist = get_avg_dist(dist_matrix, k)
    # plot_avg_dist(avg_dist)
    slope = get_slope(avg_dist, interval)
    Eps, Minpts = generate_paras(dist_matrix, slope, avg_dist, interval)
    noiseList = dataset
    cluster_id = np.ones(data_num, dtype=np.int) * -1
    cluster_num = 0
    while len(Eps):
        eps = Eps.pop(0)
        minpts = Minpts.pop(0)
        clustering = DBSCAN(eps=eps,min_samples=max([int(minpts), 2]), metric='cosine').fit(noiseList)
        cnt = 0
        for i in range(data_num):
            if cluster_id[i] == -1:
                if clustering.labels_[cnt] != -1:
                    cluster_id[i] = cluster_num + clustering.labels_[cnt]
                cnt += 1
        cluster_num = max(cluster_id) + 1
        noiseList = dataset[cluster_id == -1]
    return cluster_id


if __name__ == "__main__":
    # execute only if run as a script
    k = int(sys.argv[1])
    arr = np.load('temp_vec.npy')
    dist_matrix = CalculateDistMatrix(arr)
    avg_dist = get_avg_dist(dist_matrix, k)
    plot_avg_dist(avg_dist)
    print(do_cluster(arr, k))