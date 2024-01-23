import numpy as np
import joblib
import sys
sys.path.append('/mnt/disk1/xzeng/postgraduate/saluki_torch')
sys.setrecursionlimit(3000)  # 将默认的递归深度修改为3000
import utils
import os
def dfs(path, start, dist,visited):
    subset = np.where(dist[start]==1)[0]
    for k in range(subset.shape[0]):
        x = subset[k]
        if(x == start or visited[x]==1): continue
        visited[x] = 1
        path.append(x)
        dfs(path, x, dist,visited)
    
def f(x):
    x.append(1)

if __name__=='__main__':
    
    current_folder = os.path.dirname(os.path.abspath(__file__))

    
    os.chdir(current_folder)

    thresh = 0.001

    st_lib = '../data/human_rbp_microrna_motif.meme'
    temp_save_dir = './temp'

    with open('./temp_id_list.txt', 'r') as file:
        lines = file.readlines()

    # Remove line breaks at the end of each string
    motif_ids = [line.strip() for line in lines]
    dist = joblib.load('../data/dist.pkl')
    assert(dist.shape[0] == len(motif_ids))

    for i in range(dist.shape[0]):
        for j in range(i, dist.shape[0]):
            if(dist[i][j]<thresh):
                dist[i][j] = 1
            else:
                dist[i][j] = 0
            dist[j][i] = dist[i][j]


    visited = np.zeros(dist.shape[0])
    results = []
    cluster_id = 1
    motif_cluster = {}
    
    for i in range(dist.shape[0]):
        if(visited[i]==1): continue
        else:
            path = [i]
            visited[i] = 1
            dfs(path, i, dist, visited)
        for id in path:
            motif_cluster[motif_ids[id]]='cluster%d'%cluster_id
        cluster_id += 1
        results.append(path)
        path=[]
    utils.save_dict_to_yaml(motif_cluster, st_lib[:-5]+'_%.10f.yaml'%thresh)
    print(results)
    print(motif_cluster)


