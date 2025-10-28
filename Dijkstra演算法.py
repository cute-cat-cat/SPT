import numpy as np
import heapq

def generate_distance_map(m, g):
    """用 Dijkstra 計算每個格子到目標的最短距離"""
    M = m.shape[0]
    y = np.full((M,M), np.inf)      # 初始化距離矩陣
    visited = np.zeros((M,M), dtype=bool)

    # 取第一個目標點
    goal_idx = np.argwhere(g==1)[0]
    hq = [(0, tuple(goal_idx))]     # 優先佇列 (距離, (i,j))
    y[goal_idx[0], goal_idx[1]] = 0

    # 四個方向移動
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]

    while hq:
        dist, (i,j) = heapq.heappop(hq)
        if visited[i,j]:
            continue
        visited[i,j] = True
        for di,dj in dirs:
            ni,nj = i+di, j+dj
            if 0<=ni<M and 0<=nj<M and m[ni,nj]==0:  # 空地才走
                new_dist = dist + 1
                if new_dist < y[ni,nj]:
                    y[ni,nj] = new_dist
                    heapq.heappush(hq, (new_dist, (ni,nj)))

    # 障礙物或不可達位置設一個大值
    y[np.isinf(y)] = M*M
    return y
