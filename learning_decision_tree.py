import os
import time

import torch
import numpy as np
import math
from scipy.optimize import minimize
import itertools
from sklearn.preprocessing import StandardScaler
from itertools import combinations
os.environ['CUDA_LAUNCH_BLOCKING'] = '2'

def cal_b(x,y,g):
    b = []
    g_x = [np.dot(g[i],x[i]) for i in range(len(y))]
    b = y - g_x
    return b
##  optimize g bar
def Optimize_g_bar(index_pair):
    global g_bar
    g_bar = np.random.randn(512)
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def MaxCosine_similarity(g_bar):
        similarities = [cosine_similarity(g[i], g_bar) for i in index_pair]
        return  -sum(similarities)
    constraint = {'type': 'eq', 'fun': lambda g_bar: np.dot(g_bar, g_bar) - 1}
    result_g_bar = minimize(MaxCosine_similarity, g_bar, constraints=constraint)
    optimal_g_bar = result_g_bar.x
    return optimal_g_bar

##  optimize alpha
def Min_Alpha(x, y, b,g_bar):
    alpha = [np.random.randint(0, 1) for _ in range(512)]
    lam=10**(-6)*20**(1/2)
    ## calculate error
    def Square_error(alpha):
        error = [(np.dot(np.multiply(alpha, g_bar),x[i])+b[i]-y[i])**2 + lam*np.linalg.norm(alpha) for i in range(len(y))]
        return sum(error)/2

    result = minimize(Square_error, alpha)
    new_alpha = result.x
    scaler = StandardScaler()
    new_alpha_scaled = scaler.fit_transform(new_alpha.reshape(-1, 1)).flatten()
    alpha_transformed = []
    for a in new_alpha_scaled:
        if a>0:
            alpha_transformed.append(1)
        else:
            alpha_transformed.append(0)

    return alpha_transformed

def g_bar_conbine(v):
    # 使用itertools.combinations生成索引的组合
    index_combinations = list(itertools.combinations(range(v), 2))
    g_bar_combination = {}
    for index_pair in index_combinations:
        g_bar = Optimize_g_bar(index_pair)
        g_bar_combination[index_pair] = g_bar
        
    return g_bar_combination
## 每一个combination中包含两个节点的index 和 两个v生成的新节点u的g_bar
def alpha_conbine(v):
    index_combinations = list(itertools.combinations(range(v), 2))
    alpha_combination = {}
    for index_pair in index_combinations:
        v1,v2 = index_pair
        x_v = [x[v1],x[v2]]
        y_v = [y[v1],y[v2]]
        b_v = [b[v1],b[v2]]
        alpha = Min_Alpha(x_v, y_v, b_v,g_bar[index_pair])
        alpha_combination[index_pair] = alpha
    return alpha_combination
## 每一个combination中包含两个节点的index 和 两个v生成的新节点u的alpha

def h_x(xi,b,alpha,g_bar): ##计算h（xi）
    return np.dot(np.multiply(alpha, g_bar),xi)+b

def compute_P0(v):

    P_0 = []
    for index in range(v):
        gamma = 1/np.mean(y)
        alpha = 512*[1]
        upper = math.exp(gamma*h_x(x[index],b[index],alpha,g[index]))
        c = []
        for i in range(len(x)):
            c.append(math.exp(gamma*h_x(x[i],b[i],alpha,g[index])))
        lower = sum(c)
        P_0.append(upper/lower)
    return P_0

def node_combination_g(v):
    node_combination = {}
    index_combinations = list(itertools.combinations(range(v), 2))
    for comb in index_combinations:
        node = []
        node = [g[x] for x in range(v) if x not in comb]
        node.append(g_bar[comb])
        node_combination[comb] = node
        ## 返回当前node情况下的gbar和gi用于计算P
    return node_combination 
def node_combination_alpha(v):
    node_combination = {}
    index_combinations = list(itertools.combinations(range(v), 2))
    for comb in index_combinations:
        node = []
        node = [512*[1] for _ in range(v) if _ not in comb]
        node.append(alpha[comb])
        node_combination[comb] = node
        ## 返回当前node情况下的alpha用于计算P
    return node_combination 

def compute_log_P(v,index_pair):
    P = {}
    gamma = 1/np.mean(y)
    best_conb = []
    ## find best child h_hat(x)
    for index in range(v-1):
        P_v={}
        upper = []
        for i in range(v-1):
            upper.append(math.exp(gamma*h_x(x[index],b[index],alpha_node[index_pair][i],g_bar_node[index_pair][i])))
        max_upper_index = np.argmax(upper)
        max_upper = max(upper)
        c = []
        for i in range(len(x)):
            c.append(math.exp(gamma*h_x(x[i],b[i],alpha_node[index_pair][max_upper_index],g_bar_node[index_pair][max_upper_index])))
        lower = sum(c)
        P_v[index_pair,index] = max_upper/lower
        P[index] = P_v[max(P_v,key =P_v.get )]
        best_conb.append(max(P_v,key =P_v.get ))
    max_key = max(P, key=P.get)
    return sum(np.log(list(P.values())))


def max_log_P():
    index_combinations = list(itertools.combinations(range(v), 2))
    P = []
    for index_pair in index_combinations:
        P.append(compute_log_P(v,index_pair))
    max_log_P = max(P)
    best_comb = index_combinations[np.argmax(P)]
    return max_log_P,best_comb


if __name__ == "__main__":
    samples_folder = '/data/LZL/ICNN/output/cat/x_g_y_sd'

    feature_key_g = 'g'
    feature_key_x = 'x'
    feature_key_y = 'y'
    # 加载特征
    g = []
    x = []
    y = []

    test_list = ['2008_000056.pt', '2008_002234.pt', '2009_001419.pt',
                 '2009_002228.pt', '2009_002499.pt', '2009_004128.pt']

    for file in test_list:
        if file.endswith('.pt'):
            filepath = os.path.join(samples_folder, file)
            data_dict = torch.load(filepath)
            features_g = data_dict[feature_key_g]
            features_x = data_dict[feature_key_x]
            features_y = data_dict[feature_key_y]
            norm_g = torch.norm(features_g)
            features_g = features_g / norm_g
            features_y = features_y / norm_g
            g.append(features_g.detach().numpy())
            x.append(features_x.detach().numpy())
            y.append(features_y.detach().numpy())


    # 将列表转换为NumPy数组
    g = np.array(g)
    x = np.array(x)
    y = np.array(y)
    b = cal_b(x,y,g)
    v = len(y)
    Q = sum(np.log(compute_P0(v)))
    P = Q
    start = time.time()
    while v > 1 and P-Q >= 0:
        g_bar = g_bar_conbine(v)
        alpha = alpha_conbine(v)
        g_bar_node = node_combination_g(v)
        alpha_node = node_combination_alpha(v)
        P,new_node = max_log_P()
        print(new_node)
        g[new_node[0]] = np.array(g_bar[new_node])
        g = np.delete(g,new_node[1],axis=0)
        x = np.delete(x,new_node[1],axis=0)
        y = np.delete(y,new_node[1],axis=0)
        v = v-1
    end = time.time()
    print('Time: {:4f} min'.format((end - start) / 60))