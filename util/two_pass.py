# two_pass.py
import os

import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import sys
# sys.stdout = open(os.devnull, 'w')

# 0.14s 处理单张图片需要0.14s，速度刚好能容忍

def first_pass(g) -> list:
    start = time.time()
    graph = deepcopy(g)
    height = len(graph)
    width = len(graph[0])
    label = 1
    index_dict = {}
    for h in range(height):
        for w in range(width):
            if graph[h][w] == 0:
                continue
            if h == 0 and w == 0:
                graph[h][w] = label
                label += 1
                continue
            if h == 0 and graph[h][w-1] > 0:
                graph[h][w] = graph[h][w-1]
                continue
            if w == 0 and graph[h-1][w] > 0:
                if graph[h-1][w] <= graph[h-1][min(w+1, width-1)]:
                    graph[h][w] = graph[h-1][w]
                    index_dict[graph[h-1][min(w+1, width-1)]] = graph[h-1][w]
                elif graph[h-1][min(w+1, width-1)] > 0:
                    graph[h][w] = graph[h-1][min(w+1, width-1)]
                    index_dict[graph[h-1][w]] = graph[h-1][min(w+1, width-1)]
                continue
            if h == 0:
                graph[h][w] = label
                label += 1
                continue
            if w == 0:
                if graph[h-1][min(w+1, width-1)] > 0:
                    graph[h][w] = graph[h-1][min(w+1, width-1)]
                    continue
                graph[h][w] = label
                label += 1
                continue
            neighbors = [graph[h-1][w], graph[h][w-1], graph[h-1][w-1], graph[h-1][min(w+1, width-1)]]
            neighbors = list(filter(lambda x:x>0, neighbors))
            if len(neighbors) > 0:
                graph[h][w] = min(neighbors)
                for n in neighbors:
                    if n in index_dict:
                        index_dict[n] = min(index_dict[n], min(neighbors))
                    else:
                        index_dict[n] = min(neighbors)
                continue
            graph[h][w] = label
            label += 1
    # print("first_pass", time.time()-start)
    return graph, index_dict

def remap(idx_dict) -> dict:
    index_dict = deepcopy(idx_dict)
    for id in idx_dict:
        idv = idx_dict[id]
        while idv in idx_dict:
            if idv == idx_dict[idv]:
                break
            idv = idx_dict[idv]
        index_dict[id] = idv
    return index_dict

def second_pass(g, index_dict) -> list:
    start = time.time()
    graph = deepcopy(g)
    height = len(graph)
    width = len(graph[0])
    for h in range(height):
        for w in range(width):
            if graph[h][w] == 0:
                continue
            if graph[h][w] in index_dict:
                graph[h][w] = index_dict[graph[h][w]]
    # print("second_pass", time.time()-start)
    return graph

def flatten(g) -> list:
    graph = deepcopy(g)
    fgraph = sorted(set(list(graph.flatten())))
    flatten_dict = {}
    for i in range(len(fgraph)):
        flatten_dict[fgraph[i]] = i
    graph = second_pass(graph, flatten_dict)
    return graph

def two_pass(graph):
    start = time.time()
    graph_1, idx_dict = first_pass(graph)
    idx_dict = remap(idx_dict)
    graph_2 = second_pass(graph_1, idx_dict)
    graph_3 = flatten(graph_2)
    # print("two_pass", time.time() - start)
    return graph_3


#使用two_pass处理所有mask并将其存为npy文件
if __name__ == "__main__":
    #这个文件很重要，适合做论文图片
    # graph = cv2.imread(r"G:\dataset\MSCOCO2014\annotations\train2014\COCO_train2014_000000000154.png",0)
    path = r"E:\dataset\MSCOCO2014\annotations\train2014"
    out_path = r"E:\dataset\MSCOCO2014\annotations\two_pass"
    k=0
    for name in os.listdir(path):
        graph = cv2.imread(os.path.join(path, name), 0)
        class_list = np.unique(graph)[1:]
        for cls in class_list:
            print(k)
            k+=1
            graph[graph != cls] = 0
            graph[graph == cls] = 1
            # graph_3 = two_pass(graph)
            num_labels, graph_3, stats, centroids = cv2.connectedComponentsWithStats(graph, connectivity=8)
            cv2.imwrite(os.path.join(out_path, name.replace(".png", "_" + str(cls) + ".png")), graph_3)
    # np.random.seed(2)
    # graph = np.random.choice([0,1],size=(20,20))
    # graph_1, idx_dict = first_pass(graph)
    # idx_dict = remap(idx_dict)
    # graph_2 = second_pass(graph_1, idx_dict)
    # graph_3 = flatten(graph_2)
    # plt.subplot(131)
    # plt.imshow(graph)
    # plt.subplot(132)
    # plt.imshow(graph_3)
    # plt.subplot(133)
    # plt.imshow(graph_3>0)
    # plt.show()
    # plt.savefig('random_bin_graph.png')
