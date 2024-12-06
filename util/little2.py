import cv2
import pickle
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

pkl_path = "../pkl/1.pkl"

with open(pkl_path,'rb') as f:
    data = pickle.load(f)
print(data)
data = {k: v.detach().cpu() for k, v in data.items() if k < 60}
label = []
X = []



# for i in [10, 40]:
# for i in range(num):
#     for j in range(len(data[i])):
#         label.append(i)
#     for k in data[i]:
#         X.append(k.detach().cpu())

for k in range(60):
    X.append(data[k].detach().cpu().numpy())

X=np.array(X)
label = np.array(range(60))

print(1)
# print(data)
#
#
#
# # 假设你的字典是这样的
# data_dict = {[[
#     'point1': np.random.rand(60),
#     'point2': np.random.rand(60),
#     # ... 更多的点
# }

# 提取数据并转换为NumPy数组

color_map = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153], [153, 153, 153],
             [250, 170, 30], [220, 220, 0], [107, 142, 35],[152, 251, 152], [70, 130, 180], [220, 20, 60],
             [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]]

# 使用t-SNE进行降维到2维
tsne = TSNE(n_components=2, random_state=0)
X_2d = tsne.fit_transform(X)

# 可视化结果
plt.figure(figsize=(10, 8))#,facecolor='black')
ax = plt.gca()
# ax.set_facecolor('black')
plt.scatter(X_2d[:, 0], X_2d[:, 1],s=30, c=label, cmap='rainbow')
# plt.colorbar(label='Value')
# plt.title('60 class t-SNE visualization')
plt.axis('off')

# plt.xticks([])
# plt.yticks([])
plt.show()