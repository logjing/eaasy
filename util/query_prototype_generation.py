import numpy as np
from two_pass import two_pass
import torch
from torch import nn
import time

cos = nn.CosineSimilarity(dim=0,eps=1e-6)

def query_prototype_generation(ms, fs, fq):
    """
    Args:
        ms: mask_support [bs, shot, h, w] [1, 1, h, w]
        fs: feature_support [bs, shot, channel, h, w]
        fq: feature_query [bs, channel, h, w]
    Returns:
        pq: prototype_query [bs, channel]
    """
    start = time.time()
    bs,shot,channel,h,w = fs.shape
    # ----------------------------通过IAP得到每张图片的多个支撑集的原型------------------------ #
    instance_prototype = [] #[bs,k,channel] 这里的k不是定值是一个元素长度不等的list。将多个shot的图片里面的instance全部在一起
    for i in range(bs):
        instance_prototype_part = []
        for j in range(shot):
            two_pass_mask = two_pass(ms[i][j]) #形状为[h,w]的带有标签的灰度图
            num_region = int(torch.max(two_pass_mask).item())
            for k in range(1,num_region+1):
                mask = (two_pass_mask == k).float() #标签为k的连通区域(h,w)
                if(mask.sum()<20): continue #当前前景太小，忽略不计
                instance_prototype_part.append(torch.mean(mask * fs[i,j], (1,2))) #MAP 文心一言生成的是(2,3)
        instance_prototype.append(torch.tensor(np.array(instance_prototype_part)))
    print(time.time()-start)
    start = time.time()
    # -----------------------------计算Candidate Query Prototype--------------------------- #
    candidate_query_pro = []
    candidate_query_pro.append(torch.mean(fq[:, :, 0:h//2, 0:w//2], dim=[2, 3]))
    candidate_query_pro.append(torch.mean(fq[:, :, h//2:h, 0:w//2], dim=[2, 3]))
    candidate_query_pro.append(torch.mean(fq[:, :, 0:h//2, w//2:w], dim=[2, 3]))
    candidate_query_pro.append(torch.mean(fq[:, :, h//2:h, w//2:w], dim=[2, 3]))
    candidate_query_pro.append(
        torch.mean(fq[:, :, h//2 - h//4: h//2 + h//4, w//2 - w//4:w//2 + w//4],
                   dim=[2, 3]))
    candidate_query_pro = torch.cat(candidate_query_pro, dim=0).view(5, bs, channel).permute(1, 0, 2) # [bs,5,channel]
    print(time.time() - start)
    start = time.time()
    # -----------------------------计算similarity--------------------------- #
    similarity_map = [] # [bs,shot,5,k]是一个长度为bs*shot的list，每个元素的大小都是5*k, k不定
    for i in range(bs):
        candidate_prototype = candidate_query_pro[i] # [5,channel]
        support_instance_prototype = instance_prototype[i] # [k,channel]
        k = support_instance_prototype.shape[0]
        sim_map = torch.zeros((5,k))
        for k1 in range(5):
            for k2 in range(k):
                sim_map[k1,k2] = cos(candidate_prototype[k1], support_instance_prototype[k2])

        similarity_map.append(torch.max(sim_map, dim=1)[0])
    similarity_map = torch.tensor(np.array(similarity_map)) # [bs,5]
    similarity_map = nn.functional.softmax(similarity_map, dim=1)
    pq = torch.mean(candidate_query_pro * similarity_map.unsqueeze(-1), dim=[0,1]) # [bs, channel]
    print(time.time() - start)
    return pq

#这部分是用numpy实现的功能
# def query_prototype_generation(ms, fs, fq):
#     """
#     Args:
#         ms: mask_support [bs, shot, h, w] [1, 1, h, w]
#         fs: feature_support [bs, shot, channel, h, w]
#         fq: feature_query [bs, channel, h, w]
#     Returns:
#         pq: prototype_query [bs, channel]
#     """
#     bs,shot,channel,h,w = fs.shape
#     # ----------------------------通过IAP得到每张图片的多个支撑集的原型------------------------ #
#     instance_prototype = [] #[bs,k,channel] 这里的k不是定值是一个元素长度不等的list。将多个shot的图片里面的instance全部在一起
#     for i in range(bs):
#         instance_prototype_part = []
#         for j in range(shot):
#             two_pass_mask = two_pass(ms[i][j]) #形状为[h,w]的带有标签的灰度图
#             num_region = np.max(two_pass_mask)
#             for k in range(1,num_region+1):
#                 mask = (two_pass_mask == k).int() #标签为k的连通区域(h,w)
#                 if(sum(mask)<20): continue #当前前景太小，忽略不计
#                 instance_prototype_part.append(np.mean(mask * fs[i][j], (1,2))) #MAP
#         instance_prototype.append(np.array(instance_prototype_part))
#
#     # -----------------------------计算Candidate Query Prototype--------------------------- #
#     candidate_query_pro = []
#     candidate_query_pro.append(torch.mean(fq[:, :, 0:h//2, 0:w//2], dim=[2, 3]))
#     candidate_query_pro.append(torch.mean(fq[:, :, h//2:h, 0:w//2], dim=[2, 3]))
#     candidate_query_pro.append(torch.mean(fq[:, :, 0:h//2, w//2:w], dim=[2, 3]))
#     candidate_query_pro.append(torch.mean(fq[:, :, h//2:h, w//2:w], dim=[2, 3]))
#     candidate_query_pro.append(
#         torch.mean(fq[:, :, h//2 - h//4: h//2 + h//4, w//2 - w//4:w//2 + w//4],
#                    dim=[2, 3]))
#     candidate_query_pro = torch.cat(candidate_query_pro, dim=0).view(5, bs, channel).permute(1, 0, 2) # [bs,5,channel]
#
#     # -----------------------------计算similarity--------------------------- #
#     similarity_map = [] # [bs,shot,5,k]是一个长度为bs*shot的list，每个元素的大小都是5*k, k不定
#     for i in range(bs):
#         candidate_prototype = candidate_query_pro[i] # [5,channel]
#         support_instance_prototype = instance_prototype[i] # [k,channel]
#         k = support_instance_prototype.shape[0]
#         sim_map = np.zeros((5,k))
#         for k1 in range(5):
#             for k2 in range(k):
#                 sim_map[k1][k2] = cos(candidate_prototype[k1], support_instance_prototype[k2])
#
#         similarity_map.append(np.max(sim_map, axis=1))
#     similarity_map = np.array(similarity_map) # [bs,5]
#     similarity_map = nn.functional.softmax(similarity_map, dim=1)
#     pq = np.mean(candidate_query_pro * similarity_map, dim=[1,2]) # [bs, channel]
#     return pq

if __name__ == "__main__":
    bs = 2
    shot = 5
    channel = 64
    h = 50
    w = 50
    ms = torch.randint(0,2,(bs, shot, h, w), dtype=torch.float32)  # mask_support
    fs = torch.rand((bs, shot, channel, h, w), dtype=torch.float32)  # feature_support
    fq = torch.rand((bs, channel, h, w), dtype=torch.float32)  # feature_query

    pq = query_prototype_generation(ms,fs,fq)
    print(pq.shape)
    print(1)