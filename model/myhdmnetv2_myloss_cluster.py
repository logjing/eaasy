"""
配套trainv2_myloss使用
改了返回值，添加全局的query_feat损失
616行: final_out.max(1)[1], query_feat, main_loss + aux_loss1, distil_loss / 3, aux_loss2
out, query_feat, loss1, loss2, loss3
"""

import torch
from torch import nn
import torch.nn.functional as F
from model.Transformer import Transformer
import model.resnet as models
import model.vgg as vgg_models
from model.PSPNet import OneModel as PSPNet
from einops import rearrange
from torch.nn.functional import cosine_similarity as cosine
from torch.nn.functional import softmax
import clip
import time
from torch.cuda.amp import autocast,GradScaler
import pdb
from sklearn.cluster import KMeans
import numpy as np

cos = nn.CosineSimilarity(dim=1,eps=1e-6)

def get_query_prototype(supp_prototype, supp_feat):
    """
    Args:
        C=64
        H,W=80
        一共6400个样本点
        supp_prototype: [bs,C]
        supp_feat: [bs,C,H,W]
    Returns:
    """
    n_clusters = 4
    # start = time.time()
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(supp_feat)
    # print(time.time() - start)
    prototype = []
    for i in range(len(np.unique(kmeans.labels_))):
        prototype.append(np.mean(supp_feat[kmeans.labels_ == i], axis=0)) #计算聚类中心
    score = []
    for i in range(len(prototype)): #计算每个聚类中心的score
        score.append(nn.CosineSimilarity(dim=0, eps=1e-6)
                     (torch.tensor(prototype[i]), supp_prototype.detach().cpu()[0, :, 0, 0]))
    Max = 0
    Max_index = 0
    for i in range(len(prototype)):
        if score[i] > Max:
            Max = score[i]
            Max_index = i
    return prototype[Max_index]

class GIG(nn.Module):
    def __init__(self, in_channels=556, out_channels=256, hidden_size=256):
        super(GIG, self).__init__()

        def block(in_feat, out_feat):
            layers = [nn.Linear(in_feat, out_feat)]
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout(p=0.2))
            return layers

        if hidden_size:
            self.model = nn.Sequential(
                *block(in_channels, hidden_size),
                nn.Linear(hidden_size, out_channels),
                nn.Dropout(0.3)
            )
        else:
            self.model = nn.Sequential(
                nn.Linear(in_channels, out_channels),)
        for m in self.modules():
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, embeddings):
        return self.model(embeddings)

def cos_weighted_prototype(candidate_query_pro, instance_prototype):
    """
    :param candidate_query_pro: [bs, n, 256]
    :param instance_prototype: 支撑集原型 [bs, 256] / [bs, 256, 1, 1]
    :param n:
    :return:
    """
    bs, n = candidate_query_pro.shape[:2]

    weight = []  # [bs, n]
    query_prototype = torch.zeros(instance_prototype.shape[:2]).cuda()
    for i in range(bs):
        for j in range(n):
            weight.append(cosine(candidate_query_pro[i][j], instance_prototype[i,:], dim=0,eps=1e-6))
    weight = torch.tensor(weight).view(bs, n)
    weight = softmax(weight,dim=1).cuda()   # [bs, n]
    for i in range(bs):
        for j in range(n):
            query_prototype[i] += weight[i][j] * candidate_query_pro[i][j]
    return query_prototype

def cal_my_loss(query_prototype, negative_query_pro, background_mask):
    """
    :param query_prototype: shape(2,64)
    :param negative_query_feat: list[shape(2,64)]
    :return:
    """
    loss1 = torch.zeros(query_prototype.shape[0]).cuda()
    n = len(negative_query_pro)
    for i in negative_query_pro:
        loss1 += -cos(query_prototype, i.squeeze())
    loss1 /= n
    loss2 = -cos(query_prototype, background_mask.squeeze())
    loss = torch.mean(loss1) + torch.mean(loss2)
    margin = 3
    #margin = torch.ones(query_prototype.shape[0]).cuda() * 3
    # import pdb;pdb.set_trace()
    compare_loss = torch.relu(loss + margin)
    return compare_loss

def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat


def get_similarity(q, s, mask):
    with autocast(enabled=False):
        q = q.float()
        s = s.float()
        mask = mask.float()
        if len(mask.shape) == 3:
            mask = mask.unsqueeze(1)
        mask = F.interpolate((mask == 1).float(), q.shape[-2:])
        cosine_eps = 1e-6
        s = s * mask
        bsize, ch_sz, sp_sz, _ = q.size()[:]
        tmp_query = q
        tmp_query = tmp_query.contiguous().view(bsize, ch_sz, -1)
        tmp_query_norm = torch.norm(tmp_query, 2, 1, True)
        tmp_supp = s
        tmp_supp = tmp_supp.contiguous().view(bsize, ch_sz, -1).contiguous()
        tmp_supp = tmp_supp.contiguous().permute(0, 2, 1).contiguous()
        tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)
        similarity = torch.bmm(tmp_supp, tmp_query) / (torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)
        similarity = similarity.max(1)[0].view(bsize, sp_sz * sp_sz)
        similarity = similarity.view(bsize, 1, sp_sz, sp_sz)
    return similarity


def get_gram_matrix(fea):
    b, c, h, w = fea.shape
    fea = fea.reshape(b, c, h * w)  # C*N
    fea_T = fea.permute(0, 2, 1)  # N*C
    fea_norm = fea.norm(2, 2, True)
    fea_T_norm = fea_T.norm(2, 1, True)
    gram = torch.bmm(fea, fea_T) / (torch.bmm(fea_norm, fea_T_norm) + 1e-7)  # C*C
    return gram


def get_vgg16_layer(model):
    layer0_idx = range(0, 7)
    layer1_idx = range(7, 14)
    layer2_idx = range(14, 24)
    layer3_idx = range(24, 34)
    layer4_idx = range(34, 43)
    layers_0 = []
    layers_1 = []
    layers_2 = []
    layers_3 = []
    layers_4 = []
    for idx in layer0_idx:
        layers_0 += [model.features[idx]]
    for idx in layer1_idx:
        layers_1 += [model.features[idx]]
    for idx in layer2_idx:
        layers_2 += [model.features[idx]]
    for idx in layer3_idx:
        layers_3 += [model.features[idx]]
    for idx in layer4_idx:
        layers_4 += [model.features[idx]]
    layer0 = nn.Sequential(*layers_0)
    layer1 = nn.Sequential(*layers_1)
    layer2 = nn.Sequential(*layers_2)
    layer3 = nn.Sequential(*layers_3)
    layer4 = nn.Sequential(*layers_4)
    return layer0, layer1, layer2, layer3, layer4

# classes = ['background','aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow',
#         'diningtable','dog','horse','motorbike','person','potted-plant','sheep','sofa','train','tv/monitor']
# classes = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat','traffic light',
#            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
#            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
#            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
#            'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
#            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed',
#            'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
#            'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
#
# text = clip.tokenize(classes).to("cuda")
# embeddings = Clip.encode_text(text)
# embeddings = embeddings.detach()
class SelfAttention(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=3,stride=1,padding=1)
        self.key_conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=3,stride=1,padding=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3,stride=1,padding=1)
        # self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()

        # Query, Key, Value
        query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, width * height)
        value = self.value_conv(x).view(batch_size, -1, width * height)

        # Attention
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)

        # Apply attention to value
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)

        # Residual connection
        out = F.interpolate(out, x.size()[-1], mode='bilinear') # bilinear, nearest
        out = out + x # self.gamma * out + x
        return out


class CrossAttention2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CrossAttention2D, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, query_img, key_img, value_img):
        """
        query_img: 查询图像的特征图，形状为 (batch_size, in_channels, height, width)
        key_img: 键图像的特征图，形状为 (batch_size, in_channels, height, width)
        value_img: 值图像的特征图，形状为 (batch_size, in_channels, height, width)
        """
        # 计算query和key的特征
        batch_size, _, height, width = query_img.size()
        query = self.query_conv(query_img).view(batch_size, -1, width * height).permute(0, 2, 1)  # (batch_size, N, C)
        key = self.key_conv(key_img).view(batch_size, -1, width * height)  # (batch_size, C, N)

        # 计算注意力权重
        attention_scores = torch.matmul(query, key)  # (batch_size, N, N)
        attention_scores = attention_scores / (key.size(-1) ** 0.5)  # 归一化
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch_size, N, N)

        # 应用注意力权重到value上
        # value = self.value_conv(value_img).view(batch_size, -1, width * height).permute(0, 2, 1)  # (batch_size, C, N)
        value = value_img.view(batch_size, -1, width * height).permute(0, 2, 1)
        output = torch.matmul(attention_weights, value)  # (batch_size, N, C)
        output = output.permute(0, 2, 1).view(batch_size, -1, height, width)  # (batch_size, C, height, width)

        # 应用可学习的gamma参数
        output = output + query.permute(0,2,1).view(batch_size, -1, height, width)

        return output


# batch_size = 4
# C = 64
# H = 32
# W = 32
#
# # 随机生成图像特征图
# query_img = torch.randn(batch_size, C, H, W)
# key_img = torch.randn(batch_size, C, H, W)
# value_img = torch.randn(batch_size, C, H, W)
#
# # 创建二维交叉注意力模块
# cross_attention_2d = CrossAttention2D(C, C // 8)
#
# # 执行前向传播
# output = cross_attention_2d(query_img, key_img, value_img)
# print(output.shape)  # 输出形状应为 (batch_size, C, H, W)

# class SimpleCNN(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(SimpleCNN, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(32, out_channels, kernel_size=3, padding=1)
#         self.attention = SelfAttention(out_channels)
#
#     def forward(self, x):
#         x = self.relu(self.conv1(x))
#         x = self.conv2(x)
#         x = self.attention(x)
#         return x

    # 示例使用


# # 假设输入图像大小为(3, 224, 224)
# input_image = torch.randn(1, 3, 224, 224).cuda()
#
# # 初始化CNN模型
# model = SimpleCNN(3, 32).cuda()
#
# # 前向传播
# output_feature_map = model(input_image)
#
# # 打印输出特征图的尺寸
# print(output_feature_map.shape)  # 输出应该是 (1, 128, 224, 224)

class OneModel(nn.Module):
    def __init__(self, args, cls_type=None):
        super(OneModel, self).__init__()

        #self.teacher_output = torch.load("teacher_feat.pth")
        #self.text = torch.load("embed.pth")
        self.cls_type = cls_type  # 'Base' or 'Novel'
        self.dataset = args.data_set
        if self.dataset == 'pascal':
            self.base_classes = 15
        elif self.dataset == 'coco':
            self.base_classes = 60
        self.low_fea_id = args.low_fea[-1]

        assert args.layers in [50, 101, 152]
        from torch.nn import BatchNorm2d as BatchNorm
        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
        self.shot = args.shot
        self.vgg = args.vgg
        models.BatchNorm = BatchNorm

        PSPNet_ = PSPNet(args)
        new_param = torch.load(args.pre_weight, map_location=torch.device('cpu'))['state_dict']
        try:
            PSPNet_.load_state_dict(new_param)
        except RuntimeError:
            # dict2 = PSPNet_.state_dict().keys()
            # new_dict1 = {}
            # # 遍历dict1的项，并使用dict2的key作为新key
            # for key1, value in new_param.items():
            #     key2 = list(dict2)[list(new_param.keys()).index(key1)]  # 这种方法依赖于顺序且效率不高
            #     new_dict1[key2] = value
            # PSPNet_.load_state_dict(new_dict1)
            # print(1)
            for key in list(new_param.keys()):
                new_param[key[7:]] = new_param.pop(key)
            PSPNet_.load_state_dict(new_param)
        self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = PSPNet_.layer0, PSPNet_.layer1, PSPNet_.layer2, PSPNet_.layer3, PSPNet_.layer4
        self.ppm = PSPNet_.ppm
        self.cls = nn.Sequential(PSPNet_.cls[0], PSPNet_.cls[1])
        self.base_learnear = nn.Sequential(PSPNet_.cls[2], PSPNet_.cls[3], PSPNet_.cls[4])

        # TODO:完成cross_attention和self_attention创建
        # self.CrossAttention(q,k,v),  self.SelfAttention(x)
        self.CrossAttention = CrossAttention2D(in_channels=64, out_channels=64) # [bs, inchannel, h, w] -> [bs, outchannel, h, w]
        self.SelfAttention = SelfAttention(in_channels=64, hidden_channels=128) # [bs, ]

        self.student_net = nn.Sequential(
            nn.Linear(in_features=77, out_features=128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=128, out_features=512),
            nn.Dropout(p=0.2)
        )

        if self.vgg:
            fea_dim = 512 + 256
        else:
            fea_dim = 1024 + 512

        self.down_supp = nn.Sequential(
            nn.Conv2d(fea_dim, 256, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )

        self.down_query = nn.Sequential(
            nn.Conv2d(fea_dim, 256, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )

        self.query_merge = nn.Sequential(
            nn.Conv2d(512 + 2, 64, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )

        self.supp_merge = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )
        self.transformer = Transformer(shot=self.shot)

        self.gram_merge = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        self.gram_merge.weight = nn.Parameter(torch.tensor([[1.0], [0.0]]).reshape_as(self.gram_merge.weight))
        # Learner Ensemble
        self.cls_merge = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        self.cls_merge.weight = nn.Parameter(torch.tensor([[1.0], [0.0]]).reshape_as(self.cls_merge.weight))
        self.GIG = GIG(in_channels=512 + 256, out_channels=256, hidden_size=512)
        #self.distill_net = teacher_student_net


        # K-Shot Reweighting
        if args.shot > 1:
            self.kshot_trans_dim = args.kshot_trans_dim
            if self.kshot_trans_dim == 0:
                self.kshot_rw = nn.Conv2d(self.shot, self.shot, kernel_size=1, bias=False)
                self.kshot_rw.weight = nn.Parameter(torch.ones_like(self.kshot_rw.weight) / args.shot)
            else:
                self.kshot_rw = nn.Sequential(
                    nn.Conv2d(self.shot, self.kshot_trans_dim, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.kshot_trans_dim, self.shot, kernel_size=1))

    def forward(self, x, y_m=None, y_b=None, s_x=None, s_y=None, cat_idx=None):
        # start1 = time.time()
        bs, channel, h, w = x.shape
        _, _, query_feat_2, query_feat_3, query_feat_4, query_feat_5 = self.extract_feats(x)

        if self.vgg:
            query_feat_2 = F.interpolate(query_feat_2, size=(query_feat_3.size(2), query_feat_3.size(3)),
                                         mode='bilinear', align_corners=True)
        query_feat = torch.cat([query_feat_3, query_feat_2], 1)
        query_feat = self.down_query(query_feat)

        mask = rearrange(s_y, "b n h w -> (b n) 1 h w")
        mask = (mask == 1).float()  # 将前景部分分割出来
        # ================================= 创建与当前输入图片的前景无关但与在coco80个类别里的 negative_mask ====================================================
        negative_mask = []
        background_mask = (y_b==0).float()
        for i in y_b.unique():
            i=i.cpu()
            if (i not in [0,255]) and i-1 not in cat_idx[0]:
                n_mask = (y_b==i).float()
                negative_mask.append(n_mask)


        s_x = rearrange(s_x, "b n c h w -> (b n) c h w")
        supp_feat_0, supp_feat_1, supp_feat_2, supp_feat_3, supp_feat_4, supp_feat_5 = self.extract_feats(s_x, mask)
        if self.vgg:
            supp_feat_2 = F.interpolate(supp_feat_2, size=(supp_feat_3.size(2), supp_feat_3.size(3)), mode='bilinear',
                                        align_corners=True)
        supp_feat = torch.cat([supp_feat_3, supp_feat_2], 1)
        supp_feat = self.down_supp(supp_feat)
        supp_feat_prototype = Weighted_GAP(supp_feat, \
                                     F.interpolate(mask, size=(supp_feat_3.size(2), supp_feat_3.size(3)),
                                                   mode='bilinear', align_corners=True))
        supp_feat_bin = supp_feat_prototype.repeat(1, 1, supp_feat.shape[-2], supp_feat.shape[-1])
        supp_feat_item = eval('supp_feat_' + self.low_fea_id)
        supp_feat_item = rearrange(supp_feat_item, "(b n) c h w -> b n c h w", n=self.shot)
        supp_feat_list = [supp_feat_item[:, i, ...] for i in range(self.shot)]

        if self.shot == 1:
            similarity2 = get_similarity(query_feat_4, supp_feat_4, s_y)
            similarity1 = get_similarity(query_feat_5, supp_feat_5, s_y)
        else:
            mask = rearrange(mask, "(b n) c h w -> b n c h w", n=self.shot)
            supp_feat_4 = rearrange(supp_feat_4, "(b n) c h w -> b n c h w", n=self.shot)
            supp_feat_5 = rearrange(supp_feat_5, "(b n) c h w -> b n c h w", n=self.shot)
            similarity1 = [get_similarity(query_feat_5, supp_feat_5[:, i, ...], mask=mask[:, i, ...]) for i in
                           range(self.shot)]
            similarity2 = [get_similarity(query_feat_4, supp_feat_4[:, i, ...], mask=mask[:, i, ...]) for i in
                           range(self.shot)]
            mask = rearrange(mask, "b n c h w -> (b n) c h w")
            # supp_feat_4 = rearrange(supp_feat_4, "b n c h w -> (b n) c h w")
            # supp_feat_5 = rearrange(supp_feat_5, "b n c h w -> (b n) c h w")
            similarity2 = torch.stack(similarity2, dim=1).mean(1)
            similarity1 = torch.stack(similarity1, dim=1).mean(1)
        similarity = torch.cat([similarity1, similarity2], dim=1)

        supp_feat = self.supp_merge(torch.cat([supp_feat, supp_feat_bin], dim=1))
        supp_prototype_small = Weighted_GAP(supp_feat, F.interpolate(mask, size=(supp_feat.size(2), supp_feat.size(3)),
                                                         mode='bilinear', align_corners=True))





        # nn.CosineSimilarity(dim=1,eps=1e-6)(torch.tensor(prototype[0]), supp_prototype_small.detach().cpu()[0,:,0,0])

        supp_feat_prototype = Weighted_GAP(supp_feat,F.interpolate(mask, size=(supp_feat_3.size(2), supp_feat_3.size(3)),
                                                         mode='bilinear', align_corners=True))


        supp_feat_bin = rearrange(supp_feat_bin, "(b n) c h w -> b n c h w", n=self.shot)
        supp_feat_bin = torch.mean(supp_feat_bin, dim=1)
        query_feat = self.query_merge(torch.cat([query_feat, supp_feat_bin, similarity * 10], dim=1))

        query_feat0= query_feat.view(bs, 64, -1).permute(0, 2, 1).detach().cpu().numpy()
        query_prototype = []
        for i in range(bs):
            query_feat0 = query_feat0[i]
            query_prototype.append(get_query_prototype(supp_prototype_small, query_feat0))
        # query_prototype.append(get_query_prototype(supp_prototype_small, query_feat1))
        query_prototype = torch.tensor(np.array(query_prototype))

        # start2 = time.time()

        # TODO 1.使用query_prototype替换supp_feat_bin p201行
        # embed = torch.stack([torch.load("./embed.pth")] * bs, dim=0)  # [bs, 80, 256]
        # teacher_output = self.teacher_net(text)

        # embed = self.student_net(self.text.float())
        # # start3 = time.time()
        # # TODO 这里的log_softmax主要是为了取对数
        # temperature = 1
        # kd_loss = nn.functional.kl_div(F.log_softmax(embed / temperature, dim=1),
        #                                F.softmax(self.teacher_output / temperature, dim=1), reduction='batchmean') * (temperature ** 2)
        # import pdb;pdb.set_trace()
        # kd_loss /= (bs * 8 *512)
        # start4 = time.time()
        # channel = 256
        # instance_prototype = supp_feat_prototype.view(bs, channel)
        # #instance_prototype_1 = instance_prototype.unsqueeze(1).expand(bs, embed.shape[1], channel)
        # embed = embed.unsqueeze(0).expand(bs,-1,-1)# 复制bs份
        # supp_tensor_tmp = nn.AdaptiveAvgPool2d((8, 10)).cuda()(supp_feat).reshape(bs,256,-1).permute(0,2,1)
        # anchors = self.GIG(torch.cat((embed, supp_tensor_tmp), dim=-1))
        # row = list(range(embed.shape[0]))
        # general_prototype = anchors[row, cat_idx[0]].unsqueeze(-1).unsqueeze(-1)  # [bs, 256]
        # score = cos(general_prototype, query_feat)
        # prototype = query_feat.permute(1, 0, 2, 3) * score
        # prototype = prototype.permute(1, 0, 2, 3)
        # bs, channel, h1, w1 = prototype.shape
        #
        # # start5 = time.time()
        #
        # candidate_query_pro = []
        # candidate_query_pro.append(torch.mean(prototype[:, :, 0:h1 // 2, 0:w1 // 2], dim=[2, 3]))
        # candidate_query_pro.append(torch.mean(prototype[:, :, h1 // 2:h1, 0:w1 // 2], dim=[2, 3]))
        # candidate_query_pro.append(torch.mean(prototype[:, :, 0:h1 // 2, w1 // 2:w1], dim=[2, 3]))
        # candidate_query_pro.append(torch.mean(prototype[:, :, h1 // 2:h1, w1 // 2:w1], dim=[2, 3]))
        # candidate_query_pro.append(
        #     torch.mean(prototype[:, :, h1 // 2 - h1 // 4: h1 // 2 + h1 // 4, w1 // 2 - w1 // 4:w1 // 2 + w1 // 4],
        #                dim=[2, 3]))
        # candidate_query_pro = torch.cat(candidate_query_pro, dim=0).view(5, bs, channel).permute(1, 0, 2)
        # query_prototype = cos_weighted_prototype(candidate_query_pro, instance_prototype)

        # start6 = time.time()
        # TODO 2.使用query_prototype


        # meta_out, weights = self.transformer(query_feat, supp_feat, mask, similarity)

        # ============================================== negative_query_prototype ====================================================
        # negative_query_pro = []
        # for n_mask in negative_mask:
        #     if n_mask is not None:
        #         negative_query_pro.append(Weighted_GAP(query_feat, F.interpolate(n_mask.unsqueeze(1), size=(query_feat.size(2), query_feat.size(3)),mode='bilinear', align_corners=True)))
        # background_mask = Weighted_GAP(query_feat, F.interpolate(background_mask.unsqueeze(1),
        #                                                                  size=(query_feat.size(2), query_feat.size(3)),
        #                                                                  mode='bilinear', align_corners=True))
        # ============================================== 计算compare_loss,最小化类内距离,最大化类间距离 ====================================================
        # query_prototype: [2, 64], negative_query_feat: list(2, 64)
        # if negative_query_pro == []:
        #     compare_loss = 0 #torch.zeros(query_prototype.shape[0]).cuda()
        #     # compare_loss.requires_grad=False
        # else:
        #     compare_loss = cal_my_loss(query_prototype, negative_query_pro, background_mask)
        #
        # # start7 = time.time()
        #
        # query_prototype = query_prototype.unsqueeze(-1).unsqueeze(-1).expand(bs, channel, h1, w1)

        if (self.shot == 1):
            query_feat = self.CrossAttention(query_feat, supp_feat,
                                             F.interpolate(mask, query_feat.shape[-2], mode="nearest"))
        else:
            tmp = torch.zeros_like(query_feat)
            for index in range(self.shot):
                tmp += self.CrossAttention(query_feat, supp_feat[bs * index:bs * (index + 1)],
                                           F.interpolate(mask[bs * index:bs * (index + 1)], query_feat.shape[-2],
                                                         mode="nearest"))
            query_feat = tmp / self.shot

        query_feat = self.CrossAttention(query_feat, supp_feat, F.interpolate(mask, query_feat.shape[-2], mode="nearest"))
        supp_feat = self.SelfAttention(supp_feat)
        query_prototype = query_prototype.unsqueeze(2).unsqueeze(3).repeat(1,1,80,80).cuda()
        query_prototype.requires_grad = True
        #meta_out, weights = self.transformer(query_feat, supp_feat, mask, similarity)
        meta_out, weights = self.transformer(query_feat, query_prototype, mask, similarity)
        base_out = self.base_learnear(query_feat_5)

        meta_out_soft = meta_out.softmax(1)
        base_out_soft = base_out.softmax(1)

        # start8 = time.time()

        # K-Shot Reweighting
        bs = x.shape[0]
        que_gram = get_gram_matrix(eval('query_feat_' + self.low_fea_id))  # [bs, C, C] in (0,1)
        norm_max = torch.ones_like(que_gram).norm(dim=(1, 2))
        est_val_list = []
        for supp_item in supp_feat_list:
            supp_gram = get_gram_matrix(supp_item)
            gram_diff = que_gram - supp_gram
            est_val_list.append((gram_diff.norm(dim=(1, 2)) / norm_max).reshape(bs, 1, 1, 1))  # norm2
        est_val_total = torch.cat(est_val_list, 1)  # [bs, shot, 1, 1]
        if self.shot > 1:
            val1, idx1 = est_val_total.sort(1)
            val2, idx2 = idx1.sort(1)
            weight = self.kshot_rw(val1)
            idx3 = idx1.gather(1, idx2)
            weight = weight.gather(1, idx3)
            weight_soft = torch.softmax(weight, 1)
        else:
            weight_soft = torch.ones_like(est_val_total)
        est_val = (weight_soft * est_val_total).sum(1, True)  # [bs, 1, 1, 1]

        # Following the implementation of BAM ( https://github.com/chunbolang/BAM )
        meta_map_bg = meta_out_soft[:, 0:1, :, :]
        meta_map_fg = meta_out_soft[:, 1:, :, :]
        # start9 = time.time()
        if self.training and self.cls_type == 'Base':
            # c_id_array = torch.arange(self.base_classes + 1, device='cuda')
            base_map_list = []

            for b_id in range(bs):
                # start = time.time()
                c_id = cat_idx[0][b_id] + 1
                # c_mask = (c_id_array != 0) & (c_id_array != c_id)
                #tmp = base_out_soft.sum(1,True)
                # x = (base_out_soft.sum(1,True)[b_id] - base_out_soft[b_id][0] - base_out_soft[b_id][c_id]).unsqueeze(0)
                # x = base_out_soft[b_id, c_mask, :, :].unsqueeze(0).sum(1, True)
                # print("neibu", time.time() - start)
                base_map_list.append( (base_out_soft.sum(1,True)[b_id] - base_out_soft[b_id][0] - base_out_soft[b_id][c_id]).unsqueeze(0) )

            base_map = torch.cat(base_map_list, 0)

        else:
            base_map = base_out_soft[:, 1:, :, :].sum(1, True)

        est_map = est_val.expand_as(meta_map_fg)

        # start10 = time.time()

        meta_map_bg = self.gram_merge(torch.cat([meta_map_bg, est_map], dim=1))
        meta_map_fg = self.gram_merge(torch.cat([meta_map_fg, est_map], dim=1))

        merge_map = torch.cat([meta_map_bg, base_map], 1)
        merge_bg = self.cls_merge(merge_map)  # [bs, 1, 60, 60]
        # start11 = time.time()
        final_out = torch.cat([merge_bg, meta_map_fg], dim=1)

        # Output Part
        meta_out = F.interpolate(meta_out, size=(h, w), mode='bilinear', align_corners=True)
        base_out = F.interpolate(base_out, size=(h, w), mode='bilinear', align_corners=True)
        final_out = F.interpolate(final_out, size=(h, w), mode='bilinear', align_corners=True)
        # start12 = time.time()
        # Loss
        if self.training:
            main_loss = self.criterion(final_out, y_m.long())
            aux_loss1 = self.criterion(meta_out, y_m.long())
            aux_loss2 = self.criterion(base_out, y_b.long())
            # print("1", start2 - start1)
            # print("2", start3 - start2)
            # print("3", start4 - start3)
            # print("4", start5 - start4)
            # print("5", start6 - start5)
            # print("6", start7 - start6)
            # print("7", start8 - start7)
            # print("8", start9 - start8)
            # print("9", start10 - start9)
            # print("10", start11 - start10)
            # print("11", start12 - start11)
            #print("12", start13 - start12)


            weight_t = (y_m == 1).float()
            weight_t = torch.masked_fill(weight_t, weight_t == 0, -1e9)
            for i, weight in enumerate(weights):
                if i == 0:
                    distil_loss = self.disstil_loss(weight_t, weight)
                else:
                    distil_loss += self.disstil_loss(weight_t, weight)
                weight_t = weight.detach()
            if torch.isnan(main_loss).any() or torch.isnan(aux_loss1).any() or torch.isnan(aux_loss2).any():
                import pdb
                pdb.set_trace()

            return final_out.max(1)[1], query_feat, main_loss + aux_loss1, distil_loss / 3, aux_loss2
            #return final_out.max(1)[1], main_loss + aux_loss1, aux_loss2, compare_loss, kd_loss
        else:
            return final_out, meta_out, base_out

    def disstil_loss(self, t, s):
        if t.shape[-2:] != s.shape[-2:]:
            t = F.interpolate(t.unsqueeze(1), size=s.shape[-2:], mode='bilinear').squeeze(1)
        t = rearrange(t, "b h w -> b (h w)")
        s = rearrange(s, "b h w -> b (h w)")
        s = torch.softmax(s, dim=1)
        t = torch.softmax(t, dim=1)
        loss = t * torch.log(t + 1e-12) - t * torch.log(s + 1e-12)
        loss = loss.sum(1).mean()
        return loss

    def get_optim(self, model, args, LR):
        optimizer = torch.optim.AdamW(
            [
                {'params': model.parameters(), "lr": LR * 10},
                # {'params': model.transformer.mix_transformer.parameters()},
                # {'params': model.down_supp.parameters(), "lr": LR * 10},
                # {'params': model.down_query.parameters(), "lr": LR * 10},
                # {'params': model.supp_merge.parameters(), "lr": LR * 10},
                # {'params': model.query_merge.parameters(), "lr": LR * 10},
                # {'params': model.gram_merge.parameters(), "lr": LR * 10},
                # {'params': model.cls_merge.parameters(), "lr": LR * 10},
                # {}
            ], lr=LR, weight_decay=args.weight_decay, betas=(0.9, 0.999))
        return optimizer

    def freeze_modules(self, model):
        for param in model.layer0.parameters():
            param.requires_grad = False
        for param in model.layer1.parameters():
            param.requires_grad = False
        for param in model.layer2.parameters():
            param.requires_grad = False
        for param in model.layer3.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = False
        for param in model.ppm.parameters():
            param.requires_grad = False
        for param in model.cls.parameters():
            param.requires_grad = False
        for param in model.base_learnear.parameters():
            param.requires_grad = False

    def extract_feats(self, x, mask=None):
        results = []
        with torch.no_grad():
            if mask is not None:
                tmp_mask = F.interpolate(mask, size=x.shape[-2], mode='nearest')
                x = x * tmp_mask
            feat = self.layer0(x)
            results.append(feat)
            layers = [self.layer1, self.layer2, self.layer3, self.layer4]
            for _, layer in enumerate(layers):
                feat = layer(feat)
                results.append(feat.clone())
            feat = self.ppm(feat)
            feat = self.cls(feat)
            results.append(feat)
        return results
