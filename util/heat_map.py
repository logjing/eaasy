import os
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

def heat_map(image, mode='show', save_path=""): #将灰度图转化成热力图
    # image[image!=0] = 255 #用于mask生成
    # 应用颜色映射
    image1 = cv2.applyColorMap(image, cv2.COLORMAP_JET) # cv2.COLORMAP_JET
    # 显示热力图
    if mode=='show':
        plt.imshow(image1)
        plt.show()
    # 保存热力图（如果需要）

    height, width, _ = image1.shape
    for i in range(height):
        for j in range(width):
            if np.array_equal(image1[i, j], [128, 0, 0]):
                image1[i, j] = [0, 0, 0]
    # image1 = np.where((image == [128, 0, 0]).all(axis=2), [0, 0, 0], image1)

    if mode!='show':
        # image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        cv2.imwrite(save_path, image1)

def read_save_img(data_list, save_dir): #根据data_list读取图像，然后根据save_dir保存图像
    #E:\dataset\MSCOCO2014\val2014
    for i in data_list:
        img_path,mask_path = i
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path)
        name = img_path.split("_")[-1].replace(".jpg", ".png")
        cv2.imwrite(os.path.join(save_dir, "input",name), img)
        cv2.imwrite(os.path.join(save_dir, "mask", name), mask)

def plus_img(dir1, dir2, save_dir): #将两个文件夹中的图像叠加（通常是原图 + pred_mask）
    for name in os.listdir(dir1):
        print(name)
        img_path = os.path.join(dir1, name)
        mask_path = os.path.join(dir2, name)
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path)
        h,w = img.shape[:2]
        # mask = F.interpolate(mask, size=(h, w), mode='bilinear', align_corners=True)
        mask = cv2.resize(mask, (w,h), interpolation=cv2.INTER_LINEAR)
        out = img//2 + mask//2
        cv2.imwrite(os.path.join(save_dir, name), out)

def create_result_file(pic_num_list):
    path = r"C:\Users\Administrator\Desktop\result\HDMout"
    # path = r"C:\Users\Administrator\Desktop\result\pascal"
    out_dir = r"C:\Users\Administrator\Desktop\result\make_pic"
    for i in pic_num_list:
        name = str(i).zfill(12) + ".png"
        print(name)
        input = cv2.imread(os.path.join(path,"input", name))
        ours =  cv2.imread(os.path.join(path, "pred_input", name))
        gt =  cv2.imread(os.path.join(path, "masked_input", name))
        hdm =  cv2.imread(os.path.join(path, "official_pred_input", name))
        cv2.imwrite(os.path.join(out_dir, "input", name), input)
        cv2.imwrite(os.path.join(out_dir, "ours", name), ours)
        cv2.imwrite(os.path.join(out_dir, "HDM", name), hdm)
        cv2.imwrite(os.path.join(out_dir, "gt", name), gt)

# if __name__ == "__main__": #根据list把里面的图片和mask分别保存到一个文件夹中
#     with open(r'C:\Users\Administrator\Desktop\result\pascal\data_list.pkl', 'rb') as f:
#         loaded_list = pickle.load(f)
#     read_save_img(loaded_list, r"C:\Users\Administrator\Desktop\result\pascal")

# if __name__ == "__main__": #把灰度图变成热力图
#     path = r"C:\Users\Administrator\Desktop\result\HDMout"
#     dir = os.path.join(path, "my_output") # 输入的灰度图位置 mask | my_output
#     save_dir = os.path.join(path, "heat_myoutput") # 输出的染色图位置
#     for name in os.listdir(dir):
#         print(name)
#         image = cv2.imread(os.path.join(dir,name), cv2.IMREAD_GRAYSCALE)
#         image[image > 100] = 255 #当my_output时开启
#         image[image <= 100] = 0  # 当my_output时开启
#         heat_map(image, mode='save', save_path=os.path.join(save_dir, name))

if __name__ == "__main__": # 将两个文件夹中的input和mask叠加，保存到另一个文件夹中
    path = r"C:\Users\Administrator\Desktop\result\HDMout"
    plus_img(os.path.join(path,"input"), os.path.join(path,"heat_myoutput"), os.path.join(path,"pred_input"))

# if __name__ == "__main__":
#     create_result_file([3849,6033,6180,8646,9041,10693,12269,12731,16228,18462,20161,13300,13414,16180])