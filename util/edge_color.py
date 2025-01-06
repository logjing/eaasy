import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_edge_darkening(image, threshold=127):
    # 将图像转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 二值化图像（前景为白色，背景为黑色）
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(binary, 50, 150)

    #将四周都设置成前景的边缘部分(可选)
    edges[0,:]=255
    edges[:,0]=255
    edges[:, -1] = 255
    edges[-1, :] = 255

    # 计算距离变换，得到每个点到最近前景边缘的距离
    dist_transform = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)

    # 归一化距离变换图像到0-255
    dist_normalized = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX)
    return dist_normalized

# 读取图像
image = cv2.imread(r'E:\dataset\MSCOCO2014\annotations\val2014\COCO_val2014_000000000196.png')
ori = cv2.imread(r"E:\dataset\MSCOCO2014\val2014\COCO_val2014_000000000196.jpg")

image[image!=0]=1
image*=255
# 根据实例区域内部的点到边缘的距离来染色，绘制热力图
result_image = apply_edge_darkening(image)
result_image = np.array(result_image, dtype=np.uint8)
image1 = cv2.applyColorMap(result_image, cv2.COLORMAP_JET)
show = ori//2 + image1//2
# show = image1
plt.axis('off')
plt.imshow(show)
# plt.show()
plt.savefig(fname=r"C:\Users\Administrator\Desktop\visio\edge_hard.png", bbox_inches='tight',  pad_inches = 0)


# Load an image