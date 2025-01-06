import os
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import os
import shutil

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
    out_dir = r"C:\Users\Administrator\Desktop\result\make_pic"
    delete_images_in_folder(os.path.join(out_dir, "input"))
    delete_images_in_folder(os.path.join(out_dir, "ours"))
    delete_images_in_folder(os.path.join(out_dir, "HDM"))
    delete_images_in_folder(os.path.join(out_dir, "gt"))
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


def create_combined_image(folder_paths, output_path):
    # Initialize an empty list to hold all images
    all_images = []

    # Iterate over each folder path
    for folder_path in folder_paths:
        # Check if the folder exists
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"The folder {folder_path} does not exist.")

        # Get the image files in the folder
        image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                       f.lower().endswith(('png', 'jpg', 'jpeg'))]

        # Ensure there are exactly 3 images in the folder
        # if len(image_files) != 3:
        #     raise ValueError(f"The folder {folder_path} does not contain exactly 3 images.")

        # Open the images
        images_in_folder = [Image.open(img) for img in image_files]

        # Append the list of images in the folder to the main list
        all_images.append(images_in_folder)

    # Determine the dimensions of the final image
    num_rows = len(all_images)
    num_cols = 3
    widths, heights = zip(*(i.size for row in all_images for i in row))

    # Calculate the total width and height for the new image
    total_width = sum(widths)
    max_height = max(heights)

    # Create a new blank image with the calculated dimensions
    combined_image = Image.new('RGB', (total_width, max_height * num_rows))

    # Paste the images onto the combined image
    y_offset = 0
    for row_idx, images_row in enumerate(all_images):
        x_offset = 0
        for img in images_row:
            combined_image.paste(img, (x_offset, y_offset))
            x_offset += img.width
        y_offset += max_height

    # Save the final combined image
    combined_image.save(output_path)

def delete_images_in_folder(folder_path):
    # 确保传入的路径是一个文件夹
    if not os.path.isdir(folder_path):
        raise ValueError(f"{folder_path} is not a valid directory")

    # 定义图片文件的扩展名
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 获取文件的完整路径
        file_path = os.path.join(folder_path, filename)

        # 检查文件是否是图片文件
        if os.path.isfile(file_path) and any(filename.lower().endswith(ext) for ext in image_extensions):
            try:
                # 删除文件
                os.remove(file_path)
                print(f"Deleted {file_path}")
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")

# 示例使用
def my_combine(fold:list):
    height = 640
    width = 768
    fold_num = len(fold) #文件夹的数量
    pic_num = len(os.listdir(fold[0])) #设置图片的数量
    combined_img = np.zeros((height * fold_num, width * pic_num, 3))
    for i in range(fold_num):
        for j in range(pic_num):
            pic_name_list = os.listdir(fold[i])
            part_img = cv2.imread(os.path.join(fold[i], pic_name_list[j]))
            part_img = cv2.resize(part_img, (width, height))
            combined_img[i*height:(i+1)*height, j*width:(j+1)*width] = part_img
    cv2.imwrite(r"C:\Users\Administrator\Desktop\result\make_pic\output_img.png", combined_img)


# if __name__ == "__main__": #根据list把里面的图片和mask分别保存到一个文件夹中
#     with open(r'C:\Users\Administrator\Desktop\result\pascal\data_list.pkl', 'rb') as f:
#         loaded_list = pickle.load(f)
#     read_save_img(loaded_list, r"C:\Users\Administrator\Desktop\result\pascal")

# if __name__ == "__main__": #把灰度图变成热力图
#     path = r"C:\Users\Administrator\Desktop\result\HDMout"
#     dir = os.path.join(path, "mask") # 输入的灰度图位置 mask | my_output
#     save_dir = os.path.join(path, "mask_heat") # 输出的染色图位置
#     for name in os.listdir(dir):
#         print(name)
#         image = cv2.imread(os.path.join(dir,name), cv2.IMREAD_GRAYSCALE)
#         # image[image > 100] = 255 #当my_output时开启
#         # image[image <= 100] = 0  # 当my_output时开启
#         heat_map(image, mode='save', save_path=os.path.join(save_dir, name))

# if __name__ == "__main__": # 将两个文件夹中的input和mask叠加，保存到另一个文件夹中
#     path = r"C:\Users\Administrator\Desktop\result\HDMout"
#     plus_img(os.path.join(path,"input"), os.path.join(path,"mask_heat"), os.path.join(path,"masked_input"))

# if __name__ == "__main__":
#     # create_result_file([139, 143, 962, 1228, 1374, 6033, 6180, 7938, 9041, 12269, 12731, 21452, 24343, 25138,
#     #                     25551, 28236, 29913, 30413, 35282, 35682, 39628, 41603, 44054, 47394, 48014, 49258, 50232,
#     #                     51089, 52470, 52871, 57244, 60093, 64516])
#     create_result_file([1228, 1374, 6180, 9041, 12731, 25551])
#     # [3849, 6033, 6180, 8646, 9041, 10693, 12269, 12731, 16228, 18462, 20161, 13300, 13414, 16180]

if __name__ == "__main__":
    # Example usage:
    path = r"C:\Users\Administrator\Desktop\result\make_pic"
    folder_paths = [
        os.path.join(path, "gt"),
        os.path.join(path, "HDM"),
        os.path.join(path, "ours")
        # Add more folder paths as needed
    ]
    texts = [
        "gt",
        "HDM",
        "ours"
    ]
    output_path = os.path.join(path, "output_image.png")
    my_combine(folder_paths)
    # create_combined_image(folder_paths, output_path)
