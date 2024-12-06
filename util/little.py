import os
import cv2
abs_path = r"E:\project\insight1\HDMNet\exp\coco\HDMNet\split0\resnet50\official_path.txt"
with open(abs_path, 'r') as file:
    while True:
        line = file.readline()
        if not line:  # 如果读取到文件末尾，则line为空字符串
            break
        # 在这里处理每一行
        num, path = line.split(" ")
        predict_path = r"E:\project\insight1\HDMNet\exp\coco\HDMNet\split0\resnet50\official_result"
        predict_path = os.path.join(predict_path, num+".png")
        ori_path = os.path.join(r"E:\dataset\MSCOCO2014\val2014", path[:-1])
        ori_img = cv2.imread(ori_path)
        #H,W = ori_img.shape[:2]

        pred_img = cv2.imread(predict_path)
        #pred_img = cv2.resize(pred_img, (W,H), interpolation=cv2.INTER_NEAREST)

        save_path = r"E:\project\insight1\HDMNet\exp\coco\HDMNet\split0\resnet50\official_compare"
        save_ori = os.path.join(save_path, num + "ori.png")
        save_pred = os.path.join(save_path, num + "pred.png")

        cv2.imwrite(save_ori, ori_img)
        cv2.imwrite(save_pred, pred_img)

        print(num, path)