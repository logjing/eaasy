import clip

# classes = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat','traffic light',
#            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
#            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
#            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
#            'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
#            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed',
#            'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
#            'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
classes = ["a picture of dog",
           "a picture of cat"]
Clip, _ = clip.load("RN101", device="cpu")
text = clip.tokenize(classes).to("cpu")
embeddings = Clip.encode_text(text)
embeddings = embeddings.detach()
print(embeddings.shape)

"""
['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
RN50: [80,1024]
RN101: [80,512]
RN50*4: [80,640]
RN50*16: [80,768]
RN50*64: [80,1024]
ViT-B/32: [80,512]
ViT-B/16: [80,512]
ViT-L/14: [80,768]
ViT-L/14@336px: [80,768]

"""