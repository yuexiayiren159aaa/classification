import os


paths = os.listdir("/home/dell/SD_4G/imagenet2012/val")

with open("model_data/cls_imagenet_classes.txt",'w') as f:
    
    for name in paths:
        f.write(name+"\n")