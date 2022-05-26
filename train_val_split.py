import os
import random
import shutil

# closed_imgs = os.listdir("/home/dell/SD_4G/train_eye/Closed")
# closed_count = len(closed_imgs)
# closed = random.sample(closed_imgs,int(closed_count*0.8))


# closed_imgs = os.listdir("/home/dell/SD_4G/train_eye/Closed")
# closed_count = len(closed_imgs)
# closed = random.sample(closed_imgs,int(closed_count*0.8))



base_path = r"/home/dell/SD_4G/train_eye/train/Closed"
move_path = r"/home/dell/SD_4G/train_eye/val/Closed"

closed_imgs = os.listdir(base_path)
closed_count = len(closed_imgs)
closed = random.sample(closed_imgs,int(closed_count*0.2))


# closed_imgs = os.listdir("/home/dell/SD_4G/train_eye/Closed")
# closed_count = len(closed_imgs)
# closed = random.sample(closed_imgs,int(closed_count*0.8))

for img in closed:
    shutil.move(os.path.join(base_path,img),os.path.join(move_path,img))


