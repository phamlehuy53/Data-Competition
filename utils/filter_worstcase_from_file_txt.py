import os
import shutil
# path_img="/home/fit/Pictures/DATASET/train"
# path_labels="/home/fit/Pictures/DATASET/label"
# path_save_img = "img"
# path_save_label="lb"

path_img="/mnt/01D322563C532490/Data/face/211006/dataset/images/train"
path_labels="/mnt/01D322563C532490/Data/face/211006/dataset/labels/train"
path_save_img = "/mnt/01D322563C532490/Data/face/211006/dataset/worst/images/train"
path_save_label="/mnt/01D322563C532490/Data/face/211006/dataset/worst/labels/train"

if not os.path.isdir(path_save_img):
    os.makedirs(path_save_img)

if not os.path.isdir(path_save_label):
    os.makedirs(path_save_label)
file=os.listdir(path_labels)
#list_file = file.readlines()
for txt in file:
    #txt = txt.replace('.jpg\n','.txt')
    #print(txt)
    txt_dir=path_labels+'/'+txt
    img_dir=path_img+'/'+txt.split(".txt")[0]+'.jpg'
    pig = open(txt_dir,'r+')
    coor=pig.readlines()
    for i in coor:
        if i[0] == '2':
            shutil.copy(img_dir, path_save_img)
            shutil.copy(txt_dir, path_save_label)
