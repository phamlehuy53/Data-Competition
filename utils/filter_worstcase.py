import os
import shutil
path_img="DATAORG/DATASET/train"
path_labels="DATAORG/DATASET/labels"
path_save_img = "DATAORG/worst_case/img"
path_save_label="DATAORG/worst_case/lb"
file=os.listdir('DATAORG/lb_worst')
#list_file = file.readlines()
for txt in file:
    #txt = txt.replace('.jpg\n','.txt')
    #print(txt)
    txt_dir=path_labels+'/'+txt
    img_dir=path_img+'/'+txt.split(".txt")[0]+'.jpg'
    pig = open(txt_dir,'r+')
    coor=pig.readlines()
    k = 0
    for i in coor:
        if i[0] == '2':
            k=k+1
    if(k==0):
        print('doan sang')
        shutil.copy(img_dir, path_save_img)
        shutil.copy(txt_dir, path_save_label)
