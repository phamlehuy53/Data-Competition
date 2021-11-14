import os
import cv2
import random
# path_img="/media/sang/UBUNTU/Data-Competition-mainv1/submit/images/images"
# path_labels="/media/sang/UBUNTU/Data-Competition-mainv1/submit/labels/labels"
path_img="../DATAORG/DATASET/train"
path_labels="../DATAORG/DATASET/labels"

os.makedirs("../DATASUBMIT_11_11/data_aug011_11/data_aug_difficult/imgworst_3/images/", exist_ok=True)
os.makedirs("../DATASUBMIT_11_11/data_aug011_11/data_aug_difficult/imgworst_3/labels/", exist_ok=True)
file=os.listdir(path_labels)
#list_file = file.readlines()
index=1
for txt in file:
    txt_dir=path_labels+'/'+txt
    #print(txt_dir)
    img_dir=path_img+'/'+txt.split(".txt")[0]+'.jpg'
    pig = open(txt_dir,'r+')
    coor=pig.readlines()
    img = cv2.imread(img_dir)
    im_height,im_width = img.shape[0], img.shape[1]
    i = random.choice(coor)
    index=index+1
    if i[0] == '0':
        Cord=i.split("\n")[0].split(" ")
        #print(Cord)
        x_min=float(Cord[1])-float(Cord[3])/2
        y_min=float(Cord[2])-float(Cord[4])/2
        x_max=float(Cord[1])+float(Cord[3])/2
        y_max=float(Cord[2])+float(Cord[4])/2

        x_min = x_min * im_width
        x_max= x_max * im_width
        y_min=  y_min * im_height
        y_max =  y_max * im_height
        x_min=int(x_min)
        x_max=int(x_max)
        y_min= int(y_min)
        y_max=int(y_max)
        # Lấy tọa độ sau
        x_min_after=int(x_min)
        x_max_after=int(x_max)
        y_min_after= int(y_min)
        y_max_after=int(y_max)
        X_pad=120
        Y_pad=80
        x_min = x_min-X_pad
        y_min = y_min-Y_pad
        y_max = y_max+Y_pad
        x_max = x_max+X_pad

        #chuẩn hóa tọa độ
        if x_min < 0:
            x_min = 0
        if x_max > im_width - 1:
            x_max = im_width - 1
        if y_min < 0:
            y_min = 0
        if y_max > im_height - 1:
            y_max = im_height - 1
        # Lấy tọa độ sau
        coor_after = (X_pad,Y_pad,x_max-x_min-X_pad,y_max-y_min-Y_pad)
        crop_img = img[y_min:y_max, x_min:x_max]
        im_height_after,im_width_after = crop_img.shape[0], crop_img.shape[1]
        coor_text_XC = (coor_after[2]+coor_after[0])/(2*im_width_after)
        coor_text_YC =(coor_after[3]+coor_after[1])/(2*im_height_after)
        coor_text_WC =(coor_after[2]-coor_after[0])/im_width_after
        coor_text_HC =(coor_after[3]-coor_after[1])/im_height_after
        resave = "0"+" "+str(coor_text_XC)+" "+str(coor_text_YC)+" "+str(coor_text_WC)+" "+str(coor_text_HC)+"\n"
        cv2.imwrite("../DATASUBMIT_11_11/data_aug011_11/data_aug_difficult/imgworst_3/images/00000000003"+str(index)+".jpg",crop_img)
        txt_file=open("../DATASUBMIT_11_11/data_aug011_11/data_aug_difficult/imgworst_3/labels/00000000003"+str(index)+".txt",'w')
        txt_file.write(resave)
#000000000004
##### 30-70
##### 50-90
##### 80-120
##### 100-120
##### 100-70
##### 120-50
##### 300-150
##### 150-300
##### 500-500
##### 600-500


#####ANH HUY
##### 100-120
##### 80-60
##### 120-60
##### 50-100
