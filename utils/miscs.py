from typing import List
import numpy as np
import cv2
from utils import general
import os

def quick_show(im):
    cv2.imshow('image', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def read_label_file(lb_path):
    lb = open(lb_path).read().strip().split('\n')
    lb = [l.split() for l in lb]
    return np.array([ [float(i) for i in l] for l in lb])

def load_batch(im_paths: List[str], lab_paths: List[str], lab_type:str='xyxy'):
    assert len(im_paths) == len(lab_paths)

    im_batch = []
    lab_batch = []
    for im_path, lab_path in zip(im_paths, lab_paths):
        im = cv2.imread(im_path)
        h, w = im.shape[:2]
        labs = read_label_file(lab_path)

        if lab_type == 'xyxy':
            # labs = np.array([ labs[0], *general.xywhn2xyxy(labs[1:], w,h)])
            labs[:, 1: ] = general.xywhn2xyxy(labs[:, 1:], w, h)
            labs = labs.astype(np.int32)
            # labs[:, 1:] = general.xywhn2xyxy(labs[:, 1:])
        
        im_batch.append(im)
        lab_batch.append(labs)
    
    return im_batch, lab_batch

