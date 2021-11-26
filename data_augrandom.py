from sys import flags
from utils.datasets import *
from utils.general import check_dataset,colorstr
from utils.augmentations import cutout
from argparse import ArgumentParser
from typing import List, Optional, Tuple
import cv2
from datetime import datetime

class Augmenter(LoadImagesAndLabels):
    def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False, cache_images=False, single_cls=False, stride=32, pad=0, prefix='', tg_path=None, file_prefix='', clean=False):
        # ./images/train/*.jpg -> tg_path = './images/train
        super().__init__(path, img_size=img_size, batch_size=batch_size, augment=augment, hyp=hyp, rect=rect, image_weights=image_weights, cache_images=cache_images, single_cls=single_cls, stride=stride, pad=pad, prefix=prefix)
        assert tg_path!=None and type(file_prefix)==str
        if file_prefix=='':
            self.pre_file = datetime.now().strftime("%Y%m%d%H%M%s")
        else:
            self.pre_file = file_prefix
        
        self.im_save_dir = tg_path
        self.lab_save_dir = tg_path[::-1].replace('images'[::-1], 'labels'[::-1], 1)[::-1]

        if clean:
            shutil.rmtree(self.im_save_dir, ignore_errors=True)
            shutil.rmtree(self.lab_save_dir, ignore_errors=True)
        os.makedirs(self.im_save_dir, exist_ok=True)
        os.makedirs(self.lab_save_dir, exist_ok=True)

        self.im_name = '{}_{}_{:05d}.jpg' # prefile_srcfile_index
        pass

    def update_pre_file(self):
            self.pre_file = datetime.now().strftime("%Y%m%d%H%M%s")

    @staticmethod
    def write_label(labels: List, label_path: str):
        """Write xywhn to file

        Args:
            labels (List): [description]
            label_path (str): [description]
        """
        i2s = lambda x: str(int(x))
        f2s = lambda x: str(float(x))
        with open(label_path, 'w') as fw:
            content = '\n'.join([ ' '.join(( i2s(lab[0]), *(f2s(i) for i in lab[1:]) )) for lab in labels])
            fw.write(content)
        
    def augment_save(self, index):
        img, labels, img_file, shapes = self.__getitem__(index)
        bname = os.path.basename(img_file).split('.')[0]
        new_im_path = os.path.join(self.im_save_dir, self.im_name.format(bname, self.pre_file,index) )
        # print(new_im_path, img.shape)
        cv2.imwrite(new_im_path, img)
        new_lb_path = img2label_paths([new_im_path])[0]
        self.write_label(labels, new_lb_path)        

    def __getitem__(self, index):
        img, labels, img_file, shapes =  super().__getitem__(index)
        img = img.cpu().detach().numpy().transpose((1,2,0))[..., ::-1]

        # Not include image_weight as inherited class
        labels = labels.cpu().detach().numpy()[:, 1:]
        return img, labels, img_file, shapes

def extract_one(img: np.ndarray, labels: np.ndarray, class_ids: List[int], mode:str="blur", padding:int=0):
    """Extract multi-object image to images contain only one object

    Args:
        img (np.ndarray): 
        labels (np.ndarray): List of labels: [class_id, xc, yc, w, h] in float
        class_id (List[int]): List of class ids to extract
        mode (str): object extracting mode: ["blur", "clear"]
            blur: blur other objects
            clear: extract target object only
        padding (int): Min padding added to object. If not provided, maximum padding not overlap other objects is chosen.
    """
    assert mode in ["blur", "clear"]
    im_h, im_w = img.shape[:2]
    res_imgs = []
    res_labels = []
    if mode == "blur":
        # Case 1: blur other objects
        # TODO: padding for blured object
        labels = np.copy(labels)
        blured_img = np.copy(img)

        class_ids = [float(x) for x in class_ids]

        labels[:, 1:] = xywhn2xyxy(labels[:, 1:], im_w, im_h )
        labels = labels.astype(int)
        # labels[:, 1:] = labels[:, 1]
        objs_extract = [] # id, (x1,y1,x2,y2), img
        # breakpoint()
        for ( i, x1, y1, x2, y2 ) in labels:
            if i in class_ids:
                objs_extract.append(((i,x1,y1,x2,y2), img[y1:y2, x1:x2,...]))
            blured_img[y1:y2, x1:x2] = cv2.GaussianBlur(blured_img[y1:y2, x1:x2], (25,25), 30)

        for ( lb, obj ) in objs_extract:
            i,x1,y1,x2,y2 = lb
            res_img = np.copy(blured_img)
            res_img[y1:y2, x1:x2] = obj
            res_imgs.append(res_img)
            res_labels.append(lb)
        
        if not res_labels:
            return [], []

        res_labels = np.array(res_labels).astype(float)
        res_labels[:, 1:] = xyxy2xywhn(res_labels[:, 1:], im_w, im_h)

        return res_imgs, res_labels
    else:
        labels = np.copy(labels)
        def check_collided(box1, box2):
            # xywhn
            box1 = np.copy(box1)
            box2 = np.copy(box2)
            d = np.abs(box2[:2]-box1[:2])
            if d[0] < (box1[2]+box2[2])/2 and d[1] < (box1[3]+box2[3])/2:
                return True
            return False
        
        def get_max_area(k:int,  strides: List[ int ], anc:Tuple[float, float], bst_are:List, other_objs: List, s = [None]*4,):
            # anc: src_point x0, y0
            # other_objs: [x, y, w, h]
            # stride: [left, top, right, bottom] - included sign
            assert k in [0,1,2,3]
            cnt = 0
            while True:
                # breakpoint()
                cnt += 1
                s[k] += strides[k]

                # collided -> break
                xc, yc = ( s[2]+s[0] )/2+anc[0], (s[1]+s[3])/2+anc[1]
                w, h = s[2]-s[0], s[3]-s[1]
                # collided with image edges 
                if xc-w/2<0 or xc+w/2>1 or yc-h/2<0 or yc+h/2>1:
                    break
                    # collided with other objects
                if np.sum([ check_collided([xc,yc,w,h], b) for b in other_objs ]) > 0:
                    break
                
                # update max area
                if bst_are[2]*bst_are[3]<w*h:
                    bst_are[0] = xc
                    bst_are[1] = yc
                    bst_are[2] = w
                    bst_are[3] = h
                    # show_up(bst_are)
                # Limit k < 4
                if k == 3:
                    continue
                get_max_area(k+1, strides, anc, bst_are, other_objs, s)
                # if not r:
                #     break
            s[k] -= cnt * strides[k]
            if k < 3:
                get_max_area(k+1, strides, anc, bst_are, other_objs, s)
            # return False

        def show_up(bst_are):
            xmin, ymin, xmax, ymax = xywhn2xyxy(np.array([ bst_are ]), im_w, im_h)[0]
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
            res_imgs.append(img[ymin:ymax, xmin:xmax])
            cv2.imshow('test',img[ymin:ymax, xmin:xmax])
            ch = cv2.waitKey(0)
            if ch == 67:
                cv2.destroyAllWindows()
        for i in class_ids:
            # for j, tg_obj in enumerate(labels[labels[:, 0]==i]):
            for j, tg_obj in enumerate(labels):
                if tg_obj[0] != i:
                    continue
                # print(tg_obj, i)
                # print('other ', np.concatenate(( labels[:j], labels[j+1:] )))
                cl_id, xc, yc, w, h = tg_obj
                stride = min(0.05, 1.5*w, 1.5*h)
                strides = [-stride, -stride, stride, stride] # left, top, right, bottom
                # print('Stride 0 ', strides)
                # s = np.array([xc-stride, yc-stride, xc+stride, yc+stride])
                s = np.array(strides)
                # Uncomment this
                # if s[0] < 0 or s[1]<0 or s[2]>1 or s[3]>1:
                #     continue
                bst_are = [0,0,0,0] # xywhn 
                get_max_area(0, strides, (xc, yc), bst_are , np.concatenate(( labels[:j, 1:], labels[j+1:, 1:] )), s)
                # print('best_area ', bst_are)
                if bst_are[2]*bst_are[3] == 0:
                    continue
                xmin, ymin, xmax, ymax = xywhn2xyxy(np.array([ bst_are ]), im_w, im_h)[0]
                nw, nh = w/bst_are[2], h/bst_are[3]
                nxc, nyc = ( xc-(bst_are[0]-bst_are[2]/2) )/bst_are[2], ( yc-(bst_are[1]-bst_are[3]/2) )/bst_are[3]
                res_labels.append([cl_id, nxc, nyc, nw, nh])

                xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
                res_imgs.append(img[ymin:ymax, xmin:xmax])

                # xmin, ymin, xmax, ymax = xywhn2xyxy(np.array([ bst_are ]), im_w, im_h)[0]
                # xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
                # res_imgs.append(img[ymin:ymax, xmin:xmax])
            # print(xmin, ymin, xmax, ymax)
            # print(img[ymin:ymax, xmin:xmax].shape)
                    # break

        # def xywhn2xy4n(x: np.ndarray):
        #     y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        #     y[:, 0] -= y[:, 2]/2
        #     y[:, 1] -= y[:, 3]/2
        #     y[:, 2] += y[:, 0]
        #     y[:, 3] += y[:, 1]
        #     return np.concatenate((y[:, :2], y[:, 0, np.newaxis], y[:, 3, np.newaxis], y[:, 2:], y[:, 2, np.newaxis], y[:, 1, np.newaxis]), axis=1).reshape(-1, 4, 2)

        # label4s = xywhn2xy4n(labels[:, 1:])
        # points = label4s.reshape(-1, 2)
        # if len(labels) < 2:
        #     #TODO: break here
        #     pass
        # for i in class_ids:
        #     for j in range(len(labels)):
        #         if labels[j, 0] == i:
        #             continue
        #         tg_obj = labels[j]
        #         tg_id, xc, yc, w, h = tg_obj
        #         # TODO: overlap-tg_obj cases not handled
        #         src_point = points[np.argsort(np.sum( np.abs( points-tg_obj[1:3] ), axis=1))[4]]
        #         dxy = np.abs(src_point-tg_obj[1:3])
        #         dmin = max(dxy)
        #         # crop = np.concatenate((tg_obj[1:3], ))
        #         # Ignore cropped areas smaller than objs 
        #         if 4*dxy[0]*dxy[1] < w*h:
        #             continue
        #         # new_label = np.array([ tg_id, dxy[0], dxy[1], w/(2*dxy[0]), h/(2*dxy[1]) ])
        #         new_label = np.array([ tg_id, dmin, dmin, w/(2*dmin), h/(2*dmin) ])
        #         # xmin, ymin, xmax, ymax = [int(x) for x in  xywhn2xyxy(np.array( [ [xc, yc, *(dxy*2)] ] ), im_w, im_h)[0]]
        #         xmin, ymin, xmax, ymax = [int(x) for x in  xywhn2xyxy(np.array( [ [xc, yc, dmin*2, dmin*2] ] ), im_w, im_h)[0]]
        #         res_imgs.append(img[ymin: ymax, xmin: xmax])
        #         res_labels.append(new_label)

    return np.array( res_imgs ), np.array(res_labels)

def read_label(lb_path: str):
    data = open(lb_path, 'r').read().strip().split('\n')
    return [ [float(x) for x in l.split() ] for l in data]



def _main(args):
    assert args.iter > 0
    data_dict = check_dataset('config/data_cfg.yaml')
    #print(data_dict)
    train_path, val_path = data_dict['train'], data_dict['val']
    # gen dataset
    with open('config/hyps/hyp_finetune.yaml') as f:
        hyp: dict = yaml.safe_load(f)  # load hyper parameter dict
    
    exclude_list = None
    if args.exclude_list:
        try:
            exclude_list = open(args.exclude_list).read().strip().split('\n')
            for f in exclude_list:
	           # TODO: fixed 'train' 
                f = os.path.join(train_path, 'images/train', f)
                if os.path.isfile(f):   
                    shutil.move(f, f+'.bak')
                    # exclude_list.remove(f)
        except Exception as e:
            print("Load exclude-list file failed!", e)
            exclude_list = None
    dataset = Augmenter(train_path, 640, 16,
                                      augment=True,  # augment images
                                      hyp=hyp,  # augmentation hyperparameters
                                      rect=False,  # rectangular training
                                      cache_images=True,
                                      single_cls=False,
                                      stride=32,
                                      pad=0.0,
                                      image_weights=False,
                                      prefix=colorstr('Train: '),
                                      tg_path=args.save_dir,
                                      file_prefix=args.file_pre,
                                      clean=args.clean)

    for j in tqdm( range(args.iter) ):
	   #TODO: tidy code 
        dataset.update_pre_file()
        dataset.pre_file = f"{j:02d}_{dataset.pre_file}"
        for i in tqdm( range(len(dataset)), leave=False):
            dataset.augment_save(i)
    
    if exclude_list:
        for f in exclude_list:
            f = os.path.join(train_path, 'images/train', f)
            try:
                shutil.move(f+'.bak', f)
            except: 
                pass
    pass

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('save_dir', type=str, help="Directory to save augmented files, e.g: ~/datacomp/dataset/images/train")
    parser.add_argument('--file_pre', type=str, required=False, help="aug file prefix, if not provided, generate from datetime", default='')
    parser.add_argument('--clean', default=False, type=bool, help="Clean target directory first")
    parser.add_argument('--exclude_list', default=None, type=str, help="Path to file contains toss-out images")
    parser.add_argument('--iter', type=int, default=2, help="No. of samples from one image")
    args = parser.parse_args()
    _main(args)