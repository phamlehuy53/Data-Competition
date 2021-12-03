import enum
from sys import flags

from torch.utils import data
from yaml import parse
from utils.datasets import *
from utils.general import check_dataset,colorstr
from utils.augmentations import cutout
from argparse import ArgumentParser
from typing import List, Optional, Tuple
import cv2
from datetime import datetime
from copy import deepcopy

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

    @staticmethod
    def read_label(lb_path):
        lb = open(lb_path).read().strip().split('\n')
        lb = [l.split() for l in lb]
        return np.array([ [float(i) for i in l] for l in lb])

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

def extract_one(img: np.ndarray, labels: np.ndarray, class_ids:List[int]=[], ids: List[int]=[], mode:str="blur", padding:int=0):
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
    assert not (class_ids and ids) # extract object by its class id or its id-th in label list
    im_h, im_w = img.shape[:2]
    res_imgs = []
    res_labels = []
    if len( labels ) == 0:
        return [], []
    if mode == "blur":
        pad = 0.01
        padw = int( pad*im_w )
        padh = int( pad*im_h )
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
        for j, ( i, x1, y1, x2, y2 ) in enumerate( labels ):
            x1, y1 = max(x1-padw, 0), max(y1-padh, 0)
            x2, y2 = min(x2+padw, im_w), min(y2+padh, im_h)
            if i in class_ids or j in ids:
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
        # for j, tg_obj in enumerate(labels[labels[:, 0]==i]):

        for j, tg_obj in enumerate(labels):
            if ( tg_obj[0] not in class_ids ) and ( j not in ids ):
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

    # return np.array( res_imgs ), np.array(res_labels)
    return res_imgs , res_labels



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
    dataset = Augmenter(train_path, 1024, 16,
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

    if args.liveshow:
        liveshow(hyp, dataset)
    else:
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

def liveshow(hyp: dict, dataset: Augmenter):
    """Live adjust hyp_params
    Instruction key:
    a = previous image
    d = next image
    q = quit
    e = enter edit mode
        K_MODS: change param correspondingly
            { [ key_num: param_name, step, value ]}
    k = increase current key value by step
    j = decrease current key value by step

    c = directly change current hyp param
        s = change step
        v = change value
        enter in stdin
    Args:
        hyp (dict): [description]
        dataset (Augmenter): [description]
    """
    hyp = deepcopy(hyp)
    # idx = input()
    K_NEXT = 'd'
    K_QUIT = 'q'
    K_PREV = 'a'
    
    K_EDIT = 'e'
    K_DOWN = 'k'
    K_UP   = 'j'
    K_SET  = 'c'
    # name: {key_value, change_value, init_value}
    K_MODS = {
        '1' : [ 'hsv_h', 0.1 , 0.1 ] ,
        '2' : [ 'hsv_s', 0.1 , 0.1 ],
        '3' : [ 'hsv_v', 0.1 , 0.1 ],
        '4' : [ 'degrees'  , 5   , 0 ],
        '5' : [ 'translate', 0.001, 0.005 ],
        '6' : [ 'scale', 0.005, 0.007 ],
        '7' : [ 'shear', 5   , 0 ],
        '8' : [ 'perspective' , 0.0001,  0.0007 ]
        }

    K_SHOW = 'p'
    idx = 0

    # mw, mh = 1024, 640
    cur_key = K_MODS['1']
    # breakpoint()
    while True:
        for v in K_MODS.values():
            hyp[v[0]] = v[2]
        dataset.hyp = hyp
        img, labels, img_file, shapes = dataset[idx]
        h, w = img.shape[:2]
        print(idx, '\t', img_file, '\t', (w, h))
        # img = np.array(img)
        # r = min(mw/w, mh/h)
        img = cv2.resize(img, (w, h))
        xyxy = xywhn2xyxy(labels[:, 1:], *img.shape[:2][::-1])
        # xyxy = xyxy.astype(int)
        ixyxy = np.concatenate( (labels[:, 0, np.newaxis], xyxy), axis=1)
        ixyxy = ixyxy.astype(int)
        draw_img = draw(img, ixyxy)
        cv2.imshow('Image', draw_img)
        k = cv2.waitKey(0)

        if k & 0xFF == ord(K_EDIT):
            print('Enter edit mode')
            k1 = cv2.waitKey(0)
            if k1 & 0xFF in [ord(x) for x in K_MODS.keys()]:
                cur_key = K_MODS[chr( k1 & 0xFF )]
                print("Adjusting ", cur_key[0])
            continue
        else:
            pass

        if k & 0xFF == ord(K_SHOW):
            for v in K_MODS.values():
                print(v, end=' | ')

        if k & 0xFF == ord(K_DOWN):
            cur_key[2] += cur_key[1]
            print(cur_key)

        if k & 0xFF == ord(K_UP):
            cur_key[2] -= cur_key[1]
            print(cur_key)
        
        if k & 0xFF == ord(K_SET):
            k1 = cv2.waitKey(0)
            if k1 & 0xFF == ord('s'):
                cur_key[1] = float(input())
            if k1 & 0xFF == ord('v'):
                cur_key[2] = float(input())

        if k & 0xFF == ord(K_QUIT):
            cv2.destroyAllWindows()
            break
        if k & 0xFF == ord(K_NEXT):
            idx+=1
            idx=min(idx, len(dataset))
        if k & 0xFF == ord(K_PREV):
            idx-=1
            idx=max(0, idx)


        # idx += 1

        # str_in = input()
        # if str_in == 'quit':
        #     break
def draw(img: np.ndarray, labels: np.ndarray):
    """[summary]

    Args:
        img (np.ndarray): [description]
        labels (np.ndarray): [[class_id, xmin, ymin, xmax, ymax]] 
    """
    # TODO: gg
    # color = np.random.randint(0, 255, (3, ))
    img = np.copy(img)
    labels = np.copy(labels)
    for lab in labels:
        x1y1 = tuple(lab[1:3])
        x2y2 = tuple(lab[3:])
        thick = max( img.shape[2] // 100, 2 )
        cv2.rectangle(img, x1y1, x2y2, (0,255,0), thick )
    return img

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('save_dir', type=str, help="Directory to save augmented files, e.g: ~/datacomp/dataset/images/train")
    parser.add_argument('--file_pre', type=str, required=False, help="aug file prefix, if not provided, generate from datetime", default='')
    parser.add_argument('--clean', default=False, type=bool, help="Clean target directory first")
    parser.add_argument('--exclude_list', default=None, type=str, help="Path to file contains toss-out images")
    parser.add_argument('--iter', type=int, default=2, help="No. of samples from one image")

    # TODO: this is apart from above ones
    parser.add_argument('--liveshow', default=False, type=bool)
    args = parser.parse_args()
    _main(args)