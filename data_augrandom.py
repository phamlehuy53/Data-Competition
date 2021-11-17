from sys import flags
from utils.datasets import *
from utils.general import check_dataset,colorstr
from utils.augmentations import cutout
from argparse import ArgumentParser
from typing import List
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
        labels = labels.cpu().detach().numpy()[:, 1:]
        return img, labels, img_file, shapes

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