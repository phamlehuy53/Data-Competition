from sys import flags
from utils.datasets import *
from utils.general import check_dataset,colorstr
from utils.augmentations import cutout

class GenImagesAndLabels(Dataset):  # for training/testing
    def __init__(self, path, img_size=640, batch_size=1, augment=True, hyp=None, rect=False, image_weights=False,
                 cache_images=False, single_cls=False, stride=32, pad=0.0, prefix=''):
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.path = path
        self.albumentations = Albumentations() if augment else None

        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                    # f = list(p.rglob('**/*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p, 'r') as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                        # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise Exception(f'{prefix}{p} does not exist')
            self.img_files = sorted([x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS])
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in img_formats])  # pathlib
            assert self.img_files, f'{prefix}No images found'
        except Exception as e:
            raise Exception(f'{prefix}Error loading config from {path}: {e}\nSee {HELP_URL}')

        # Check cache
        self.label_files = img2label_paths(self.img_files)  # labels
        cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix('.cache')
        try:
            cache, exists = np.load(cache_path, allow_pickle=True).item(), True  # load dict
            assert cache['version'] == 0.4 and cache['hash'] == get_hash(self.label_files + self.img_files)
        except:
            cache, exists = self.cache_labels(cache_path, prefix), False  # cache

        # Display cache
        nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupted, total
        if exists:
            d = f"Scanning '{cache_path}' images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupted"
            tqdm(None, desc=prefix + d, total=n, initial=n)  # display cache results
            if cache['msgs']:
                logging.info('\n'.join(cache['msgs']))  # display warnings
        assert nf > 0 or not augment, f'{prefix}No labels in {cache_path}. Can not train without labels. See {HELP_URL}'

        # Read cache
        [cache.pop(k) for k in ('hash', 'version', 'msgs')]  # remove items
        labels, shapes, self.segments = zip(*cache.values())
        self.labels = list(labels)
        #print(self.labels[0])
        self.shapes = np.array(shapes, dtype=np.float64)
        self.img_files = list(cache.keys())  # update
        self.label_files = img2label_paths(cache.keys())  # update
        if single_cls:
            for x in self.labels:
                x[:, 0] = 0

        n = len(shapes)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = range(n)
        # Rectangular Training
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int) * stride

        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        self.imgs, self.img_npy = [None] * n, [None] * n
        if cache_images:
            if cache_images == 'disk':
                self.im_cache_dir = Path(Path(self.img_files[0]).parent.as_posix() + '_npy')
                self.img_npy = [self.im_cache_dir / Path(f).with_suffix('.npy').name for f in self.img_files]
                self.im_cache_dir.mkdir(parents=True, exist_ok=True)
            gb = 0  # Gigabytes of cached images
            self.img_hw0, self.img_hw = [None] * n, [None] * n
            results = ThreadPool(NUM_THREADS).imap(lambda x: load_image(*x), zip(repeat(self), range(n)))
            pbar = tqdm(enumerate(results), total=n)
            for i, x in pbar:
                if cache_images == 'disk':
                    if not self.img_npy[i].exists():
                        np.save(self.img_npy[i].as_posix(), x[0])
                    gb += self.img_npy[i].stat().st_size
                else:
                    self.imgs[i], self.img_hw0[i], self.img_hw[i] = x  # im, hw_orig, hw_resized = load_image(self, i)
                    gb += self.imgs[i].nbytes
                pbar.desc = f'{prefix}Caching images ({gb / 1E9:.1f}GB {cache_images})'
            pbar.close()

    def cache_labels(self, path=Path('./labels.cache'), prefix=''):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels..."
        with Pool(NUM_THREADS) as pool:
            pbar = tqdm(pool.imap_unordered(verify_image_label, zip(self.img_files, self.label_files, repeat(prefix))),
                        desc=desc, total=len(self.img_files))
            for im_file, l, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x[im_file] = [l, shape, segments]
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc}{nf} found, {nm} missing, {ne} empty, {nc} corrupted"

        pbar.close()
        if msgs:
            logging.info('\n'.join(msgs))
        if nf == 0:
            logging.info(f'{prefix}WARNING: No labels found in {path}. See {HELP_URL}')
        x['hash'] = get_hash(self.label_files + self.img_files)
        x['results'] = nf, nm, ne, nc, len(self.img_files)
        x['msgs'] = msgs  # warnings
        x['version'] = 0.4  # cache version
        try:
            np.save(path, x)  # save cache for next time
            path.with_suffix('.cache.npy').rename(path)  # remove .npy suffix
            logging.info(f'{prefix}New cache created: {path}')
        except Exception as e:
            logging.info(f'{prefix}WARNING: Cache directory {path.parent} is not writeable: {e}')  # path not writeable
        return x

    # def __len__(self):
    #     return len(self.img_files)

    # def __iter__(self):
    #     self.count = -1
    #     print('ran dataset iter')
    #     #self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
    #     return self

    def gen(self):
        print("len file: ",len(self.img_files))
        for index in range(len(self.img_files)):

            index = self.indices[index]  # linear, shuffled, or image_weights

            hyp = self.hyp
            mosaic = self.mosaic and random.random() < hyp['mosaic']
            #print('mosaic:  ',mosaic)
            if mosaic:
                # Load mosaic
                img, labels = load_mosaic(self, index)
                shapes = None

                # MixUp augmentation
                if random.random() < hyp['mixup']:
                    img, labels = mixup(img, labels, *load_mosaic(self, random.randint(0, self.n - 1)))

            else:
                # Load image
                img, (h0, w0), (h, w) = load_image(self, index)
                
                # Letterbox
                shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
                img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
                shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling
                labels = self.labels[index].copy()
                if labels.size:  # normalized xywh to pixel xyxy format
                    labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])
                    #print("labels: ",labels)
                if self.augment:
                    #print("hyp:  ",hyp['degrees'])
                    img, labels = random_perspective(img, labels,
                                                    degrees=hyp['degrees'],
                                                    translate=hyp['translate'],
                                                    scale=hyp['scale'],
                                                    shear=hyp['shear'],
                                                    perspective=hyp['perspective'])
                    # if len(labels)>0:
                    
                    #     print(labels)
                    #     cv2.rectangle(img,(int(labels[0][1]),int(labels[0][2])),(int(labels[0][3]),int(labels[0][4])), (255,0,0), 2)
                    #     #cv2.rectangle(img,(int(labels[0][1]),int(labels[0][2])),(int(labels[0][3]),int(labels[0][4])))
                    #     cv2.imshow("img",img)
                    #     cv2.waitKey(0)


            nl = len(labels)  # number of labels
            if nl:
                labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1E-3)

            # TODO: Check

            if self.augment:
                # Albumentations
                img, labels = self.albumentations(img, labels)
                
                nl = len(labels)  # update after albumentations

                # HSV color-space
                augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

                # Flip up-down
                # if random.random() < hyp['flipud']:
                #     img = np.flipud(img)
                #     if nl:
                #         labels[:, 2] = 1 - labels[:, 2]
                        #print(labels)
                    
                    

                #Flip left-right
                if random.random() < hyp['fliplr']:
                    img = np.fliplr(img)
                    if nl:
                        labels[:, 1] = 1 - labels[:, 1]
                    
                #Cutouts
                #labels = cutout(img, labels, p=0.2)
            #print("len ::::",len(labels))
            #print("shape:::",labels.shape)
            #print(index)
            os.makedirs("../dataset/data_aug/211114/images/", exist_ok=True)
            os.makedirs("../dataset/data_aug/211114/labels/", exist_ok=True)
            cv2.imwrite("../dataset/data_aug/211114/images/111100000004"+str(index)+".jpg",img)
            txt_file=open("../dataset/data_aug/211114/labels/111100000004"+str(index)+".txt",'w')
            for item in labels:
                item=item.tolist()
                item[0]=int(item[0])
                string =""
                for vvv in item:
                    string =string +str(vvv)+" "
                string_list=list(string)
                string_list.pop()
                string ="".join(string_list)
                string =string+'\n'
                #print(string)
                txt_file.write(string)


data_dict = check_dataset('config/data_cfg.yaml')
#print(data_dict)
train_path, val_path = data_dict['train'], data_dict['val']
# gen dataset
with open('config/hyps/hyp_finetune.yaml') as f:
    hyp: dict = yaml.safe_load(f)  # load hyper parameter dict
#print(hyp)
dataset = GenImagesAndLabels(train_path, 640, 16,
                                      augment=True,  # augment images
                                      hyp=hyp,  # augmentation hyperparameters
                                      rect=True,  # rectangular training
                                      cache_images=True,
                                      single_cls=False,
                                      stride=32,
                                      pad=0.0,
                                      image_weights=False,
                                      prefix=colorstr('Train: '))
dataset.gen()