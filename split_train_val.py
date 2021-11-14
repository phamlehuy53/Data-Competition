import os
import random
import shutil
from glob import glob
from tqdm import tqdm

src_dir = '/mnt/01D322563C532490/Data/face/face-mask/dataset/data_aug/211114' # {src_dir}/images/*.jpg, {src_dir}/labels/*.txt
tg_dir = '/mnt/01D322563C532490/Data/face/face-mask/dataset/data_aug/211114/submit'  # {tg_dir}/images/train/*.jpg, {tg_dir}/labels/train/*.jpg

os.makedirs(f"{tg_dir}/images/train", exist_ok=True)
os.makedirs(f"{tg_dir}/images/val", exist_ok=True)
os.makedirs(f"{tg_dir}/labels/train", exist_ok=True)
os.makedirs(f"{tg_dir}/labels/val", exist_ok=True)

images_paths = glob(f"{src_dir}/images/*.jpg")
ids = list(range(len(images_paths))) 
random.shuffle(ids)

split_rate = 0.9

# train
for i in tqdm(ids[:int(split_rate*len(ids))]):
	shutil.copy(images_paths[i], f"{tg_dir}/images/train")

	# TODO: rewrite
	bname = os.path.basename(images_paths[i]).split('.')[0] 
	shutil.copy(os.path.join(src_dir, 'labels', f"{bname}.txt"), f"{tg_dir}/labels/train")

for i in tqdm(ids[int(split_rate*len(ids)):]):
	shutil.copy(images_paths[i], f"{tg_dir}/images/val")

	# TODO: rewrite
	bname = os.path.basename(images_paths[i]).split('.')[0] 
	shutil.copy(os.path.join(src_dir, 'labels', f"{bname}.txt"), f"{tg_dir}/labels/val")