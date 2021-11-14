# %%
import os
os.chdir('../')
import numpy as np
import importlib
from utils import miscs, augmentations, datasets, general

# %% Reload 
augmentations = importlib.reload(augmentations)
miscs = importlib.reload(miscs)
# %%
load_obj = datasets.LoadImagesAndLabels('../dataset/images/public_test')
idx = np.random.randint(0, len(load_obj), (10, ))
ims_paths = np.array(load_obj.img_files)[idx]
labs_paths = np.array(load_obj.label_files)[idx]

# %%
ims, labels = miscs.load_batch(ims_paths, labs_paths)

# %%
res = augmentations.intense(ims, labels)


# %%
import cv2
tg_dir = '/tmp/results'
if not os.path.isdir(tg_dir):
	os.makedirs(tg_dir)


for i, (im, lab) in enumerate(zip(*res)):
	cv2.imwrite(os.path.join(tg_dir, f"{i:03d}.jpg"), im)
	_lab = lab.copy().astype(float)
	_lab[:, 1:] = general.xyxy2xywhn(_lab[:, 1:], im.shape[1], im.shape[0])
	_lab[:, 0] = _lab[:, 0].astype(int)
	# print(_lab)
	with open(os.path.join(tg_dir, f"{i:03d}.txt"), 'w') as fw:
		fw.write('\n'.join( [  str(int(l[0]))+' '+' '.join(str(word)for word in l[1:])  for l in _lab ] ))