import glob
import cv2
import numpy as np
import tqdm

# import shutil

# for f in glob.glob('valid/*/rgb*'):
#     n = f.split('/')[-2]
#     shutil.move(f,f.replace(n,'images').replace('rgb', n))


# for f in glob.glob('valid/*/labels*'):
#     dst = f.split('/')
#     dst[-1] = dst[-2] + '.png'
#     dst[-2] = 'labels'
#     shutil.move(f,'/'.join(dst))


colors = []
for f in tqdm.tqdm(glob.glob('valid/labels/*')):
    img = cv2.imread(f)
    u = np.unique(img.reshape(-1,3),axis=0)
    colors.extend(u)
print(np.unique(colors, axis=0))