import cv2
import os
import numpy as np

srcdir="/home2/guoming/Dataset/FFHQ/FFHQ-lmk/image/image"
lmkdir="/home2/guoming/Dataset/FFHQ/FFHQ-lmk/dlib_lm5p"
srclist=os.listdir(srcdir)

for f in srclist[:5]:
    if not (f.endswith("jpg") or f.endswith("png")):
        continue
    img=cv2.imread(os.path.join(srcdir,f))
    lmk=np.loadtxt(os.path.join(lmkdir,f[:-3]+"txt"))
    
    for i in range(len(lmk)):
        img=cv2.circle(img,lmk[i].astype("int"),1,(0,0,255),-1)
        img=cv2.putText(img,str(i),lmk[i].astype("int")+[2,2], cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0), 1)
    cv2.imwrite(f,img)