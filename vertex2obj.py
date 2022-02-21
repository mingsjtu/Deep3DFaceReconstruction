import os
import numpy as np
import shutil

def read_obj(obj_path):
    v_num = 0
    f_num = 0
    f_obj = open(os.path.join(obj_path), 'r')
    v_list = []
    f_list = []
    for line in f_obj:
        if line.startswith('v '):
            line_list = line.strip('\n').strip('\r').split(' ')
            v_list.append(
                np.array([float(line_list[1]), float(line_list[2]), float(line_list[3])]))
            v_num += 1
        else:
            if line.startswith('f '):
                line_list = line.strip('\n').strip('\r').split(' ')
                f_num += 1
                f_list.append(np.array([int(line_list[1].split(
                    '/')[0]), int(line_list[2].split('/')[0]), int(line_list[3].split('/')[0])]))

    print("v_num  %d" % v_num)
    return np.asarray(v_list), np.asarray(f_list)

def save_obj(v,f,save_path):
    f_obj=open(save_path,"w")
    for i in range(len(v)):
        line="v %f %f %f\n"%(v[i,0],v[i,1],v[i,2])
        f_obj.write(line)
    
    for i in range(len(f)):
        line="f %d %d %d\n"%(f[i,0],f[i,1],f[i,2])
        f_obj.write(line)

    f_obj.close()


v,f=read_obj("/home2/guoming/Dataset/FFHQ/FFHQ-lmk/msra_recon_out/00000._mesh.obj")
v=np.load("meanshape.npy")
v=np.reshape(v,(-1,3))
print(v.shape)
# if np.shape(v)[0]==3:
#     v=np.swapaxes(v,0,1)
print(np.shape(f))
save_obj(v, f, "meanshape.obj")