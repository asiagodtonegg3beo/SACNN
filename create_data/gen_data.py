
# coding: utf-8

# In[1]:


'''
截取pos，neg,part三种类型图片并resize成12x12大小作为PNet的输入
'''
import os
import cv2
import sys
import numpy as np
import argparse
npr=np.random
from tqdm import  tqdm
from utils import IOU 

parser = argparse.ArgumentParser()
parser.add_argument("-size",help='input the generate face size ex: python gen_data.py -size 16')
args = parser.parse_args()
args.size = int(args.size)
if args.size not in [12, 16, 20, 24, 36, 48, 64]:
   print('Error! Invalid size: {}'.format(args.size))
   sys.exit(1)  
net_size = args.size

#face的id對應label的txt
#anno_file='wider_face_train_filtered.txt'
anno_file='train.txt' # 改
#圖片路徑
#im_dir='../../dataset/WIDER_train/images/'
im_dir = 'D:/SACNN/widerface-test/dataset/WIDER_train/images/'
#pos，part,neg 裁切圖片放到這些位置
pos_save_dir= 'face'+str(net_size)+'/positive/0'
part_save_dir='face'+str(net_size)+'/part/0'
neg_save_dir='face'+str(net_size)+'/negative/0'
#PNet 數據路徑
save_dir='face'+str(net_size)


print(os.getcwd())
# In[2]:


if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(pos_save_dir):
    os.makedirs(pos_save_dir)
if not os.path.exists(part_save_dir):
    os.makedirs(part_save_dir)
if not os.path.exists(neg_save_dir):
    os.makedirs(neg_save_dir)
    
f1=open(os.path.join(save_dir,'pos_'+str(net_size)+'.txt'),'w')
f2=open(os.path.join(save_dir,'neg_'+str(net_size)+'.txt'),'w')
f3=open(os.path.join(save_dir,'part_'+str(net_size)+'.txt'),'w')

with open(anno_file,'r') as f:
    annotations=f.readlines()
num=len(annotations)
print('总共的图片数： %d' % num)
#紀錄pos,neg,part三類的生成數量
p_idx=0
n_idx=0
d_idx=0
#紀錄讀取圖片數量
idx=0
for annotation in tqdm(annotations):
    annotation=annotation.strip().split(' ')
    im_path=annotation[0]
    box=list(map(float,annotation[1:]))
    
    boxes=np.array(box,dtype=np.float32).reshape(-1,4)
    
    img=cv2.imread(os.path.join(im_dir,im_path+'.jpg'))
    idx+=1
    height,width,channel=img.shape
    
    neg_num=0
    #先採樣一定數量的neg圖片
    while neg_num<50:

        #隨機選取擷取圖片大小
        size=npr.randint(16,min(width,height)/2)
        #隨機選取左上座標
        nx=npr.randint(0,width-size)
        ny=npr.randint(0,height-size)
        #截取box
        crop_box=np.array([nx,ny,nx+size,ny+size])
        #計算iou值
        Iou=IOU(crop_box,boxes)
        #擷取圖片並resize成NxN(12x12)大小
        cropped_im=img[ny:ny+size,nx:nx+size,:]
        resized_im=cv2.resize(cropped_im,(net_size,net_size),interpolation=cv2.INTER_LINEAR)

        
        
        #iou值小於0.3判定為neg圖片(0.3=threshold)
        if np.max(Iou)<0.3:
            save_file=os.path.join(neg_save_dir,'%s.jpg'%n_idx)
            f2.write(neg_save_dir+'/%s.jpg'%n_idx+' 0\n')
            cv2.imwrite(save_file,resized_im)
            n_idx+=1
            neg_num+=1
    
    for box in boxes:
        #左上右下座標
        x1,y1,x2,y2=box
        w=x2-x1+1
        h=y2-y1+1
        #捨棄圖片過小和box在圖片外的圖片
        if max(w,h)<20 or x1<0 or y1<0:
            continue
        for i in range(5):
            size=npr.randint(net_size,min(width,height)/2)

                     

            #隨機生成的關於x1,y1的偏移量，并且保證 x1+delta_x > 0 , y1+delta_y > 0
            delta_x=npr.randint(max(-size,-x1),w)
            delta_y=npr.randint(max(-21,-y1),h)
            #擷取後的左上角座標
            nx1=int(max(0,x1+delta_x))
            ny1=int(max(0,y1+delta_y))
            #排除大於圖片尺度的圖片
            if nx1+size>width or ny1+size>height:
                continue
            crop_box=np.array([nx1,ny1,nx1+size,ny1+size])
            Iou=IOU(crop_box,boxes)
            cropped_im=img[ny1:ny1+size,nx1:nx1+size,:]
            resized_im=cv2.resize(cropped_im,(net_size,net_size),interpolation=cv2.INTER_LINEAR)
            
            if np.max(Iou)<0.3:
                save_file=os.path.join(neg_save_dir,'%s.jpg'%n_idx)
                f2.write(neg_save_dir+'/%s.jpg'%n_idx+' 0\n')
                cv2.imwrite(save_file,resized_im)
                n_idx+=1
        for i in range(20):
            # 縮小隨機選取size範圍，更多擷取pos和part圖片
            size=npr.randint(int(min(w,h)*0.8),np.ceil(1.25*max(w,h)))
 
            

            # 除去size小的
            if w<5:
                continue
            # 偏移量，範圍縮小了
            delta_x=npr.randint(-w*0.2,w*0.2)
            delta_y=npr.randint(-h*0.2,h*0.2)
            # 截取圖片左上座標計算，是先計算 x1+w/2 表示的中心座標，再加上 delta_x 偏移量，再-size/2，
            # 變成新的左上座標
            nx1=int(max(x1+w/2+delta_x-size/2,0))
            ny1=int(max(y1+h/2+delta_y-size/2,0))
            nx2=nx1+size
            ny2=ny1+size
            
            #排除超出的圖片
            if nx2>width or ny2>height:
                continue
            crop_box=np.array([nx1,ny1,nx2,ny2])
            #人臉框相對於擷取圖片的偏移量並做歸一化處理
            offset_x1=(x1-nx1)/float(size)
            offset_y1=(y1-ny1)/float(size)
            offset_x2=(x2-nx2)/float(size)
            offset_y2=(y2-ny2)/float(size)
            
            cropped_im=img[ny1:ny2,nx1:nx2,:]
            resized_im=cv2.resize(cropped_im,(net_size,net_size),interpolation=cv2.INTER_LINEAR)
            #box擴充一个維度作為iou輸入
            box_=box.reshape(1,-1)
            iou=IOU(crop_box,box_)
            if iou>=0.65:
                save_file=os.path.join(pos_save_dir,'%s.jpg'%p_idx)
                f1.write(pos_save_dir+'/%s.jpg'%p_idx+' 1 %.2f %.2f %.2f %.2f\n'%(offset_x1,
                        offset_y1,offset_x2,offset_y2))
                cv2.imwrite(save_file,resized_im)
                p_idx+=1
            elif iou>=0.4:
                save_file=os.path.join(part_save_dir,'%s.jpg'%d_idx)
                f3.write(part_save_dir+'/%s.jpg'%d_idx+' -1 %.2f %.2f %.2f %.2f\n'%(offset_x1,
                        offset_y1,offset_x2,offset_y2))
                cv2.imwrite(save_file,resized_im)
                d_idx+=1
   
   
print('%s 個圖片已處理，pos：%s  part: %s neg:%s'%(idx,p_idx,d_idx,n_idx))
f1.close()
f2.close()
f3.close()

