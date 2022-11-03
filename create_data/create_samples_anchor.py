from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time #計算運行時間用
start = time.time() #計算運行時間用

import os, sys, cv2
import numpy as np
import numpy.random as npr
from enum import Enum
from mytools import IoU, view_bar_gen, read_image, view_bar_gen_add_anchor_face
import argparse
import random

 
class DatasetType(Enum):
    WIDER = 0
    CELEBA = 1
    NONE = -1

class ClassFlag(Enum):
    POSITIVE = 1
    NEGATIVE = 0
    PART = -1

def get_bbox_info(box):
   x1, y1, x2, y2 = box
   cw = x2-x1+1
   ch = y2-y1+1
   cx = np.floor(cw/2) + x1
   cy = np.floor(ch/2) + y1
   return cx, cy, cw, ch

def get_bbox_anchor(img_shape, bbox_in, landmark_in,aug,scale):
  if aug == 1:
      x1, y1, x2, y2 = bbox_in
      w = x2-x1+1
      h = y2-y1+1
      #aug_size = npr.randint(int(min(w, h) * 0.8*scale), np.ceil(scale*1.25 * max(w, h)))
      aug_size = int(scale*(npr.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))))
      delta_x = npr.randint(-w * 0.2, w * 0.2)
      delta_y = npr.randint(-h * 0.2, h * 0.2)
      #x1_new = int(max(x1 + w / 2 + delta_x - aug_size / 2, 0))
      #y1_new = int(max(y1 + h / 2 + delta_y - aug_size / 2, 0))
      x1_new = int(x1 + w / 2 + delta_x - aug_size / 2)
      y1_new = int(y1 + h / 2 + delta_y - aug_size / 2)
      x2_new = x1_new + aug_size
      y2_new = y1_new + aug_size
    
      osx1 = (x1-x1_new) / float(aug_size)
      osy1 = (y1-y1_new) / float(aug_size)
      osx2 = (x2-x2_new) / float(aug_size)
      osy2 = (y2-y2_new) / float(aug_size)
    
      bbox_out = [x1_new, y1_new, x2_new, y2_new]
      bbox_out = [int(round(bbox)) for bbox in bbox_out]
      bbox_offset = [osx1, osy1, osx2, osy2]
      
      cw_new = x2_new-x1_new+1
      ch_new = y2_new-y1_new+1
      
      if len(landmark_in) == 10:
        m1, m2, m3, m4, m5, m6, m7, m8, m9, m10 = landmark_in
        odd_m = (np.array([m1, m3, m5, m7, m9])-x1_new) / float(cw_new)
        m1, m3, m5, m7, m9 = odd_m.tolist()
        even_m = (np.array([m2, m4, m6, m8, m10])-y1_new) / float(ch_new)
        m2, m4, m6, m8, m10 = even_m.tolist()
        landmark_offset = [m1, m2, m3, m4, m5, m6, m7, m8, m9, m10]
      else:
        landmark_offset = []
      return bbox_out, bbox_offset, landmark_offset
  elif aug == 0:
      x1, y1, x2, y2 = bbox_in     
      w = x2-x1+1
      h = y2-y1+1
      #aug_size = npr.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))
      aug_size =  np.ceil((1 * max(w, h))*scale)
      #delta_x = npr.randint(-w * 0.2, w * 0.2)
      #delta_y = npr.randint(-h * 0.2, h * 0.2)
     
      #x1_new = int(max(x1 + w / 2 - aug_size / 2, 0)) #這可能會導致黑邊問題_詳情見2020/02/26 meeintg內容
      #y1_new = int(max(y1 + h / 2 - aug_size / 2, 0))
      
      x1_new = int(x1 + w / 2 - aug_size / 2)
      y1_new = int(y1 + h / 2 - aug_size / 2)
      
      x2_new = x1_new +aug_size
      y2_new = y1_new +aug_size
    
      osx1 = (x1-x1_new) / float(aug_size)
      osy1 = (y1-y1_new) / float(aug_size)
      osx2 = (x2-x2_new) / float(aug_size)
      osy2 = (y2-y2_new) / float(aug_size)
    
      bbox_out = [x1_new, y1_new, x2_new, y2_new]
      bbox_out = [int(round(bbox)) for bbox in bbox_out]
      bbox_offset = [osx1, osy1, osx2, osy2]
      
      cw_new = x2_new-x1_new+1
      ch_new = y2_new-y1_new+1
      
      if len(landmark_in) == 10:
        m1, m2, m3, m4, m5, m6, m7, m8, m9, m10 = landmark_in
        odd_m = (np.array([m1, m3, m5, m7, m9])-x1_new) / float(cw_new)
        m1, m3, m5, m7, m9 = odd_m.tolist()
        even_m = (np.array([m2, m4, m6, m8, m10])-y1_new) / float(ch_new)
        m2, m4, m6, m8, m10 = even_m.tolist()
        landmark_offset = [m1, m2, m3, m4, m5, m6, m7, m8, m9, m10]
      else:
        landmark_offset = []
      return bbox_out, bbox_offset, landmark_offset
      

def get_save_path(part_idx, part_save_dir):
  folder_index = int(np.ceil(part_idx/1000.0))
  save_folder = "{}{:05d}/".format(part_save_dir, folder_index)
  if not os.path.exists(save_folder):
    os.mkdir(save_folder)
  save_path = "{}{}.png".format(save_folder, part_idx)
  return save_path

def get_avg_width(bboxes):
  width_list = []
  for bbox in bboxes:
    x1, y1, x2, y2 = bbox
    width_list.append(x2-x1)
  return int(round(np.sum(width_list) / len(bboxes)))

class DirCreator():
  def __init__(self, net_size):
    self.save_dir = "face{}/".format(net_size)
    self.neg_save_dir = os.path.join(self.save_dir, "negative/")
    self.pos_save_dir = os.path.join(self.save_dir, "positive/")
    self.part_save_dir = os.path.join(self.save_dir, "part/")
    self.anchor_face_save_dir = os.path.join(self.save_dir, "positive/")
    if not os.path.exists(self.save_dir):
        os.mkdir(self.save_dir)
#    if not os.path.exists(self.pos_save_dir):
#        os.mkdir(self.pos_save_dir)
#    if not os.path.exists(self.part_save_dir):
#        os.mkdir(self.part_save_dir)
    if not os.path.exists(self.neg_save_dir):
        os.mkdir(self.neg_save_dir)
    if not os.path.exists(self.anchor_face_save_dir):
        os.mkdir(self.anchor_face_save_dir)
    self.fneg = open(os.path.join(self.save_dir, 'neg_face{}.txt'.format(net_size)), 'w')
#    self.fpos = open(os.path.join(self.save_dir, 'pos_face{}.txt'.format(net_size)), 'w')
#    self.fpart = open(os.path.join(self.save_dir, 'part_face{}.txt'.format(net_size)), 'w')
    self.f_anchor_face = open(os.path.join(self.save_dir, 'positive{}.txt'.format(net_size)), 'w')
  def __enter__(self):
    return self
  def __exit__(self, exc_type, exc_value, traceback):
#    self.fpos.close()
#    self.fpart.close()
    self.fneg.close()

def get_dataset_lines(dataset_type):
  wider_file = 'wider_face_train_filtered.txt'
  celeba_file = 'list_bbox_celeba_clean.txt'
  lines =[]
  if dataset_type == DatasetType.WIDER:
    with open(wider_file) as fp:
      wider_lines = [[DatasetType.WIDER, line.strip()] for line in fp]
    lines.extend(wider_lines)
  elif dataset_type == DatasetType.CELEBA:
    with open(celeba_file) as fp:
      celeba_lines = [[DatasetType.CELEBA, line.strip()] for line in fp]
    lines.extend(celeba_lines)
  else:
     print("Error! Unknown dataset.")
  return lines

if __name__ == '__main__':
    
  aug = 1 # 控制是否要 data augmentation，0 代表 False
  anchor_mode = 0 # 0為一個anchor、1為 4個 anchors
  scale_small = 0.79 # 每個 anchor 的縮小比例
  
  parser = argparse.ArgumentParser()
  parser.add_argument("-size")
  args = parser.parse_args()
  args.size = int(args.size)
  if args.size not in [12, 16, 20 ,36, 48]:
     print('Error! Invalid size: {}'.format(args.size))
     sys.exit(1)
  net_size = args.size
 
  MODEL_W, MODEL_H = net_size, net_size
  wider_dir = "../../dataset/WIDER_train/images/"
  celeba_dir = "../../dataset/CelebA/img_celeba/"
  pos_idx = part_idx = neg_idx = anchor_face_idx = 0
  with DirCreator(net_size) as dp:
    #for dataset_type in [DatasetType.WIDER, DatasetType.CELEBA]:
    #for dataset_type in [DatasetType.CELEBA]:
    for dataset_type in [DatasetType.WIDER]:
      print('\n'+dataset_type.name)
      lines = get_dataset_lines(dataset_type)      
      for j,line in enumerate(lines):
        #if j % 200 != 0: continue        
        #if j >= 30: break    
        dataset_type, line = line[0], line[1]
        words = line.strip().split(' ')
        img_name = words[0]
        if dataset_type == DatasetType.WIDER:
          img_dir = wider_dir
          bboxes =list(map(float, words[1:]))
          bboxes = np.array(bboxes, dtype=np.float32).reshape(-1, 4)
          landmarks = []
          NUM_POS, NUM_NEG = 18, 150 #超參數 暴力測試出來的 # pos : part : neg = 1:1:3
          #NUM_POS, NUM_NEG = 1, 9
          img_path = img_dir + img_name + '.jpg'
        if dataset_type == DatasetType.CELEBA:
          img_dir = celeba_dir
          bboxes = list(map(float, words[1:5]))
          bboxes = np.array(bboxes, dtype=np.float32).reshape(1, 4)
          landmarks = list(map(float, words[5:]))
          NUM_POS, NUM_NEG = 15, 12
          img_path = img_dir + img_name
        img = read_image(img_path)

        avg_width = get_avg_width(bboxes)
        for k in range(NUM_NEG):
          img_height, img_width = img.shape[0], img.shape[1]
          min_shape = min(img_height, img_width)
          rand_upper = avg_width*2.0
          rand_lower = avg_width*0.7
          if rand_lower >= rand_upper:
             continue
          size_back = int(npr.randint(rand_lower, rand_upper))
          if size_back >= min_shape:
            size_back = avg_width
            
          ##偵錯用##  
          intinfo = np.iinfo(int)
          maxnum1 = img.shape[1] - size_back
          maxnum2 = img.shape[0] - size_back
            
          try :  
            nx1 = npr.randint(0, maxnum1) #將"img.shape[1] - size_back"改為maxnum1
          except ValueError:
            nx1 = npr.randint(0, 1)
            
          try :  
            ny1 = npr.randint(0, maxnum2) #將"img.shape[0] - size_back"改為maxnum2
          except ValueError:
            ny1 = npr.randint(0, 1)            
            
          ##偵錯用##
          
          nx2, ny2 = nx1+size_back, ny1+size_back
          bbox_iou = np.max(IoU([nx1, ny1, nx2, ny2], bboxes))
          if bbox_iou < 0.3:
              neg_idx += 1
              img_crop = img[ny1:ny2, nx1:nx2, :]
              img_resize = cv2.resize(img_crop, (MODEL_H, MODEL_W), interpolation=cv2.INTER_LINEAR)
              save_path = get_save_path(neg_idx, dp.neg_save_dir)              
              dp.fneg.write("{} {}\n".format(save_path, 0))
              cv2.imwrite(save_path, img_resize)
              
        for bbox in bboxes:
          for k in range(NUM_POS):
            img_flag = True 
            string_all = ""
            anchor_flag = False
            
            x1, y1, x2, y2 = bbox            
            if min(x2-x1, y2-y1) < 10 or x1 < 0 or y1 < 0:
              continue
            if  x1 < 0 or y1 < 0:
              continue
          
            if anchor_mode == True:                            
                anchor_count_j = random.randint(-2,0) #產生四種type的 anchor #改anchor
                i_num = 3
            elif anchor_mode == False:
                anchor_count_j = 0
                i_num = 1
            
            for anchor_count_i in range(i_num):
                bbox_new, bbox_offset, landmark_offset = get_bbox_anchor(img.shape, bbox, landmarks,aug,pow(scale_small,anchor_count_i+anchor_count_j))                        
                nx1,ny1,nx2,ny2 = bbox_new                
                if nx2 > img_width or ny2 > img_height or nx1 < 0 or ny1 < 0:
                    break
                osx1,osy1,osx2,osy2 = bbox_offset
                bbox_iou = np.max(IoU([nx1, ny1, nx2, ny2], bboxes))
                
                if img_flag == True :                                            
                    img_crop = img[ny1:ny2, nx1:nx2, :]
                    img_flag = False
                if bbox_iou >= 0.65:
                    save_flag = 1
                    anchor_flag = True
                    string_temp = " {} {:.4f} {:.4f} {:.4f} {:.4f}".format(save_flag , osx1, osy1, osx2, osy2)
                    string_all = string_all +string_temp
                elif bbox_iou >= 0.4:
                    save_flag = -1
                    anchor_flag = True
                    string_temp = " {} {:.4f} {:.4f} {:.4f} {:.4f}".format(save_flag , osx1, osy1, osx2, osy2)
                    string_all = string_all +string_temp
                ''' 
                elif:    
                    save_flag = 0
                    string_temp = " {} {:.2f} {:.2f} {:.2f} {:.2f}".format(save_flag , 0, 0, 0, 0)
                    string_all = string_all +string_temp                
                '''
            if nx2 > img_width or ny2 > img_height or nx1 < 0 or ny1 < 0:
                    #string_temp = " {:.3f} {:.4f} {:.4f} {:.4f} {:.4f}".format(0 , 0, 0, 0, 0)
                    #string_all = string_all +string_temp
                    #continue                
                    break
            
            save_idx = False
                
            if anchor_flag == True:## 判斷是否四個框都不為 0
                anchor_face_idx += 1                
                save_idx = anchor_face_idx
                save_dir = dp.anchor_face_save_dir
                save_fp = dp.f_anchor_face
            
            if save_idx:                
            #if (save_flag or anchor_mode) == True:                
              #if dataset_type == DatasetType.WIDER:
              save_path = get_save_path(save_idx, save_dir)                                  
              string_title ="{} ".format(save_path)
              string_save = string_title +string_all                
              save_fp.write(string_save+"\n")
                
              '''
              if dataset_type == DatasetType.CELEBA: #  先不考慮 landmark
                m1, m2, m3, m4, m5, m6, m7, m8, m9, m10 = landmark_offset
                save_path = get_save_path(save_idx, save_dir)
                save_fp.write("{} {} {} {:.4f} {:.4f} {:.4f} {:.4f} ".format(save_path, save_flag, DatasetType.CELEBA.value, osx1, osy1, osx2, osy2))
                save_fp.write("{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(m1, m2, m3, m4, m5, m6, m7, m8, m9, m10))
              '''              
              img_resize = cv2.resize(img_crop, (MODEL_H, MODEL_W), interpolation=cv2.INTER_LINEAR)
              cv2.imwrite(save_path, img_resize)
    
            view_bar_gen_add_anchor_face('processing', j+1,  len(lines), pos_idx, part_idx, neg_idx, anchor_face_idx)                   
  print()

end = time.time() #計算運行時間用
elapsed = end - start #計算運行時間用
print ("Time taken: ", elapsed, "seconds.") #計算運行時間用