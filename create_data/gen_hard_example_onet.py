#coding:utf-8
import sys
#sys.path.append("../")
sys.path.insert(0,'..')
import numpy as np
import argparse
import os
import pickle as pickle
import cv2
from loader import TestLoader
import tensorflow as tf
from utils import *
from data_utils import *
from time import sleep
from tqdm import tqdm, trange
import model



#def t_net(pnet,rnet, onet): 
def t_net(model_Pnet16,model_Rnet24,model_Onet36):    
    detections = []
    image_size = 36

    
    #############    
    
    basedir = '../../dataset'
    filename = './wider_face_train_bbx_gt.txt'  # 获取检测框的ground truth值
    data = read_annotation(basedir, filename)  # 读pic的文件名，和box的ground truth值，data['images']==all image pathes
    
    #############                              #  data['bboxes'] =all image bboxes
    im_idx_list = data['images']        # data保存WIDER_train里面所有的image，读pic的文件名，和box的ground truth值，
    gt_boxes_list = data['bboxes']      # data['images']==all image pathes
                                        # data['bboxes'] =all image bboxes（4列）

    num_of_images = len(im_idx_list)  # 12880
    print("processing %d images in total" % num_of_images)

    neg_label_file = "face%d/neg_hard_example%d.txt" % (image_size, image_size)     # neg_label_file == '24/neg_24.txt'
    neg_file = open(neg_label_file, 'w')                     # 打开保存pos， neg， part的文件夹
    pos_label_file = "face%d/pos_hard_example%d.txt" % (image_size, image_size)
    pos_file = open(pos_label_file, 'w')
    part_label_file = "face%d/part_hard_example%d.txt" % (image_size, image_size)
    part_file = open(part_label_file, 'w')

    
   
 #   assert len(det_boxes) == num_of_images # len(det_boxes) == num_of_images应该为TRUE，如果不是，则返回错误。"incorrect detections or ground truths"

    n_idx = 0  # index of neg, pos and part face, used as their image names
    p_idx = 0
    d_idx = 0
    for im_idx, gts in tqdm(zip(im_idx_list, gt_boxes_list),total = len(im_idx_list)):
    #for i in tqdm(range(5)):
        gts = np.array(gts, dtype=np.float32).reshape(-1, 4)             # gts保存txt里面的存储的label
        img = cv2.imread(im_idx)
        dets = detect_face(img,model_Pnet16,model_Rnet24, model_Onet36)
        #dets = mtcnn_net.detect_face(img,pnet,rnet, onet)  
        if dets.shape[0] == 0:
            continue
        
        scores = dets[:,4]
        dets = convert_to_square(dets)       # 通过偏移左上角，把框框变成正方形
        dets[:, 0:4] = np.round(dets[:, 0:4])  # 取出detections的前4维信息
        neg_num = 0
        for box, score in zip(dets,scores): # 遍历detections的前4维信息
            #print('score',score)
            x_left, y_top, x_right, y_bottom, score = box.astype(int)
            width = x_right - x_left + 1
            height = y_bottom - y_top + 1
            
            if width < 20 or x_left < 0 or y_top < 0 or x_right > img.shape[1] - 1 or y_bottom > img.shape[0] - 1:
                continue

            Iou = IoU(box, gts)      # compute intersection over union(IoU) between current box and all gt boxes

            cropped_im = img[y_top:y_bottom + 1, x_left:x_right + 1, :]  # 拿到检测框
            resized_im = cv2.resize(cropped_im, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    
            if np.max(Iou) < 0.3:   # save negative images and write label，Iou with all gts must below 0.3   
                save_file = get_path(neg_dir, "%s.jpg" % n_idx)  # save the examples，'24/negative/0.jpg'
                neg_file.write(save_file + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1
                neg_num += 1
            
            else:
                
                
                idx = np.argmax(Iou)  # find gt_box with the highest iou,找正例
                assigned_gt = gts[idx]
                x1, y1, x2, y2 = assigned_gt

                offset_x1 = (x1 - x_left) / float(width)  # compute bbox reg label计算偏移，x1是label值
                offset_y1 = (y1 - y_top) / float(height)
                offset_x2 = (x2 - x_right) / float(width)
                offset_y2 = (y2 - y_bottom) / float(height)
                  
                if (np.max(Iou) >= 0.65):
                    save_file = get_path(pos_dir, "%s.jpg" % p_idx)
                    pos_file.write(save_file + ' 1 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    p_idx += 1
                
                elif (np.max(Iou) >= 0.4):
                    save_file = get_path(part_dir, "%s.jpg" % d_idx)
                    part_file.write(save_file + ' -1 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    d_idx += 1
                 
               
               
                #elif np.max(Iou) >= 0.4:
                #    save_file = os.path.join(part_dir, "%s.jpg" % d_idx)
                #    part_file.write(save_file + ' -1 %.4f %.4f %.4f %.4f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2, np.max(Iou)))
                #    cv2.imwrite(save_file, resized_im)
                #    d_idx += 1
            
            
        sleep(0.001)
    neg_file.close()
    part_file.close()
    pos_file.close()
    
def detect_face(img, Pnet16 ,Rnet24 ,Onet36 ):
    """Detects faces in an image, and returns bounding boxes and points for them.
    img: input image
    minsize: minimum faces' size
    pnet, rnet, onet: caffemodel
    threshold: threshold=[th1, th2, th3], th1-3 are three steps's threshold
    factor: the factor used to create a scaling pyramid of face sizes to detect in the image.
    """

    threshold = [0.6,0.2,0.2]
    #factor = 0.79    
    factor = 0.709 ##初步測試期間減少圖像金字塔的密度，用以加快運算速度
    count = 0
    factor_count=0
    total_boxes=np.empty((0,9))
    h=img.shape[0]
    w=img.shape[1]

    minl=np.amin([h, w])
    m=16.0/20 #默認 minsize = 20
    minl=minl*m     #使 minl = 12
    # create scale pyramid
    scales=[]    
    temp = 0
    while minl>=20:
       
        #m=16.0/16
        '''
        if( temp == True):
            m = 16.0/12
            minl=minl*m
            factor_count -= 1
            temp = False
        '''
        scales += [m*np.power(factor, factor_count)]    #原版 MTCNN 默認 factor = 0.709
        minl = minl*factor
        factor_count += 1    
    # first stage    
    for scale in scales:
        
        hs=int(np.ceil(h*scale))    #np.ceil //向上取整數：0.7>>1 , -0.1 >> 0
        ws=int(np.ceil(w*scale))
        #boxes_list = []
        #print(hs,ws)
        
        im_data = imresample(img, (hs, ws))
        #im_data = (im_data-127.5)*0.0078125                
        im_data = (im_data-127.5) / 128 
        im_data = im_data.astype(np.float32)
        img_x = np.expand_dims(im_data, 0)        
        
        predictions = Pnet16(img_x,training=False) # out0:reg, out1:cls_score  

        out0 = predictions[0].numpy()
        out1 = predictions[1].numpy()

        boxes, _ = generateBoundingBox(out0[0,:,:,1].copy(), out1[0,:,:,:].copy(), scale, threshold[0])
            #boxes_list.append(boxes)
        #boxes = np.concatenate((boxes_list[0], boxes_list[1]))
        count = count+boxes.shape[0]
        
        pick = nms(boxes.copy(), 0.5, 'Union')
        if boxes.size>0 and pick.size>0:
            boxes = boxes[pick,:]
            total_boxes = np.append(total_boxes, boxes, axis=0)  
            
    #print('count :',count)
    numbox = total_boxes.shape[0]    
    if numbox>0:
        #total_boxes_P = total_boxes  
        pick = nms(total_boxes.copy(), 0.7, 'Union')
        total_boxes = total_boxes[pick,:]
        regw = total_boxes[:,2]-total_boxes[:,0]
        regh = total_boxes[:,3]-total_boxes[:,1]
        qq1 = total_boxes[:,0]+total_boxes[:,5]*regw
        qq2 = total_boxes[:,1]+total_boxes[:,6]*regh
        qq3 = total_boxes[:,2]+total_boxes[:,7]*regw
        qq4 = total_boxes[:,3]+total_boxes[:,8]*regh
        total_boxes = np.transpose(np.vstack([qq1, qq2, qq3, qq4, total_boxes[:,4]]))
                
        total_boxes_pnet = total_boxes ### 輸出 APN24 的偵測結果
        total_boxes = rerec(total_boxes.copy())
        total_boxes[:,0:4] = np.fix(total_boxes[:,0:4]).astype(np.int32)
        dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad(total_boxes.copy(), w, h)
        
    #return total_boxes
    numbox = total_boxes.shape[0]
 
    if numbox>0:
        #total_boxes = np.fix(total_boxes).astype(np.int32)
        #dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad(total_boxes.copy(), w, h)
        tempimg = np.zeros((24,24,3,numbox))
        #print('tmph:',tmph)
        for k in range(0,numbox):
            tmp = np.zeros((int(tmph[k]),int(tmpw[k]),3))
            #tmp = np.zeros((abs(int(tmph[k])),abs(int(tmpw[k])),3))
            tmp[dy[k]-1:edy[k],dx[k]-1:edx[k],:] = img[y[k]-1:ey[k],x[k]-1:ex[k],:]
            if tmp.shape[0]>=0 and tmp.shape[1]>=0 or tmp.shape[0]==0 and tmp.shape[1]==0:
                tempimg[:,:,:,k] = imresample(tmp, (24, 24))
            else:
                continue
        #print('tempimg',tempimg,'\n')
        tempimg = (tempimg-127.5)*0.0078125
        #print('tempimg:',tempimg.shape)
        tempimg1 = np.transpose(tempimg, (3,0,1,2))
        #print('tempimg1:',tempimg1.shape)
        tempimg1 = tempimg1.astype(np.float32)
        prediction = Rnet24(tempimg1,training=False)
        
        out0 = prediction[0].numpy()
        out1 = prediction[1].numpy()
        #print('out0',out0.shape)
        out0 = np.transpose(out0)
        out1 = np.transpose(out1)
        #print('out0.T',out0.shape)
        out0 = out0[:,0,0,:]
        out1 = out1[:,0,0,:]
        #print(out0[1,:])
        score = out0[1,:]
        
        
        ipass = np.where(score>threshold[1])
        
        total_boxes = np.hstack([total_boxes[ipass[0],0:4].copy(), np.expand_dims(score[ipass].copy(),1)])
        #print(ipass[0])
        #print(out1)
        mv = out1[:,ipass[0]]
        #print('Onet_num :', total_boxes.shape[0])
        
        if total_boxes.shape[0]>0:   
            pick = nms(total_boxes, 0.7, 'Union')
            total_boxes = total_boxes[pick,:]
            total_boxes = bbreg(total_boxes.copy(), np.transpose(mv[:,pick]))
            total_boxes = rerec(total_boxes.copy())
          
              
    numbox = total_boxes.shape[0]    
    #return rnet_result 只取Rnet資料，取消註解 #return total_boxes      
    #return total_boxes
    
    if numbox>0:
        total_boxes = np.fix(total_boxes).astype(np.int32)
        dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad(total_boxes.copy(), w, h)
        tempimg = np.zeros((36,36,3,numbox))
        #print('tmph:',tmph)
        for k in range(0,numbox):
            tmp = np.zeros((int(tmph[k]),int(tmpw[k]),3))
            #tmp = np.zeros((abs(int(tmph[k])),abs(int(tmpw[k])),3))
            tmp[dy[k]-1:edy[k],dx[k]-1:edx[k],:] = img[y[k]-1:ey[k],x[k]-1:ex[k],:]
            if tmp.shape[0]>=0 and tmp.shape[1]>=0 or tmp.shape[0]==0 and tmp.shape[1]==0:
                tempimg[:,:,:,k] = imresample(tmp, (36, 36))
            else:
                continue
        #print('tempimg',tempimg,'\n')
        tempimg = (tempimg-127.5)*0.0078125
        #print('tempimg:',tempimg.shape)
        tempimg1 = np.transpose(tempimg, (3,0,1,2))
        #print('tempimg1:',tempimg1.shape)
        prediction = Onet36(tempimg1,training=False)
        
        out0 = prediction[0].numpy()
        out1 = prediction[1].numpy()
       
        
       
        #print(out0.shape)
        out0 = np.transpose(out0)
        out1 = np.transpose(out1)
        
        out0 = out0[:,0,0,:]
        out1 = out1[:,0,0,:]
              
        
        score = out0[1,:]
        
        ipass = np.where(score>threshold[2])
        
        total_boxes = np.hstack([total_boxes[ipass[0],0:4].copy(), np.expand_dims(score[ipass].copy(),1)])
        #print(ipass[0])
        mv = out1[:,ipass[0]]
        #print('Onet_num :', total_boxes.shape[0])
        
        if total_boxes.shape[0]>0:   
            total_boxes = bbreg(total_boxes.copy(), np.transpose(mv))
            pick = nms(total_boxes.copy(), 0.7, 'Min')
            total_boxes = total_boxes[pick,:]
                      
    return total_boxes
      
    
def bbreg(boundingbox,reg):
    """Calibrate bounding boxes"""
    if reg.shape[1]==1:
        reg = np.reshape(reg, (reg.shape[2], reg.shape[3]))

    w = boundingbox[:,2]-boundingbox[:,0]+1
    h = boundingbox[:,3]-boundingbox[:,1]+1
    b1 = boundingbox[:,0]+reg[:,0]*w
    b2 = boundingbox[:,1]+reg[:,1]*h
    b3 = boundingbox[:,2]+reg[:,2]*w
    b4 = boundingbox[:,3]+reg[:,3]*h
    boundingbox[:,0:4] = np.transpose(np.vstack([b1, b2, b3, b4 ]))
    return boundingbox

def generateBoundingBox(imap, reg, scale, t):
    """Use heatmap to generate bounding boxes"""
    #stride=2
    #cellsize=12 # 這是原版 MTCNN
    stride=2 # 原先的面積是 PNet:12*12，現在是 APN24:24*24，所以 stride*2
    cellsize=16 #APN24 的緣故，所以改成 24

    imap = np.transpose(imap)
    dx1 = np.transpose(reg[:,:,0])
    dy1 = np.transpose(reg[:,:,1])
    dx2 = np.transpose(reg[:,:,2])
    dy2 = np.transpose(reg[:,:,3])
    y, x = np.where(imap >= t)
    if y.shape[0]==1:
        dx1 = np.flipud(dx1)
        dy1 = np.flipud(dy1)
        dx2 = np.flipud(dx2)
        dy2 = np.flipud(dy2)
    score = imap[(y,x)]
    reg = np.transpose(np.vstack([ dx1[(y,x)], dy1[(y,x)], dx2[(y,x)], dy2[(y,x)] ]))
    if reg.size==0:
        reg = np.empty((0,3))
    bb = np.transpose(np.vstack([y,x]))
    q1 = np.fix((stride*bb+1)/scale)
    q2 = np.fix((stride*bb+cellsize-1+1)/scale)
    boundingbox = np.hstack([q1, q2, np.expand_dims(score,1), reg])
    return boundingbox, reg
 
# function pick = nms(boxes,threshold,type)
def nms(boxes, threshold, method):
    if boxes.size==0:
        return np.empty((0,3))
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    s = boxes[:,4]
    area = (x2-x1+1) * (y2-y1+1)
    I = np.argsort(s)
    pick = np.zeros_like(s, dtype=np.int16)
    counter = 0
    while I.size>0:
        i = I[-1]
        pick[counter] = i
        counter += 1
        idx = I[0:-1]
        xx1 = np.maximum(x1[i], x1[idx])
        yy1 = np.maximum(y1[i], y1[idx])
        xx2 = np.minimum(x2[i], x2[idx])
        yy2 = np.minimum(y2[i], y2[idx])
        w = np.maximum(0.0, xx2-xx1+1)
        h = np.maximum(0.0, yy2-yy1+1)
        inter = w * h
        if method is 'Min':
            o = inter / np.minimum(area[i], area[idx])
        else:
            o = inter / (area[i] + area[idx] - inter)
        I = I[np.where(o<=threshold)]
    pick = pick[0:counter]
    return pick

# function [dy edy dx edx y ey x ex tmpw tmph] = pad(total_boxes,w,h)
def pad(total_boxes, w, h):
    """Compute the padding coordinates (pad the bounding boxes to square)"""
    tmpw = (total_boxes[:,2]-total_boxes[:,0]+1).astype(np.int32)
    tmph = (total_boxes[:,3]-total_boxes[:,1]+1).astype(np.int32)
    numbox = total_boxes.shape[0]

    dx = np.ones((numbox), dtype=np.int32)
    dy = np.ones((numbox), dtype=np.int32)
    edx = tmpw.copy().astype(np.int32)
    edy = tmph.copy().astype(np.int32)

    x = total_boxes[:,0].copy().astype(np.int32)
    y = total_boxes[:,1].copy().astype(np.int32)
    ex = total_boxes[:,2].copy().astype(np.int32)
    ey = total_boxes[:,3].copy().astype(np.int32)

    tmp = np.where(ex>w)
    edx.flat[tmp] = np.expand_dims(-ex[tmp]+w+tmpw[tmp],1)
    ex[tmp] = w
    
    tmp = np.where(ey>h)
    edy.flat[tmp] = np.expand_dims(-ey[tmp]+h+tmph[tmp],1)
    ey[tmp] = h

    tmp = np.where(x<1)
    dx.flat[tmp] = np.expand_dims(2-x[tmp],1)
    x[tmp] = 1

    tmp = np.where(y<1)
    dy.flat[tmp] = np.expand_dims(2-y[tmp],1)
    y[tmp] = 1
    
    return dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph

# function [bboxA] = rerec(bboxA)
def rerec(bboxA):
    """Convert bboxA to square."""
    h = bboxA[:,3]-bboxA[:,1]
    w = bboxA[:,2]-bboxA[:,0]
    l = np.maximum(w, h)
    bboxA[:,0] = bboxA[:,0]+w*0.5-l*0.5
    bboxA[:,1] = bboxA[:,1]+h*0.5-l*0.5
    bboxA[:,2:4] = bboxA[:,0:2] + np.transpose(np.tile(l,(2,1)))
    return bboxA

def imresample(img, sz):
    im_data = cv2.resize(img, (sz[1], sz[0]), interpolation=cv2.INTER_AREA) #@UndefinedVariable
    return im_data

    # This method is kept for debugging purpose
#     h=img.shape[0]
#     w=img.shape[1]
#     hs, ws = sz
#     dx = float(w) / ws
#     dy = float(h) / hs
#     im_data = np.zeros((hs,ws,3))
#     for a1 in range(0,hs):
#         for a2 in range(0,ws):
#             for a3 in range(0,3):
#                 im_data[a1,a2,a3] = img[int(floor(a1*dy)),int(floor(a2*dx)),a3]
#     return im_data



def draw_cv2_bboxes(img, bboxes):
  for bbox in bboxes:
    score = bbox[4]
    bbox = list(map(int, bbox[0:4]))
    cv2.putText(img, str(np.round(score, 2)), (bbox[0], bbox[1]), cv2.FONT_HERSHEY_TRIPLEX, 1, color=(0,255,0))
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2],bbox[3]), (0,255,0), thickness=2)

def draw_cv2_landmarks(img, landmarks):
  for j in range(landmarks.shape[1]):
    for i in range(5):
      landmark = (int(landmarks[i,j]), int(landmarks[i+5,j]))
      cv2.circle(img, landmark, radius=3, color=(0,255,0), thickness=-1)

def save_cv2_annotations(img, img_path, result_dir):
  if not os.path.exists(result_dir):
     os.mkdir(result_dir)
  save_path = os.path.join(result_dir, os.path.basename(img_path))
  cv2.imwrite(save_path, img)
    
if __name__ == '__main__':

    image_size = 36

    base_dir = "../../dataset/WIDER_train"
    data_dir = 'face%s' % str(image_size)  
    
    neg_dir = get_path(data_dir, 'negative/hard_example')
    pos_dir = get_path(data_dir, 'positive/hard_example')
    part_dir = get_path(data_dir, 'part/hard_example')
    

    for dir_path in [neg_dir, pos_dir, part_dir]:      # create dictionary shuffle
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)  

    print('Called with argument:')
    
    #-------------------------------------
    '''
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions()
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet,rnet, onet = mtcnn_net.create_mtcnn(sess, None)
    '''
            
    model_Pnet16 = model.Pnet16_DepthConv()
    model_Rnet24 = model.Rnet24_DepthConv()
    model_Onet36 = model.Onet36_DepthConv()
    Pnet16_ckpt_path = ("ckpt\\"+"paper"+"\\cp-{epoch:04d}.ckpt".format(epoch = 266))
    Rnet24_ckpt_path = ("ckpt\\"+"paper"+"\\cp-{epoch:04d}.ckpt".format(epoch = 79))
    Onet36_ckpt_path = ("ckpt\\"+"paper"+"\\cp-{epoch:04d}.ckpt".format(epoch = 436))
    model_Pnet16.load_weights(Pnet16_ckpt_path)
    model_Rnet24.load_weights(Rnet24_ckpt_path)
    model_Onet36.load_weights(Onet36_ckpt_path)
    t_net(model_Pnet16,model_Rnet24,model_Onet36)
    #t_net(pnet,rnet, onet)