import time
import os, re, sys, cv2, glob
import tensorflow as tf
import numpy as np
import model
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import PReLU, UpSampling2D, Concatenate
import pickle

def detect_face(img, Pnet16 ,Rnet24 ,Onet36  ):
    """Detects faces in an image, and returns bounding boxes and points for them.
    img: input image
    minsize: minimum faces' size
    pnet, rnet, onet: caffemodel
    threshold: threshold=[th1, th2, th3], th1-3 are three steps's threshold
    factor: the factor used to create a scaling pyramid of face sizes to detect in the image.
    """

    threshold = [0.65,0.4,0.5] # FDDB測試為 0.65 0.25 0.2
    #factor = 0.79    
    factor = 0.709 ##初步測試期間減少圖像金字塔的密度，用以加快運算速度
    count = 0
    factor_count=0
    total_boxes=np.empty((0,9))
    h=img.shape[0]
    w=img.shape[1]
    print(h,w)
    t3 = 0
    minl=np.amin([h, w])
    m=16.0/20 #默認 minsize = 20
    minl=minl*m     #使 minl = 12
    # create scale pyramid
    scales=[]    
    temp = 0
    while minl>= 20:
       
        scales += [m*np.power(factor, factor_count)]    #原版 MTCNN 默認 factor = 0.709
        minl = minl*factor
        factor_count += 1    
    # first stage    
    for scale in scales:
        
        hs=int(np.ceil(h*scale))    #np.ceil //向上取整數：0.7>>1 , -0.1 >> 0
        ws=int(np.ceil(w*scale))
        #print(hs,ws)
        
        
        im_data = imresample(img, (hs, ws))
        #im_data = (im_data-127.5)*0.0078125                
        im_data = (im_data-127.5) / 128   
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
    print('numbox Pnet:',numbox)
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
        #rnet_result = total_boxes
        mv = out1[:,ipass[0]]
        #print('Onet_num :', total_boxes.shape[0])
        
        if total_boxes.shape[0]>0:   
            pick = nms(total_boxes, 0.7, 'Union')
            total_boxes = total_boxes[pick,:]
            total_boxes = bbreg(total_boxes.copy(), np.transpose(mv[:,pick]))
            total_boxes = rerec(total_boxes.copy())
          
              
    numbox = total_boxes.shape[0]  
    print('numbox Pnet:',numbox)
    #return rnet_result           
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
    numbox = total_boxes.shape[0] 
    print('numbox Onet:',numbox)                  
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
        if method == 'Min':
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
    #cv2.putText(img, str(np.round(score, 2)), (bbox[0], bbox[1]), cv2.FONT_HERSHEY_TRIPLEX, 1, color=(0,255,0))
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


if __name__ == "__main__":
    
    Pnet16 = model.Pnet16_DepthConv()
    Rnet24 = model.Rnet24_DepthConv()
    Onet36 = model.Onet36_DepthConv()
    Pnet16_path = ("ckpt\\"+"paper"+"\\cp-{epoch:04d}.ckpt".format(epoch = 266))
    Rnet24_path = ("ckpt\\"+"paper"+"\\cp-{epoch:04d}.ckpt".format(epoch = 79))
    Onet36_path = ("ckpt\\"+"paper"+"\\cp-{epoch:04d}.ckpt".format(epoch = 436))
    result_dir = 'result_dir' 
    Pnet16.load_weights(Pnet16_path)
    Rnet24.load_weights(Rnet24_path)
    Onet36.load_weights(Onet36_path)    
 
    test_dir = 'test_dir_detection'
    result_dir = 'result_dir_detection'
    #print(model_APN20.summary())
    img_paths = glob.glob(test_dir + '/*.jpg')

    t1 = time.time() #計算運行時間用
    
    for img_path in img_paths:
        print(img_path)
        #img = misc.imread(img_path)
        img = cv2.imread(img_path)
        #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        #bboxes, landmarks = mtcnn.run(img)
        #print('img shape', img.shape)
        save_name = img_path[:-4]
        bboxes = detect_face(img, Pnet16, Rnet24, Onet36)
        
        #print("bboxes_num:",bboxes.shape[0])  
        print(bboxes[:,2]-bboxes[:,0])
        draw_cv2_bboxes(img, bboxes)        
        save_cv2_annotations(img, img_path, result_dir)
        
    t2 = time.time() #計算運行時間用   
print ("Time taken: ", t2-t1, "seconds.") #計算運行時間用
