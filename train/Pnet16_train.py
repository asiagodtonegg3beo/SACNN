import time #計算運行時間用
start = time.time() #計算運行時間用

import os, re, sys
import random
import tensorflow as tf
import numpy as np
import keras
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import PReLU, UpSampling2D, Concatenate
from focal_losses_part import categorical_focal_loss
from keras.preprocessing.image import ImageDataGenerator
import tensorflow_addons as tfa
import model
tf.executing_eagerly()

def combine_roi_cls(roi, label_cls_ep):
    label_cls_ep = tf.cast(label_cls_ep, tf.float32)
    label_cls_ep_dim = tf.expand_dims(label_cls_ep,0)
    label_cls_ep_dim = tf.expand_dims(label_cls_ep_dim,0)

    roi_com = tf.concat([tf.expand_dims(roi,0),label_cls_ep_dim],1)
    roi_com = tf.squeeze(roi_com,0)
    return roi_com

    
def bbox_mse(y_true, y_pred):      #fixed_ver1   
    roi_target = y_true[:,0:4]
    cls_label_target = tf.abs(y_true[:,4:5])
    cls_label_target = tf.squeeze(cls_label_target,1)
    mask_bool = tf.reduce_any(tf.cast(cls_label_target, dtype=bool))
    mask = tf.cast(mask_bool, dtype=tf.float32)
    pred_picked = tf.cond(tf.equal(mask,0),true_fn=lambda:tf.zeros_like(y_pred, dtype=tf.float32),false_fn=lambda:tf.boolean_mask(y_pred,cls_label_target))
    pred_picked = tf.squeeze(pred_picked,[1,2]) # fix
    target_picked = tf.cond(tf.equal(mask,0),true_fn=lambda:tf.zeros_like(roi_target, dtype=tf.float32),false_fn=lambda:tf.boolean_mask(roi_target,cls_label_target))
    
    return tf.reduce_mean(tf.square(pred_picked-target_picked))
def cls_ohem(cls_prob,label):
    '''计算类别损失
    参数：
      cls_prob：预测类别，是否有人
      label：真实值
    返回值：
      损失
    '''
    zeros=tf.zeros_like(label)
    #只把pos的label设定为1,其余都为0
    label_filter_invalid=tf.where(tf.less(label,0),zeros,label)
    #类别size[2*batch]
    num_cls_prob=tf.size(cls_prob)
    cls_prob_reshpae=tf.reshape(cls_prob,[num_cls_prob,-1])
    label_int=tf.cast(label_filter_invalid,tf.int32)
    #获取batch数
    num_row = tf.cast(cls_prob.get_shape()[0], tf.int32)
    #num_row=tf.to_int32(cls_prob.get_shape()[0])
    #对应某一batch而言，batch*2为非人类别概率，batch*2+1为人概率类别,indices为对应 cls_prob_reshpae
    #应该的真实值，后续用交叉熵计算损失
    row=tf.range(num_row)*2
    indices_=row+label_int
    #真实标签对应的概率
    label_prob=tf.squeeze(tf.gather(cls_prob_reshpae,indices_))
    loss=-tf.log(label_prob+1e-10)
    zeros=tf.zeros_like(label_prob,dtype=tf.float32)
    ones=tf.ones_like(label_prob,dtype=tf.float32)
    #统计neg和pos的数量
    valid_inds=tf.where(label<zeros,zeros,ones)
    num_valid=tf.reduce_sum(valid_inds)
    #选取70%的数据
    keep_num=tf.cast(num_valid*num_keep_radio,dtype=tf.int32)
    #只选取neg，pos的70%损失
    loss=loss*valid_inds
    loss,_=tf.nn.top_k(loss,k=keep_num)
    return tf.reduce_mean(loss)
    
def bbox_ohem(y_true, y_pred):
    '''计算box的损失'''
    roi_target = y_true[:,0:4]
    cls_label_target = y_true[:,4:5]
    y_pred = tf.squeeze(y_pred)
    #tf.print('y_pred,',y_pred)
    #tf.print('roi_target,',roi_target)
    #tf.print('cls_label_target,',cls_label_target)
    zeros_index=tf.zeros_like(cls_label_target,dtype=tf.float32)
    ones_index=tf.ones_like(cls_label_target,dtype=tf.float32)
    #保留pos和part的数据
    valid_inds=tf.where(tf.equal(tf.abs(cls_label_target),1),ones_index,zeros_index)
    #计算平方差损失
    #tf.print('valid_inds,',valid_inds)
    square_error=tf.square(y_pred-roi_target)
    square_error=tf.reduce_sum(square_error,axis=1)
    #保留的数据的个数
    num_valid=tf.reduce_sum(valid_inds)
    keep_num=tf.cast(num_valid,dtype=tf.int32)
    #保留pos和part部分的损失
    square_error=square_error*valid_inds
    square_error,_=tf.nn.top_k(square_error,k=keep_num)
    return tf.reduce_mean(square_error)
    '''计算box的损失'''
    roi_target = y_true[:,0:4]
    cls_label_target = tf.abs(y_true[:,4:5])
    #cls_label_target = tf.squeeze(cls_label_target,1)
    print('cls_label_targetcls_label_targetcls_label_targetcls_label_target,',cls_label_target)
    zeros_index=tf.zeros_like(cls_label_target,dtype=tf.float32)
    ones_index=tf.ones_like(cls_label_target,dtype=tf.float32)
    #保留pos和part的数据
    valid_inds=tf.where(tf.equal(tf.abs(cls_label_target),1),ones_index,zeros_index)
    #计算平方差损失
    
    square_error=tf.square(y_pred-roi_target)
    square_error=tf.reduce_sum(square_error,axis=1)
    #保留的数据的个数
    num_valid=tf.reduce_sum(valid_inds)
    keep_num=tf.cast(num_valid,dtype=tf.int32)
    #保留pos和part部分的损失
    square_error=square_error*valid_inds
    square_error,_=tf.nn.top_k(square_error,k=keep_num)
    return tf.reduce_mean(square_error)
   
def cal_accuracy(y_true, y_pred):
    '''計算分類準確率'''
    #預測最大概率的類別，0代表無人，1代表有人
    y_pred = tf.squeeze(y_pred,[1,2])
    pred=tf.argmax(y_pred,axis=1)
    label_int=tf.cast(y_true,tf.int64)
    #保留label>=0的數據，即pos和neg的數據
    #tf.print('\n label_int',label_int)
    #tf.print('\n pred',pred)
    cond=tf.where(tf.greater_equal(label_int,0))
    picked=tf.squeeze(cond)
    
    #tf.print('\n picked',picked[:,0:1])
    #獲取pos和neg的label值
    label_picked=tf.gather(label_int,picked[:,0:1])
    pred_picked=tf.gather(pred,picked[:,0:1])
    #tf.print('\n label_picked',label_picked)
    #tf.print('\n equal',tf.cast(tf.equal(label_picked,pred_picked),tf.float32))
    #計算準確率
    acc = K.cast(K.equal(K.cast(K.max(label_picked, axis=-1),K.floatx()), K.cast(pred_picked, K.floatx())),K.floatx())
    acc = tf.squeeze(acc)
    #tf.print('\n acc',acc)
    #accuracy_op=tf.reduce_mean(tf.cast(tf.equal(label_picked,pred_picked),tf.float32))
    
    return acc
    
def acc(y_true, y_pred):
    #custom_sparse_categorical_acc
    y_pred = tf.squeeze(y_pred,[1,2])
    a = K.cast(K.equal(K.max(y_true, axis=-1),K.cast(K.argmax(y_pred, axis=-1), K.floatx())),K.floatx())
    tf.print('\n a',a)
    return K.cast(K.equal(K.max(y_true, axis=-1),
                          K.cast(K.argmax(y_pred, axis=-1), K.floatx())),
                  K.floatx())
        
def acc_nan(y_true, y_pred):
    a = float('nan')
    return a


def extract_features(example, image_size):
    image_features = tf.io.parse_single_example(
        example,
        features={
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
            'image/label_cls1': tf.io.FixedLenFeature([], tf.int64),
            'image/roi1': tf.io.FixedLenFeature([4], tf.float32),
        }
    )
    image = tf.io.decode_raw(image_features['image/encoded'], tf.uint8)
    image = tf.reshape(image, [image_size, image_size, 3])
    
    label_cls1 = tf.cast(image_features['image/label_cls1'], tf.int64)    
    roi1 = tf.cast(image_features['image/roi1'], tf.float32)
    
    ## resize img
    random_num = tf.random.uniform(shape=[], maxval=2, dtype=tf.int32, seed=10)
    image_16 = tf.image.resize(image, [16,16])  
    image_16 = tf.image.random_contrast(image_16, lower=0.5, upper=1.5)
    image_16 = tf.image.random_brightness(image_16, max_delta=0.2)
    image_16 = tf.image.random_hue(image_16,max_delta= 0.2)
    image_16 = tf.image.random_saturation(image_16,lower = 0.5, upper= 1.5)
    #tf.print('roi1',roi1)
    if random_num == 1:
        image_16 = tf.image.flip_left_right(image_16)  # 隨機翻左翻右
        roi_flip = [-roi1[2],roi1[1],-roi1[0],roi1[3]]
        roi_com1 = combine_roi_cls(roi_flip, label_cls1)
        #tf.print('roi_com1_flip',roi_com1)
    else:
        roi_com1 = combine_roi_cls(roi1, label_cls1)
        #tf.print('roi1 new',roi1)
        
        
    image_16 = (tf.cast(image_16, tf.float32)-127.5) / 128
    
    
    
    '''
    label_cls2 = tf.cast(image_features['image/label_cls2'], tf.int64)    
    roi2 = tf.cast(image_features['image/roi2'], tf.float32)
    roi_com2 = combine_roi_cls(roi2, label_cls2)
    '''


    return image_16, (label_cls1, roi_com1)



def get_TFrecord_size(dataset):
    dataset_num = 0    
    for record in dataset:
        dataset_num += 1
        sys.stdout.write ("\rsize of the dataset...:{}".format(dataset_num))
    print ("Total size of the dataset:", dataset_num)
    return dataset_num

    
    
if __name__ == "__main__":
    
    assert tf.executing_eagerly();
        
    # train
    tfrecords_path = 'tfdata_16_4585077_1bbox.tfrecords' ## 改
    
    print('TFRecords path is:', tfrecords_path)
    tfrecords_strlist = re.split(r"[_,.]",tfrecords_path)
    
    img_size = tfrecords_strlist[1]    
    dataset_num = tfrecords_strlist[2] 
    
    # val
    validation_path = 'tfdata_16_509083_1bbox.tfrecords'
    validation_strlist = re.split(r"[_,.]",validation_path)
    validation_size = validation_strlist[1] 
    validation_num = validation_strlist[2]
    
    batch_size = 256
    # 訓練次數，原本值設800次
    epochs_num = 1600
    
    
    path_name = 'Pnet16_'+str(epochs_num)+'_epochs_'+str(dataset_num) ## 檔案名稱更改
  
    model = model.net16_2p_Depthwise()
    # 預訓練檔
    # 如果有用到預訓練檔，下面model.fit的epoch部分要加上原來的訓練次數 (epochs_num + 原先訓練次數)
    pretrain_ckpt_path = ("ckpt\\"+"Pnet16_1600_epochs_4585077net16_2p_Depthwise"+"\\cp-{epoch:04d}.ckpt".format(epoch = 354))
    model.load_weights(pretrain_ckpt_path)
    model.summary()
               
 #-------------------------------------------------------------------------------------------------------------------   
    #train dataset   
    
    dataset = tf.data.TFRecordDataset(tfrecords_path)    
    #dataset_num = get_TFrecord_size(dataset)   
    dataset = dataset.map(lambda x: extract_features(x, int(img_size)))     
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size, drop_remainder=True)
    
    
    
    
    #validation dataset

    validation_dataset = tf.data.TFRecordDataset(validation_path)     
    validation_dataset = validation_dataset.map(lambda y: extract_features(y, int(validation_size)))     
    validation_dataset = validation_dataset.shuffle(buffer_size=30000) 
    validation_dataset = validation_dataset.repeat()
    validation_dataset = validation_dataset.batch(batch_size, drop_remainder=True)
 
#---------------------------------------------------------------------------------------------------------------------
    file_name = (path_name[0:(len(path_name))]+"net16_2p_Depthwise")  ## 檔案名稱更改

    # ckpt
    ckpt_path = ("ckpt\\"+file_name+"\\cp-{epoch:04d}.ckpt")
    ckpt_path_check = ("ckpt\\"+file_name+"\\")
    if os.path.exists(ckpt_path_check) == False:
        os.makedirs(ckpt_path_check)  
    cp_callback = keras.callbacks.ModelCheckpoint(ckpt_path, save_weights_only=True, verbose=1, save_freq='epoch')
    
 #---------------------------------------------------------------------------------------------------------------------   
    # tb
    logs_dir = os.path.join("logs\\"+file_name+"_log\\")    
    if os.path.exists(logs_dir) == False:
          #os.mkdir(logs_dir)
          os.makedirs(logs_dir)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logs_dir, histogram_freq=1)
 #---------------------------------------------------------------------------------------------------------------------   
    loss_fun = input('loss_fun : 0.focal_loss 1.softmax crossentropy :')
    optimizer=tfa.optimizers.Lookahead(tfa.optimizers.RectifiedAdam(learning_rate=0.001))
    
    if (loss_fun == '0'): #focal_loss
        model.compile(optimizer=optimizer,loss=[categorical_focal_loss(gamma=2, alpha=0.25,keep_ratio=1),bbox_ohem]
        ,metrics=[[cal_accuracy],[acc_nan]])  
    
    elif(loss_fun == '1'): # softmax crossentropy
        model.compile(optimizer=optimizer,loss=[categorical_focal_loss(gamma=0, alpha=1,keep_ratio=0.7),bbox_ohem]
        ,metrics=[[cal_accuracy],[acc_nan]])  
    else:
        print('error\n')
        
       
    # validation fit
    model.fit(
        dataset, 
        validation_data  = validation_dataset ,
        initial_epoch = 354,
        epochs = epochs_num,
        steps_per_epoch = int(dataset_num)//batch_size,
        validation_steps = int(validation_num)//batch_size,
        callbacks=[cp_callback, tensorboard_callback])
    
    # train fit
    model.fit(
        dataset , 
        initial_epoch = 354,
        epochs = epochs_num, 
        steps_per_epoch = int(dataset_num)//batch_size , 
        callbacks=[cp_callback, tensorboard_callback])
        


       

end = time.time() #計算運行時間用
elapsed = end - start #計算運行時間用
print ("Time taken: ", elapsed, "seconds.") #計算運行時間用
