import time #計算運行時間用
start = time.time() #計算運行時間用

import os, re, sys
import tensorflow as tf
import numpy as np
import keras
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import PReLU, UpSampling2D, Concatenate
from focal_losses_part import categorical_focal_loss
from keras.preprocessing.image import ImageDataGenerator
import model
import tensorflow_addons as tfa

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
     
def acc(y_true, y_pred):
    #custom_sparse_categorical_acc
    y_pred = tf.squeeze(y_pred,[1,2])
    return K.cast(K.equal(K.max(y_true, axis=-1),
                          K.cast(K.argmax(y_pred, axis=-1), K.floatx())),
                  K.floatx())
        
def acc_nan(y_true, y_pred):
    a = float('nan')
    return a

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
    image_36 = tf.image.resize(image, [36,36])  
    image_36 = tf.image.random_contrast(image_36, lower=0.5, upper=1.5)
    image_36 = tf.image.random_brightness(image_36, max_delta=0.2)
    image_36 = tf.image.random_hue(image_36,max_delta= 0.2)
    image_36 = tf.image.random_saturation(image_36,lower = 0.5, upper= 1.5)
    #tf.print('roi1',roi1)
    if random_num == 1:
        image_36 = tf.image.flip_left_right(image_36)  # 隨機翻左翻右
        roi_flip = [-roi1[2],roi1[1],-roi1[0],roi1[3]]
        roi_com1 = combine_roi_cls(roi_flip, label_cls1)
        #tf.print('roi_com1_flip',roi_com1)
    else:
        roi_com1 = combine_roi_cls(roi1, label_cls1)
        #tf.print('roi1 new',roi1)
        
        
    image_36 = (tf.cast(image_36, tf.float32)-127.5) / 128
    
    
    
    '''
    label_cls2 = tf.cast(image_features['image/label_cls2'], tf.int64)    
    roi2 = tf.cast(image_features['image/roi2'], tf.float32)
    roi_com2 = combine_roi_cls(roi2, label_cls2)
    '''


    return image_36, (label_cls1, roi_com1)




def get_TFrecord_size(dataset):
    dataset_num = 0    
    for record in dataset:
        dataset_num += 1
        sys.stdout.write ("\rsize of the dataset...:{}".format(dataset_num))
    print ("Total size of the dataset:", dataset_num)
    return dataset_num

if __name__ == "__main__":
    
    assert tf.executing_eagerly();
        
    tfrecords_path = 'tfdata_36_145050_1bbox.tfrecords' ## 改
    #validation_path = 'tfdata_36_821950_1bbox.tfrecords'
    print('TFRecords path is:', tfrecords_path)
    tfrecords_strlist = re.split(r"[_,.]",tfrecords_path)
    #validation_strlist = re.split(r"[_,.]",validation_path)
    img_size = tfrecords_strlist[1]    
    dataset_num = tfrecords_strlist[2] 
    #validation_size = validation_strlist[1] 
    #validation_num = validation_strlist[2]
    
    batch_size = 256
    epochs_num = 1000
    
    
    path_name = 'Onet36_'+str(epochs_num)+'epochs_'+str(dataset_num) ## 檔案名稱更改
  
    model = model.Onet36_DepthConv()
    # 預訓練檔
    pretrain_ckpt_path = ("ckpt\\"+"Onet36_1000epochs_145050Onet36_5m30d_de"+"\\cp-{epoch:04d}.ckpt".format(epoch = 504))

    #pretrain_ckpt_path = ("ckpt\\"+"net16_1000epochs_5637887net16_4m8dhard"+"\\cp-{epoch:04d}.ckpt".format(epoch = 123))
    #model.load_weights(pretrain_ckpt_path)
    #model.summary()
               
 #-------------------------------------------------------------------------------------------------------------------   
    #train dataset            
    dataset = tf.data.TFRecordDataset(tfrecords_path)    
    #dataset_num = get_TFrecord_size(dataset)    
    dataset = dataset.map(lambda x: extract_features(x, int(img_size)))     
    dataset = dataset.shuffle(buffer_size=10000) 
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size, drop_remainder=True)
    
    
    #validation dataset
    '''
    validation_dataset = tf.data.TFRecordDataset(validation_path)     
    validation_dataset = validation_dataset.map(lambda y: extract_features(y, int(validation_size)))     
    validation_dataset = validation_dataset.shuffle(buffer_size=30000) 
    validation_dataset = validation_dataset.repeat()
    validation_dataset = validation_dataset.batch(batch_size, drop_remainder=True)
    '''  
#---------------------------------------------------------------------------------------------------------------------
    file_name = (path_name[0:(len(path_name))]+"Onet36_5m30d_de")  ## 檔案名稱更改
    
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
        model.compile(loss=[categorical_focal_loss(gamma=0, alpha=1,keep_ratio=0.7), bbox_mse],
                           optimizer='adam', metrics=[[cal_accuracy],[acc_nan]], callbacks = [cp_callback]) 
    else:
        print('error\n')
        

    
     
    
    #model.fit(dataset, validation_data  = validation_dataset , epochs = epochs_num, steps_per_epoch = int(dataset_num)//batch_size, validation_steps = int(validation_num)//batch_size, callbacks=[cp_callback, tensorboard_callback])  
    model.fit(dataset , epochs = epochs_num, steps_per_epoch = int(dataset_num)//batch_size , callbacks=[cp_callback, tensorboard_callback])     
    # steps_per_epoch = 訓練資料量除以 batch_size     

end = time.time() #計算運行時間用
elapsed = end - start #計算運行時間用
print ("Time taken: ", elapsed, "seconds.") #計算運行時間用