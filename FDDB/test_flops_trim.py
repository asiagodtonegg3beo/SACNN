import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import PReLU, UpSampling2D, Concatenate, MaxPooling2D, AveragePooling2D ,Conv2D ,DepthwiseConv2D,ReLU
from keras_flops import get_flops
import numpy as np

def net12(height, width): 
  inputs = tf.keras.Input((height, width, 3,));
  x = conv(10,(3,3),(1,1),'valid','PReLU')(inputs)
  x = MaxPooling2D(2, 2,'valid')(x)
  x = conv(16,(3,3),(1,1),'valid','PReLU')(x)
  x = conv(32,(3,3),(1,1),'valid','PReLU')(x)
  
  cls_result = conv(2,(1,1),(1,1),'valid','Softmax')(x)
  bbox_result = Conv2D(4, 1, 1, 'valid')(x)
  return tf.keras.Model(inputs = inputs, outputs = (cls_result , bbox_result))
  
def net16_DepthConv(height, width): 
    inputs = tf.keras.Input((height, width, 3,));
#----------------------------------------------------------------------   
    conv00 = conv(8,(1,1),(1,1),'valid','PReLU')(inputs)
    DepthConv0 = DepthConv(1,(3,3),(1,1),'valid','PReLU')(conv00)
    conv0 = conv(8,(1,1),(1,1),'valid','linear')(DepthConv0)
#----------------------------------------------------------------------       
    pool1 = MaxPooling2D(2, 2,'same')(conv0)
    #conv1 = conv(8,(1,1),(1,1),'valid','linear')(pool1)
#----------------------------------------------------------------------   
    conv01 = conv(8,(1,1),(1,1),'valid','PReLU')(pool1)
    DepthConv1 = DepthConv(1,(3,3),(1,1),'valid','PReLU')(conv01)
    conv1_0 = conv(8,(1,1),(1,1),'valid','linear')(DepthConv1)
#----------------------------------------------------------------------      
    DepthConv1_1 = DepthConv(1,(2,2),(1,1),'valid','PReLU')(conv01)
    conv1_1 = conv(8,(1,1),(1,1),'valid','linear')(DepthConv1_1)
#----------------------------------------------------------------------      
    conv01_2 = conv(8,(1,1),(1,1),'valid','PReLU')(conv1_1)
    DepthConv1_2 = DepthConv(1,(2,2),(1,1),'valid','PReLU')(conv01_2)
    conv1_2 = conv(8,(1,1),(1,1),'valid','linear')(DepthConv1_2)
#----------------------------------------------------------------------    
    concat1 = tf.keras.layers.Concatenate()([conv1_0, conv1_2])
    conv02_1 = conv(16,(1,1),(1,1),'valid','PReLU')(concat1)
    DepthConv2 = DepthConv(1,(3,3),(1,1),'valid','PReLU')(conv02_1)
    conv2_0 = conv(16,(1,1),(1,1),'valid','linear')(DepthConv2)
#----------------------------------------------------------------------   
    DepthConv2_1 = DepthConv(1,(2,2),(1,1),'valid','PReLU')(conv02_1)
    conv2_1 = conv(16,(1,1),(1,1),'valid','linear')(DepthConv2_1)
#----------------------------------------------------------------------   
    conv02_2 = conv(16,(1,1),(1,1),'valid','PReLU')(conv2_1)  
    DepthConv2_2 = DepthConv(1,(2,2),(1,1),'valid','PReLU')(conv02_2)
    conv2_2 = conv(16,(1,1),(1,1),'valid','linear')(DepthConv2_2)
#----------------------------------------------------------------------   
    concat2 = tf.keras.layers.Concatenate()([conv2_0, conv2_2])   
    conv3 = conv(32,(1,1),(1,1),'valid','PReLU')(concat2)  
    DepthConv2 = DepthConv(1,(3,3),(1,1),'valid','PReLU')(conv3)
    conv4 = conv(32,(1,1),(1,1),'valid','linear')(DepthConv2)
#----------------------------------------------------------------------   
    cls_result = conv(2,(1,1),(1,1),'valid','Softmax')(conv4)
    bbox_result = bbox(4,(1,1),'valid')(conv4)
    print('cls_result',cls_result.shape)
    print('bbox_result',bbox_result.shape)
    return tf.keras.Model(inputs = inputs, outputs = (cls_result , bbox_result));
    
def net16_DepthConv_one(height, width): 
    inputs = tf.keras.Input((height, width, 3,));
#----------------------------------------------------------------------   
    conv00 = conv(8,(1,1),(1,1),'valid','PReLU')(inputs)
    DepthConv0 = DepthConv(1,(3,3),(1,1),'valid','PReLU')(conv00)
    conv0 = conv(8,(1,1),(1,1),'valid','linear')(DepthConv0)
#----------------------------------------------------------------------       
    pool1 = MaxPooling2D(2, 2,'same')(conv0)
    #conv1 = conv(8,(1,1),(1,1),'valid','linear')(pool1)
#----------------------------------------------------------------------   
    conv01 = conv(16,(1,1),(1,1),'valid','PReLU')(pool1)
    DepthConv1 = DepthConv(1,(3,3),(1,1),'valid','PReLU')(conv01)
    conv1_0 = conv(16,(1,1),(1,1),'valid','linear')(DepthConv1)    
#----------------------------------------------------------------------    
    conv02_1 = conv(24,(1,1),(1,1),'valid','PReLU')(conv1_0)
    DepthConv2 = DepthConv(1,(3,3),(1,1),'valid','PReLU')(conv02_1)
    conv2_0 = conv(24,(1,1),(1,1),'valid','linear')(DepthConv2)
#----------------------------------------------------------------------      
    conv3 = conv(32,(1,1),(1,1),'valid','PReLU')(conv2_0)  
    DepthConv2 = DepthConv(1,(3,3),(1,1),'valid','PReLU')(conv3)
    conv4 = conv(32,(1,1),(1,1),'valid','linear')(DepthConv2)
#----------------------------------------------------------------------   
    cls_result = conv(2,(1,1),(1,1),'valid','Softmax')(conv4)
    bbox_result = bbox(4,(1,1),'valid')(conv4)
    print('cls_result',cls_result.shape)
    print('bbox_result',bbox_result.shape)
    return tf.keras.Model(inputs = inputs, outputs = (cls_result , bbox_result));  
    
def net16(height, width): 
    inputs = tf.keras.Input((height, width, 3,));
    conv0 = conv(4,(3,3),(1,1),'valid','PReLU')(inputs)
    pool1 = MaxPooling2D(2, 2,'same')(conv0)

    #conv1 = conv(8,(3,3),(1,1),'valid','PReLU')(pool1)
    conv2 = conv(8,(3,3),(1,1),'valid','PReLU')(pool1)
    conv2_1 = conv(8,(2,2),(1,1),'valid','PReLU')(pool1)
    conv2_2 = conv(8,(2,2),(1,1),'valid','PReLU')(conv2_1)
    concat1 = tf.keras.layers.Concatenate()([conv2, conv2_2])

    conv3 = conv(16,(3,3),(1,1),'valid','PReLU')(concat1)
    conv3_1 = conv(16,(2,2),(1,1),'valid','PReLU')(concat1)
    conv3_2 = conv(16,(2,2),(1,1),'valid','PReLU')(conv3_1)
    #print(conv3.shape)
    #print(conv3_2.shape)
    concat2 = tf.keras.layers.Concatenate()([conv3, conv3_2])
    #print(concat2.shape)
    conv4 = conv(32,(3,3),(1,1),'valid','PReLU')(concat2)
    #print(conv4.shape)
    cls_result = conv(2,(1,1),(1,1),'valid','Softmax')(conv4)
    #conv4_1 = conv(12,(2,2),(1,1),'valid','PReLU')(concat2)
    
    bbox_result = bbox(4,(1,1),'valid')(conv4)
    print(cls_result.shape)
    print(bbox_result.shape)
    return tf.keras.Model(inputs = inputs, outputs = (cls_result , bbox_result));      

def Rnet24(height, width): 
    inputs = tf.keras.Input((height, width, 3,));
    conv0 = conv(28,(3,3),(1,1),'valid','PReLU')(inputs)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3),strides=(2,2),padding = 'same')(conv0)
    print('pool1 ',pool1.shape)
    conv1 = conv(48,(3,3),(1,1),'valid','PReLU')(pool1)
    print('conv1 ',conv1.shape)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3),strides=(2,2),padding = 'valid')(conv1)
    print('pool2 ',pool2.shape)
    conv2 = conv(64,(2,2),(1,1),'valid','PReLU')(pool2)
    print('conv2 ',conv2.shape)
    conv3 = conv(128,(3,3),(1,1),'valid','PReLU')(conv2)
    print('conv3 ',conv3.shape)
    cls_res = conv(2,(1,1),(1,1),'valid','Softmax')(conv3)
    bbox_res = bbox(4,(1,1),'valid')(conv3)
    print('cls ',cls_res.shape)
    print('bbox ',bbox_res.shape)

    return tf.keras.Model(inputs = inputs, outputs = (cls_res, bbox_res));
    
def Rnet24_DepthConv(height, width): 
    inputs = tf.keras.Input((height, width, 3,));
#---------------------------------------------------------------------- 
    conv0 = conv(28,(1,1),(1,1),'valid','PReLU')(inputs)
    DepthConv0 = DepthConv(1,(3,3),(1,1),'valid','PReLU')(conv0)
    conv00 = conv(28,(1,1),(1,1),'valid','linear')(DepthConv0)
#----------------------------------------------------------------------    
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3),strides=(2,2),padding = 'same')(conv00)
    conv1 = conv(48,(1,1),(1,1),'valid','PReLU')(pool1)
    DepthConv1 = DepthConv(1,(3,3),(1,1),'valid','PReLU')(conv1)
    conv01 = conv(48,(1,1),(1,1),'valid','linear')(DepthConv1)
#---------------------------------------------------------------------- 
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3),strides=(2,2),padding = 'valid')(conv01)
    conv2 = conv(64,(1,1),(1,1),'valid','PReLU')(pool2)
    DepthConv2 = DepthConv(1,(2,2),(1,1),'valid','PReLU')(conv2)
    conv02 = conv(64,(1,1),(1,1),'valid','linear')(DepthConv2)
#----------------------------------------------------------------------     
    conv3 = conv(128,(1,1),(1,1),'valid','PReLU')(conv02)
    DepthConv3 = DepthConv(1,(3,3),(1,1),'valid','PReLU')(conv3)
    conv03 = conv(128,(1,1),(1,1),'valid','linear')(DepthConv3)
#---------------------------------------------------------------------- 
    print('conv03 ',conv03.shape)
    cls_res = conv(2,(1,1),(1,1),'valid','Softmax')(conv03)
    bbox_res = bbox(4,(1,1),'valid')(conv03)
    print('cls ',cls_res.shape)
    print('bbox ',bbox_res.shape)

    return tf.keras.Model(inputs = inputs, outputs = (cls_res, bbox_res));

def Onet36(height, width): 
    inputs = tf.keras.Input((height, width, 3,));
    conv0 = conv(32,(3,3),(1,1),'valid','PReLU')(inputs)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3),strides=(2,2),padding = 'valid')(conv0)
    print('pool1 ',pool1.shape)
    conv1 = conv(32,(3,3),(1,1),'valid','PReLU')(pool1)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3),strides=(2,2),padding = 'same')(conv1)
    print('pool2 ',pool2.shape)
    conv2 = conv(64,(3,3),(1,1),'valid','PReLU')(pool2)
    print('conv2 ',conv2.shape)
    conv3 = conv(64,(3,3),(1,1),'valid','PReLU')(conv2)
    print('conv3 ',conv3.shape)
    conv4 = conv(128,(3,3),(1,1),'valid','PReLU')(conv3)
    
    cls_res = conv(2,(1,1),(1,1),'valid','Softmax')(conv4)
    bbox_res = bbox(4,(1,1),'valid')(conv4)
    print('cls ',cls_res.shape)
    print('bbox ',bbox_res.shape)
    return tf.keras.Model(inputs = inputs, outputs = (cls_res , bbox_res));

def Onet36_DepthConv(height, width): 
    inputs = tf.keras.Input((height, width, 3,));
    conv0 = conv(32,(1,1),(1,1),'valid','PReLU')(inputs)
    DepthConv0 = DepthConv(1,(3,3),(1,1),'valid','PReLU')(conv0)
    conv00 = conv(32,(1,1),(1,1),'valid','linear')(DepthConv0)
#----------------------------------------------------------------------  
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3),strides=(2,2),padding = 'valid')(conv00)
    print('pool1 ',pool1.shape)
    conv1 = conv(32,(1,1),(1,1),'valid','PReLU')(pool1)
    DepthConv1 = DepthConv(1,(3,3),(1,1),'valid','PReLU')(conv1)
    conv01 = conv(32,(1,1),(1,1),'valid','linear')(DepthConv1)
#----------------------------------------------------------------------  
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3),strides=(2,2),padding = 'same')(conv01)
    print('pool2 ',pool2.shape)
    conv3 = conv(64,(1,1),(1,1),'valid','PReLU')(pool2)
    DepthConv3 = DepthConv(1,(3,3),(1,1),'valid','PReLU')(conv3)
    conv03 = conv(64,(1,1),(1,1),'valid','linear')(DepthConv3)
    print('conv03 ',conv03.shape)
#----------------------------------------------------------------------  
    conv4 = conv(64,(1,1),(1,1),'valid','PReLU')(conv03)
    DepthConv4 = DepthConv(1,(3,3),(1,1),'valid','PReLU')(conv4)
    conv04 = conv(64,(1,1),(1,1),'valid','linear')(DepthConv4)
#----------------------------------------------------------------------  
    conv5 = conv(128,(1,1),(1,1),'valid','PReLU')(conv04)
    DepthConv5 = DepthConv(1,(3,3),(1,1),'valid','PReLU')(conv5)
    conv05 = conv(128,(1,1),(1,1),'valid','linear')(DepthConv5)
#----------------------------------------------------------------------      
    cls_res = conv(2,(1,1),(1,1),'valid','Softmax')(conv05)
    bbox_res = bbox(4,(1,1),'valid')(conv05)
    print('cls ',cls_res.shape)
    print('bbox ',bbox_res.shape)
    return tf.keras.Model(inputs = inputs, outputs = (cls_res , bbox_res));

def MCTNN_Onet(height, width): 
    inputs = tf.keras.Input((height, width, 3,));
    conv0 = conv(32,(3,3),(1,1),'valid','PReLU')(inputs)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3),strides=(2,2),padding = 'valid')(conv0)
    print('pool1 ',pool1.shape)
    conv1 = conv(64,(3,3),(1,1),'valid','PReLU')(pool1)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3),strides=(2,2),padding = 'valid')(conv1)
    print('pool2 ',pool2.shape)
    conv2 = conv(64,(3,3),(1,1),'valid','PReLU')(pool2)
    print('conv2 ',conv2.shape)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3),strides=(2,2),padding = 'same')(conv2)
    print('pool3 ',pool3.shape)
    conv3 = conv(128,(2,2),(1,1),'valid','PReLU')(pool3)
    print('conv3 ',conv3.shape)

    
    cls_res = conv(2,(3,3),(1,1),'valid','Softmax')(conv3)
    bbox_res = bbox(4,(3,3),'valid')(conv3)
    print('cls ',cls_res.shape)
    print('bbox ',bbox_res.shape)
    return tf.keras.Model(inputs = inputs, outputs = (cls_res , bbox_res));
    
def bbox(filters_,kernel_size_,padding_):
    return tf.keras.layers.Conv2D(filters = filters_,kernel_size = kernel_size_, padding = padding_)

def net16n(height, width):
    inputs = tf.keras.Input((height, width, 3,));
    conv1 = conv(6,(3,3),(1,1),'valid','PReLU')(inputs)
    conv2 = conv(6,(3,3),(1,1),'valid','PReLU')(conv1)
    conv2_1 = conv(6,(2,2),(1,1),'valid','PReLU')(conv1)
    conv2_2 = conv(6,(2,2),(1,1),'valid','PReLU')(conv2_1)
    concat1 = tf.keras.layers.Concatenate()([conv2, conv2_2])
   
    conv3 = conv(8,(3,3),(1,1),'valid','PReLU')(concat1)
    conv3_1 = conv(8,(2,2),(1,1),'valid','PReLU')(concat1)
    conv3_2 = conv(8,(2,2),(1,1),'valid','PReLU')(conv3_1)
    concat2 = tf.keras.layers.Concatenate()([conv3, conv3_2])

    pool1 = MaxPooling2D(2, 2,'valid')(concat2)
    conv4 = conv(12,(3,3),(1,1),'valid','PReLU')(pool1)
    conv5 = conv(12,(3,3),(1,1),'valid','PReLU')(conv4)
    cls_result = conv(2,(1,1),(1,1),'valid','Softmax')(conv5)
   
    conv4_1 = conv(12,(3,3),(1,1),'valid','PReLU')(pool1)
    conv5_1 = conv(12,(3,3),(1,1),'valid','PReLU')(conv4_1)
    bbox_result = bbox(4,(1,1),'valid')(conv5_1)
    return tf.keras.Model(inputs = inputs, outputs = (cls_result , bbox_result))


    inputs = tf.keras.Input((height, width, 3,));
#----------------------------------------------------------------------   
    conv00 = conv(8,(1,1),(1,1),'valid','PReLU')(inputs)
    DepthConv0 = DepthConv(1,(3,3),(1,1),'valid','PReLU')(conv00)
    conv0 = conv(8,(1,1),(1,1),'valid','linear')(DepthConv0)
#----------------------------------------------------------------------       
    pool1 = MaxPooling2D(2, 2,'same')(conv0)
    #conv1 = conv(8,(1,1),(1,1),'valid','linear')(pool1)
#----------------------------------------------------------------------   
    conv01 = conv(8,(1,1),(1,1),'valid','PReLU')(pool1)
    DepthConv1 = DepthConv(1,(3,3),(1,1),'valid','PReLU')(conv01)
    conv1_0 = conv(8,(1,1),(1,1),'valid','linear')(DepthConv1)
#----------------------------------------------------------------------      
    DepthConv1_1 = DepthConv(1,(2,2),(1,1),'valid','PReLU')(conv01)
    conv1_1 = conv(8,(1,1),(1,1),'valid','linear')(DepthConv1_1)
#----------------------------------------------------------------------      
    conv01_2 = conv(8,(1,1),(1,1),'valid','PReLU')(conv1_1)
    DepthConv1_2 = DepthConv(1,(2,2),(1,1),'valid','PReLU')(conv01_2)
    conv1_2 = conv(8,(1,1),(1,1),'valid','linear')(DepthConv1_2)
#----------------------------------------------------------------------    
    concat1 = tf.keras.layers.Concatenate()([conv1_0, conv1_2])
    conv02_1 = conv(16,(1,1),(1,1),'valid','PReLU')(concat1)
    DepthConv2 = DepthConv(1,(3,3),(1,1),'valid','PReLU')(conv02_1)
    conv2_0 = conv(16,(1,1),(1,1),'valid','linear')(DepthConv2)
#----------------------------------------------------------------------   
    DepthConv2_1 = DepthConv(1,(2,2),(1,1),'valid','PReLU')(conv02_1)
    conv2_1 = conv(16,(1,1),(1,1),'valid','linear')(DepthConv2_1)
#----------------------------------------------------------------------   
    conv02_2 = conv(16,(1,1),(1,1),'valid','PReLU')(conv2_1)  
    DepthConv2_2 = DepthConv(1,(2,2),(1,1),'valid','PReLU')(conv02_2)
    conv2_2 = conv(16,(1,1),(1,1),'valid','linear')(DepthConv2_2)
#----------------------------------------------------------------------   
    concat2 = tf.keras.layers.Concatenate()([conv2_0, conv2_2])   
    pool2 = AveragePooling2D (3, 3,'valid')(concat2)
#----------------------------------------------------------------------   
    cls_result = conv(2,(1,1),(1,1),'valid','Softmax')(pool2)
    bbox_result = bbox(4,(1,1),'valid')(pool2)
    print('cls_result',cls_result.shape)
    print('bbox_result',bbox_result.shape)
    return tf.keras.Model(inputs = inputs, outputs = (cls_result , bbox_result));
    
def net16s(height, width): 
    inputs = tf.keras.Input((height, width, 3,));
    conv1 = conv(4,3,1,'valid','PReLU')(inputs)
    conv2 = conv(4,3,1,'valid','PReLU')(conv1)
    conv2_1 = conv(4,2,1,'valid','PReLU')(conv1)
    conv2_2 = conv(4,2,1,'valid','PReLU')(conv2_1)
    concat1 = tf.keras.layers.Concatenate()([conv2, conv2_2])

    conv3 = conv(8,3,1,'valid','PReLU')(concat1)
    conv3_1 = conv(8,2,1,'valid','PReLU')(concat1)
    conv3_2 = conv(8,2,1,'valid','PReLU')(conv3_1)
    concat2 = tf.keras.layers.Concatenate()([conv3, conv3_2])

    pool1 = MaxPooling2D(2, 2,'valid')(concat2)
    conv4 = conv(8,3,1,'valid','PReLU')(pool1)
    conv5 = conv(8,3,1,'valid','PReLU')(conv4)
    cls_branch = conv(2,1,1,'valid','Softmax')(conv5)

    conv4_1 = conv(8,3,1,'valid','PReLU')(pool1)
    conv5_1 = conv(8,3,1,'valid','PReLU')(conv4_1)
    bbox_branch = Conv2D(4,1,1,'valid')(conv5_1)
    return tf.keras.Model(inputs = inputs, outputs = (cls_branch, bbox_branch))
    


  

def net16_2p(height, width):
  inputs = tf.keras.Input((height, width, 3,));
  x = conv(8, (3,3), (1,1), 'valid', 'PReLU')(inputs)
  x = conv(8, (3,3), (1,1), 'valid', 'PReLU')(x)
  x = MaxPooling2D(2, 2,'valid')(x)
  x = conv(16, (3,3), (1,1), 'valid', 'PReLU')(x)
  x = MaxPooling2D(2, 2,'valid')(x)
  x = conv(16, (2,2), (1,1), 'valid', 'PReLU')(x)
  cls_result = conv(2, (1,1), (1,1),'valid', 'Softmax')(x)
  bbox_result = bbox(4, (1,1), 'valid')(x)
  return tf.keras.Model(inputs = inputs, outputs = (cls_result , bbox_result))
  
  
def net16_2p_Depthwise(height, width):
  inputs = tf.keras.Input((height, width, 3,));
  x = DSblock(8, 3, 1, 'valid', 'PReLU',inputs)
  x = DSblock(8, 3, 1, 'valid', 'PReLU',x)
  x = MaxPooling2D(2, 2,'valid')(x)
  x = DSblock(16, 3, 1, 'valid', 'PReLU',x)
  x = MaxPooling2D(2, 2,'valid')(x)
  x = DSblock(16, 2, 1, 'valid', 'PReLU',x)
  cls_result = Conv2D(2, 1, 1,'valid', activation='Softmax')(x)
  bbox_result = Conv2D(4, 1, 1, 'valid')(x)
  return tf.keras.Model(inputs = inputs, outputs = (cls_result , bbox_result))
  
  

def conv(filters_,kernel_size_,strides_,padding_,activation_):
    if activation_ == 'PReLU':  
        activation_ = PReLU(shared_axes=[1, 2]) 
    elif activation_ == 'Softmax':  
        activation_ = tf.keras.layers.Softmax()    
    elif activation_ == 'linear':  
        activation_ = None
    return tf.keras.layers.Conv2D(filters = filters_,kernel_size = kernel_size_,strides = strides_ ,padding = padding_,activation = activation_)

def DepthConv(depth_multiplier_,kernel_size_,strides_,padding_,activation_):
    if activation_ == 'PReLU':  
        #activation_ = ReLU(max_value=6) 
        activation_ = PReLU(shared_axes=[1, 2])
    elif activation_ == 'Softmax':  
        activation_ = tf.keras.layers.Softmax()         
    return DepthwiseConv2D(depth_multiplier = depth_multiplier_, kernel_size = kernel_size_,strides = strides_ ,padding = padding_,activation = activation_)   
#test_sizes = [(224, 224), (384, 384), (480, 640)]

def DSblock(channel , kernel_size_ , strides_ , padding_ ,activation_ , inputs):
    conv0 = conv(channel,(1,1),(1,1),'valid','PReLU')(inputs)
    DepthConv0 = DepthConv(1,(kernel_size_,kernel_size_),(strides_,strides_),padding_,activation_)(conv0)
    conv1 = conv(channel,(1,1),(1,1),'valid','linear')(DepthConv0)
    return conv1

if 1:
  model16 = net16_DepthConv(640,480)
  test_model = model16
  test_model.summary()
  flops = get_flops(test_model, batch_size=1)
  print(f"Model FLOPS: {flops / 10 ** 9:.03} G")

if 0:
    flops = 0
    factor = 0.709
    factor_count = 0
    minl=np.amin([640, 480])
    m=16.0/16 #默認 minsize = 20
    minl=minl*m     #使 minl = 12
    # create scale pyramid
    scales=[]    
    temp = 0
    while minl>=20:
        scales += [m*np.power(factor, factor_count)]    #原版 MTCNN 默認 factor = 0.709
        minl = minl*factor
        factor_count += 1    
    # first stage    
    for scale in scales:
        
        hs=int(np.ceil(640*scale))    #np.ceil //向上取整數：0.7>>1 , -0.1 >> 0
        ws=int(np.ceil(480*scale))
        print('hs,ws',hs,ws)
        model12 = net12(hs, ws)
        #model16s = net16s(512, 512)
        test_model = model12
        #test_model.summary()
        flops = flops + get_flops(test_model, batch_size=1)
        
    test_model.summary()
    print(f"Model FLOPS: {flops / 10 ** 9:.03} G")

