import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import PReLU, UpSampling2D, Concatenate,MaxPooling2D,Conv2D,DepthwiseConv2D,ReLU

#Pnet16 未使用DSblock

def net16(): 
    inputs = tf.keras.Input((None, None, 3,));
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

#Pnet16 使用DSblock

def Pnet16_DepthConv(): 
    inputs = tf.keras.Input((16, 16, 3,));
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
    

    
def Rnet24(): 
    inputs = tf.keras.Input((24, 24, 3,));
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

#Rnet24 DSblock
def Rnet24_DepthConv(): 
    inputs = tf.keras.Input((24, 24, 3,));
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
    
#Onet36未使用DSblock   
def Onet36(): 
    inputs = tf.keras.Input((36, 36, 3,));
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
    
#Onet36使用DSblock
def Onet36_DepthConv(): 
    inputs = tf.keras.Input((36, 36, 3,));
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
    
def net16_2p_Depthwise():
  inputs = tf.keras.Input((16, 16, 3,));
  x = DSblock(8, 3, 1, 'valid', 'PReLU',inputs)
  x = DSblock(8, 3, 1, 'valid', 'PReLU',x)
  x = MaxPooling2D(2, 2,'valid')(x)
  x = DSblock(16, 3, 1, 'valid', 'PReLU',x)
  x = MaxPooling2D(2, 2,'valid')(x)
  x = DSblock(16, 2, 1, 'valid', 'PReLU',x)
  cls_result = Conv2D(2, 1, 1,'valid', activation=tf.keras.layers.Softmax())(x)
  bbox_result = Conv2D(4, 1, 1, 'valid')(x)
  return tf.keras.Model(inputs = inputs, outputs = (cls_result , bbox_result))
    
def conv(filters_,kernel_size_,strides_,padding_,activation_):
    if activation_ == 'PReLU':  
        #activation_ = ReLU(max_value=6) 
        activation_ = PReLU(shared_axes=[1, 2]) 
    elif activation_ == 'Softmax':  
        activation_ = tf.keras.layers.Softmax()    
    return tf.keras.layers.Conv2D(filters = filters_,kernel_size = kernel_size_,strides = strides_ ,padding = padding_,activation = activation_)

def DepthConv(depth_multiplier_,kernel_size_,strides_,padding_,activation_):
    if activation_ == 'PReLU':  
        #activation_ = ReLU(max_value=6) 
        activation_ = PReLU(shared_axes=[1, 2])  
    elif activation_ == 'Softmax':  
        activation_ = tf.keras.layers.Softmax() 
    elif activation_ == 'linear':  
        activation_ = None        
    return DepthwiseConv2D(depth_multiplier = depth_multiplier_, kernel_size = kernel_size_,strides = strides_ ,padding = padding_,activation = activation_)

def DSblock(channel , kernel_size_ , strides_ , padding_ ,activation_ , inputs):
    conv0 = conv(channel,(1,1),(1,1),'valid','PReLU')(inputs)
    DepthConv0 = DepthConv(1,(kernel_size_,kernel_size_),(strides_,strides_),padding_,activation_)(conv0)
    conv1 = conv(channel,(1,1),(1,1),'valid','linear')(DepthConv0)
    return conv1
    
def cls(filters_,kernel_size_,padding_,activation_):
    if activation_ == 'PReLU':  
        activation_ = PReLU(shared_axes=[1, 2]) 
    elif activation_ == 'Softmax':  
        activation_ = tf.keras.layers.Softmax() 
    return tf.keras.layers.Conv2D(filters = filters_,kernel_size = kernel_size_,padding = padding_,activation = activation_)
  
def bbox(filters_,kernel_size_,padding_):

    return tf.keras.layers.Conv2D(filters = filters_,kernel_size = kernel_size_, padding = padding_)
