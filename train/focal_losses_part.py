"""
Define our custom loss function.
"""
#from keras import backend as K
import tensorflow as tf
import tensorflow.keras.backend as K 


#import dill


def binary_focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.

      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)

      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.

    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)

    """
    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """        
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               -K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return binary_focal_loss_fixed


def categorical_focal_loss(gamma=2., alpha=.25,keep_ratio=0.7):
    """
    Softmax version of focal loss.

           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1

      where m = number of classes, c = class and o = observation

    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy
      gamma -- focusing parameter for modulating factor (1-p)

    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper

    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy

    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
  
    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """
#        print("cls loss")
#        print()
#        print("y_true", y_true)
#        print()
#        print("y_pred", y_pred)
        #print()
        
        # y_pred 訓練階段預處理   
        
        #tf.print('y_true',y_true)
        y_pred = tf.squeeze(y_pred,[1,2])
 

        
        # y_true 轉 one-hot 預處理
        y_true = tf.cast(y_true, tf.float32)
        y_true = tf.one_hot(tf.cast(y_true, tf.int32), 2)
        y_true = tf.squeeze(y_true,-2)   
        #y_true = tf.multiply(y_true,mask_concat)
        
        #tf.print('y_true one_hot',y_true)
        
        # Scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        
        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()        
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        
        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)
        
        # Calculate Focal Loss
        focal_loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
        focal_loss = K.sum(focal_loss,axis=-1,keepdims=True)

        # OHEM     
        y_true_num=tf.reduce_sum(y_true)    
        keep_num = tf.cast(y_true_num*keep_ratio, dtype=tf.int32)
        
        loss_sort = tf.sort(focal_loss, axis =0)
        #tf.print('\nloss_sort fix',loss_sort)
        loss_sort = tf.squeeze(loss_sort,1)    
        #tf.print('loss_sort squeeze',loss_sort)
        OHEM_loss,_ = tf.nn.top_k(loss_sort, keep_num)
        loss = K.mean(OHEM_loss)  
        
        return loss

    return categorical_focal_loss_fixed

def iou_loss(gamma=2., alpha=.25,keep_ratio=0.7):
    """
    Softmax version of focal loss.

           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1

      where m = number of classes, c = class and o = observation

    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy
      gamma -- focusing parameter for modulating factor (1-p)

    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper

    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy

    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
  
    def categorical__loss_iou(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """
        print("y_true", y_true)
        print()
        print("y_pred", y_pred)
        print()
        
        # y_pred 訓練階段預處理   
        y_pred = tf.squeeze(y_pred,[1,2]) 
        
    
        #y_true 轉 one-hot 預處理
    #        y_true = tf.cast(y_true, tf.float32)
    #        y_true = tf.one_hot(tf.cast(y_true, tf.int32), 2)
    #        y_true = tf.squeeze(y_true,-2)    
        
        print("y_pred", y_pred)
        print()
        
        #print( y_true - y_pred )
        #print()
        
        # Scale predictions so that the class probas of each sample sum to 1
        #y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        
        #print("y_pred", y_pred)
        #print()
        
        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()      
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        
        # Calculate Cross Entropy
        #cross_entropy = -y_true * K.log(y_pred)
        cross_entropy_1 = -y_true * K.log(y_pred)
        cross_entropy_2 = -(1-y_true) * K.log((1-y_pred))
        cross_entropy_ios = cross_entropy_1 + cross_entropy_2
        print("cross_entropy_ios",cross_entropy_ios)
        print()
        # Calculate Focal Loss
        #focal_loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy_ios
        #focal_loss = K.sum(focal_loss,axis=-1,keepdims=True)
    
        # OHEM     
    #    y_true_num=tf.reduce_sum(y_true)   
    #    print("y_true_num",y_true_num)
    #    print()
    #    keep_num = tf.cast(y_true_num*keep_ratio, dtype=tf.int32)
    #    print("keep_num",keep_num)
    #    
    #    loss_sort = tf.sort(focal_loss, axis =0)
    #    loss_sort = tf.squeeze(loss_sort,1)    
    #    
    #    OHEM_loss,_ = tf.nn.top_k(loss_sort, keep_num)
        
        #cross_entropy_ios = tf.reduce_sum(cross_entropy_ios)
        
        loss = K.mean(cross_entropy_ios)  
        
        return loss

    return categorical__loss_iou

if __name__ == '__main__':

    # Test serialization of nested functions
    bin_inner = dill.loads(dill.dumps(binary_focal_loss(gamma=2., alpha=.25)))
    print(bin_inner)

    cat_inner = dill.loads(dill.dumps(categorical_focal_loss(gamma=2., alpha=.25)))
    print(cat_inner)