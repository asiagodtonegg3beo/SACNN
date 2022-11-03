import time #計算運行時間用
start = time.time() #計算運行時間用

import os, sys, cv2
import random
import tensorflow as tf
import numpy as np
import numpy.random as npr
from mytools import view_bar, bytes_feature, read_image, floats_feature, int64_feature, floats_feature_iou
#from create_samples import DatasetType, ClassFlag
import argparse

# Annotation Samples:
# face12/positive/00001/1.png 1 0 -0.0036 0.0620 -0.1089 0.1540
# face12/part/00001/1.png -1 0 0.0449 0.1587 -0.2211 0.0545
# face12/negative/00001/1.png 0 -1 -1 -1 -1

def get_network_examples(annotation_path, net_size, data_flag): # word [1], word[2] 交換
  print(annotation_path)
  with open(annotation_path) as fp:
     lines = [line.strip() for line in fp]
  examples = []
  for cur, line in enumerate(lines):
      view_bar('processing', cur+1, len(lines))
      words = line.split()
      img_path = words[0]
      img = read_image(img_path)
      img_height, img_width = img.shape[0:2]
      if img_height != net_size or img_width != net_size:
          img = cv2.resize(img, (net_size, net_size))
      image_raw = img.astype('uint8').tostring()
      #data_flag = int(words[1])      
      if data_flag == 1:        
        cls_label1 = int(words[1]) 
        bbox1 = np.array([float(word) for word in words[2:6]], dtype='float32')
                       
        
      else:     
        cls_label1 = 0
        bbox1 = np.zeros(4, dtype='float32')   
        
        
      feature = {'image/label_cls1': int64_feature(cls_label1),                 
                 'image/roi1': floats_feature(bbox1),                                                 
                 'image/encoded': bytes_feature(image_raw)}
      
      example = tf.train.Example(features=tf.train.Features(feature=feature))
      examples.append(example)
  print()
  return examples

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument("-size")
  args = parser.parse_args()
  args.size = int(args.size)
  if args.size not in [12, 20, 16,36,24, 48]:
     print('Error! Invalid size: {}'.format(args.size))
     sys.exit(1)
  net_size = args.size
  
  examples = []
  #pos_path = 'face{}/pos_{}.txt'.format(net_size, net_size)
  pos_path2 = 'face{}/pos_hard_example{}.txt'.format(net_size, net_size)#
  #part_path = 'face{}/part_{}.txt'.format(net_size, net_size)
  #neg_path = 'face{}/neg_{}.txt'.format(net_size, net_size)  
  neg_path2 = 'face{}/neg_hard_example{}.txt'.format(net_size, net_size) # 
  #examples.extend(get_network_examples(pos_path, net_size, data_flag = 1))
  examples.extend(get_network_examples(pos_path2, net_size, data_flag = 1)) #
  #examples.extend(get_network_examples(part_path, net_size, data_flag = 1))
  #examples.extend(get_network_examples(neg_path, net_size, data_flag = 0))
  examples.extend(get_network_examples(neg_path2, net_size, data_flag = 0))  #
  print('len', len(examples))

  random.shuffle(examples)
    
  tfdata_name = 'tfdata_{}_{}_1bbox.tfrecords'.format(net_size, len(examples))
  if os.path.exists(tfdata_name):
     print('Error! {} already exists.'.format(tfdata_name))
     sys.exit(1)
  
  #tf_writer = tf.python_io.TFRecordWriter(tfdata_name)
  tf_writer = tf.compat.v1.python_io.TFRecordWriter(tfdata_name)
  for example in examples:
      tf_writer.write(example.SerializeToString())
  tf_writer.close()

end = time.time() #計算運行時間用
elapsed = end - start #計算運行時間用
print ("Time taken: ", elapsed, "seconds.") #計算運行時間用