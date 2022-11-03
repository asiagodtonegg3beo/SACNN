import sys, cv2
import tensorflow as tf
import numpy as np

def view_bar_gen_add_anchor_face(msg, num, total, pos_idx, part_idx, neg_idx,anchor_face_idx):
  sys.stdout.write('\r{:10s} [{:>7d} / {:<7d}] pos:{}, part:{}, neg:{}, anchor_face:{}'.format(msg, num, total, pos_idx, part_idx, neg_idx, anchor_face_idx))
  sys.stdout.flush()

def view_bar_gen(msg, num, total, pos_idx, part_idx, neg_idx):
  sys.stdout.write('\r{:10s} [{:>7d} / {:<7d}] pos:{}, part:{} neg:{}'.format(msg, num, total, pos_idx, part_idx, neg_idx))
  sys.stdout.flush()

def view_bar(msg, num, total):
  sys.stdout.write('\r{:10s} [{:>7d} / {:<7d}]'.format(msg, num, total))
  sys.stdout.flush()

def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def floats_feature(value):    
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def floats_feature_iou(value):
    if not isinstance(value, list):
        value = [value]    
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def IoU(box, boxes):
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)
    inter = w * h
    ovr = inter / (box_area + area - inter)
    return ovr
    
def IoU2(box, boxes):
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area = (boxes[2] - boxes[0] + 1) * (boxes[3] - boxes[1] + 1)
    xx1 = np.maximum(box[0], boxes[0])
    yy1 = np.maximum(box[1], boxes[1])
    xx2 = np.minimum(box[2], boxes[2])
    yy2 = np.minimum(box[3], boxes[3])
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)
    inter = w * h
    ovr = inter / (box_area + area - inter)
    return ovr

def IoM(box, boxes):
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area = (boxes[2] - boxes[0] + 1) * (boxes[3] - boxes[1] + 1)
    xx1 = np.maximum(box[0], boxes[0])
    yy1 = np.maximum(box[1], boxes[1])
    xx2 = np.minimum(box[2], boxes[2])
    yy2 = np.minimum(box[3], boxes[3])
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)
    inter = w * h
    ovr = inter / np.minimum(box_area,area)
    return ovr
    
def read_image(img_path):
  img = cv2.imread(img_path)
  if img is None:
     print('Error! Cannot open', img_path)
     sys.exit(1)
  return img
