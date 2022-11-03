.py檔案:

gen_12net_data.py 生成資料的程式 (我從github找來改的，論文使用)
create_samples_anchor.py (生成資料的程式 上一屆學長留下的)
create_tfdata_1bbox.py (生成tfrecord的程式)
gen_hard_example.py (生成困難樣本，論文中使用Pnet生成Rnet樣本，使用Rnet生成Onet樣本)

沒提到的就是一些工具 不重要
txt檔案:

wider_val_gt.txt  (widerface驗證集的ground truth)
wider_gt.txt (widerface訓練集的ground truth)

wider_face_train_filtered.txt(widerface訓練集的ground truth 過濾某些太模糊的人臉)
list_bbox_celeba_clean.txt(celeba的ground truth,我的論文沒有使用)