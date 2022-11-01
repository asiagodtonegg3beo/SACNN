from os import system as sys

path_evaluate = 'data\\FDDB_Dev_cpp\\FDDB.exe'
a = 'data\\FDDB-folds\\FDDB-fold-all-ellipseList.txt'
d = 'data\\FDDB_output\\FDDB-det-fold-all.txt' ###
f = '0'
i = 'data/originalPics/'
l = 'data\\FDDB-folds\\FDDB-fold-all.txt'
r = 'data/FDDB_output/' # 選擇輸出的地方
z = '.jpg'
doPath = path_evaluate+' -a '+a+' -d '+d+' -f '+f+' -i '+i+' -l '+l+' -r '+r+' -z '+z
print(doPath)
sys(doPath)

print ("finish!")

'''
evaluate參數說明
-a : 答案的TXT檔(FDDB-fold-ellipseList-all.txt)
-d : 你的演算法產生的偵測TXT檔(HOG-result.txt)
-f : 0 使用矩形，1 使用橢圓，2 使用點陣列
-i : 放置照片的目錄 (originalPics)
-l : 記錄所有圖片的TXT檔(FDDB-fold-all.txt)
-r : 將要存放ROC輸出的目錄
-z : 使用".jpg"
'''
