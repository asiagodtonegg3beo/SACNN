from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

path_t1 = "data\\FDDB_output\\DiscROC_0824.txt"
path_t2 = "data\\FDDB_output\\DiscROC.txt"

#path_t2 = "data\\FDDB_output\\Pnet_paper_266_fl.txt"
path_t3 = "data\\FDDB_output\\DSFD_DiscROC.txt"
path_t4 = "data\\FDDB_output\\mxnet-face-fr50-DiscROC.txt"
path_imgSave = "data\\FDDB_output\\result.png"

# PNet
#fp_num1 = 120000
#fp_num2 = 240000 


# RNet
#fp_num1 = 12000
#fp_num2 = 20000

# ONet
#fp_num1 = 300
#fp_num2 = 600

fp_num1 = 50000 # 設定中段值
fp_num2 = 60000

#fp_num1 = 30000 # 設定中段值
#fp_num2 = 31000
    

# t1
with open(path_t1, 'r') as fp:
    t1 = fp.readlines()
Plot = [line.split() for line in t1]
t1_x = [float(x[1]) for x in Plot]
t1_y = [float(y[0]) for y in Plot]

#print(t1_y[0])
for i in range(len(Plot)):    
    fp1 = int(t1_x[i])
    if (fp1 <= fp_num2) & (fp1 >= (fp_num2 - 20)) == True:
        dis1_b = round(t1_y[i-1] * 100, 2)
        print(dis1_b)
    elif (fp1 <= fp_num1) & (fp1 >= (fp_num1 - 20)) == True:         
        dis1_a = round(t1_y[i-1] * 100, 2)
        break

# t2
with open(path_t2, 'r') as fp:
    t2 = fp.readlines()        
t2 = [line.split() for line in t2]
t2_x = [float(x[1]) for x in t2]
t2_y = [float(y[0]) for y in t2] 


for i in range(len(t2)):    
    fp2 = int(t2_x[i])
    t2_y[i-1] = t2_y[i-1] 
    if (fp2 <= fp_num2) & (fp2 >= (fp_num2 - 20)) == True:
        dis2_b = round(t2_y[i-1] * 100, 2)
    elif (fp2 <= fp_num1) & (fp2 >= (fp_num1 - 20)) == True:         
        dis2_a = round(t2_y[i-1] * 100, 2)
        break

'''
# t3
with open(path_t3, 'r') as fp:
    t3 = fp.readlines()
t3 = [line.split() for line in t3]
t3_x = [float(x[1]) for x in t3]
t3_y = [float(y[0]) for y in t3]

for i in range(len(t3)):    
    fp3 = int(t3_x[i])
    if (fp3 <= fp_num2) & (fp3 >= (fp_num2 - 20)) == True:
        dis3_b = round(t3_y[i-1] * 100, 2)
    elif (fp3 <= fp_num1) & (fp3 >= (fp_num1 - 20)) == True:         
        dis3_a = round(t3_y[i-1] * 100, 2)
        break


# t4
with open(path_t4, 'r') as fp:
    t4 = fp.readlines()        
t4 = [line.split() for line in t4]
t4_x = [float(x[1]) for x in t4]
t4_y = [float(y[0]) for y in t4]

for i in range(len(t4)):    
    fp4 = int(t4_x[i])
    if (fp4 <= fp_num2) & (fp4 >= (fp_num2 - 20)) == True:
        dis4_b = round(t4_y[i-1] * 100, 2)
    elif (fp4 <= fp_num1) & (fp4 >= (fp_num1 - 20)) == True:         
        dis4_a = round(t4_y[i-1] * 100, 2)
'''
'''
# t5
with open(path_t5, 'r') as fp:
    t5 = fp.readlines()        
t5 = [line.split() for line in t5]
t5_x = [float(x[1]) for x in t5]
t5_y = [float(y[0]) for y in t5]

for i in range(len(t5)):    
    fp5 = int(t5_x[i])
    if (fp5 <= fp_num2) & (fp5 >= (fp_num2 - 20)) == True:
        dis5_b = round(t5_y[i-1] * 100, 2)
    elif (fp5 <= fp_num1) & (fp5 >= (fp_num1 - 20)) == True:         
        dis5_a = round(t5_y[i-1] * 100, 2)
        break

# t6
with open(path_t6, 'r') as fp:
    t6 = fp.readlines()        
t6 = [line.split() for line in t6]
t6_x = [float(x[1]) for x in t6]
t6_y = [float(y[0]) for y in t6]

for i in range(len(t6)):    
    fp6 = int(t6_x[i])
    if (fp6 <= fp_num2) & (fp6 >= (fp_num2 - 20)) == True:
        dis6_b = round(t6_y[i-1] * 100, 2)
    elif (fp6 <= fp_num1) & (fp6 >= (fp_num1 - 20)) == True:         
        dis6_a = round(t6_y[i-1] * 100, 2)
        break

'''


# get data we need to be print
count = len(Plot)

### plot data
plt.figure()
plt.grid() #設置網格

### 設定Y軸刻度
my_y_ticks = np.arange(0, 1.1, 0.1)
plt.yticks(my_y_ticks)

# set y limite
plt.ylim((-0.07,1))
# print label
plt.xlabel('False Positive (FP)')
plt.ylabel('True Positive Rate (TPR)')


# plot data
plt.text(t1_x[0] - t1_x[0] / 3,t1_y[0] + 0.03,'')
plt.plot(t1_x,t1_y,label = "2 pooling Pnet16 :\n"  #改曲線名稱
         + "  FP "+ str(fp_num1) + " : " + str(dis1_a) + "%, "
         + "  FP " + str(fp_num2) + " : " + str(dis1_b) +  "%, "
         + "  total : "+ str(round(t1_y[0] * 100, 2)) + "%"
         ,color = '#FF0000', linewidth = 1.0)

plt.text(t2_x[0] - t2_x[0] / 3,t2_y[0] + 0.03,'')
plt.plot(t2_x,t2_y,label = "Paper Pnet16:\n" 
         + "  FP "+ str(fp_num1) + " : " + str(dis2_a) + "%, "
         + "  FP " + str(fp_num2) + " : " + str(dis2_b) +  "%, "
         + "  total : "+ str(round(t2_y[0] * 100, 2)) + "%"
         ,color = '#2A52BE', linewidth = 1.0)


'''
plt.text(t3_x[0] - t3_x[0] / 3,t3_y[0] + 0.03,'')
plt.plot(t3_x,t3_y,label = "DSFD_DiscROC :\n" 
         + "  FP "+ str(fp_num1) + " : " + str(dis3_a) + "%, "
         + "  FP " + str(fp_num2) + " : " + str(dis3_b) +  "%, "
         + "  toltal : "+ str(round(t3_y[0] * 100, 2)) + "%"
         ,color = '#02C874', linewidth = 1.0)  #02C874 綠色
'''
'''
plt.text(t4_x[0] - t4_x[0] / 3,t4_y[0] + 0.03,'')
plt.plot(t4_x,t4_y,label = "Mxnet :\n" 
         + "  FP "+ str(fp_num1) + " : " + str(dis4_a) + "%, "
         + "  FP " + str(fp_num2) + " : " + str(dis4_b) +  "%, "
         + "  toltal : "+ str(round(t4_y[0] * 100, 2)) + "%"
         ,color = '#921AFF', linewidth = 1.0)
'''
'''
plt.text(t5_x[0] - t5_x[0] / 3,t5_y[0] + 0.03,'')
plt.plot(t5_x,t5_y,label = "AIT_RNet :\n" 
         + "  FP "+ str(fp_num1) + " : " + str(dis5_a) + "%, "
         + "  FP " + str(fp_num2) + " : " + str(dis5_b) +  "%, "
         + "  toltal : "+ str(round(t5_y[0] * 100, 2)) + "%"
      ,color = 'purple', linewidth = 1.0)

plt.text(t6_x[0] - t6_x[0] / 3,t6_y[0] + 0.03,'')
plt.plot(t6_x,t6_y,label = "80 epochs :\n" 
         + "  FP "+ str(fp_num1) + " : " + str(dis6_a) + "%, "
         + "  FP " + str(fp_num2) + " : " + str(dis6_b) +  "%, "
         + "  toltal : "+ str(round(t6_y[0] * 100, 2)) + "%"
         ,color = 'purple', linewidth = 1.0)

'''


plt.legend(loc='lower right')

'''
#FF7300 橘色
#2A52BE 藍色
'''

# print data text
#plt.title('title')








# save img
plt.savefig(path_imgSave)
plt.show()

