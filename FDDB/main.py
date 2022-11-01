import os
import sys
'''

ECHO 生成資料到 /data/FDDB_output
python FDDB_test.py
ECHO merge_FDDB_files.py
python merge_FDDB_files.py
ECHO ROC_result.py
python ROC_result.py
ECHO ROC_draw_epochs.py 產生曲線圖
python ROC_draw_epochs.py
pause
'''

def runfile(current_file):
    print("執行",current_file)

    try:
        os.system('python '+ current_file)
    except Exception:
        print(current_file,"執行失敗!")
        sys.exit()


try:
    runfile("FDDB_test.py")
    runfile("merge_FDDB_files.py")
    runfile("ROC_result.py")
    #runfile("ROC_draw_epochs.py")
    print("執行成功")
   
except Exception:
    print('執行失敗')
    sys.exit()
