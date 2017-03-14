# 載入需要的...
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# 準備數據 ... 假設我要畫一個sin波 從0~180度

x = np.arange(0,180)
y = np.sin(x * np.pi / 180.0)


# 開始畫圖

    # 設定要畫的的x,y數據list....

plt.plot(x,y)

    # 設定圖的範圍, 不設的話，系統會自行決定
plt.xlim(-30,390)
plt.ylim(-1.5,1.5)
    # 照需要寫入x 軸和y軸的 label 以及title

plt.title("The Title")


    # 在這個指令之前，都還在做畫圖的動作
    # 這個指令算是 "秀圖"
plt.show()


plt.savefig("filename.png",dpi=300,format="png")

    # 如果要存成圖形檔:
    # 把 pyplot.show() 換成下面這行:
