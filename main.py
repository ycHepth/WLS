import numpy as np
import pandas as pd
# import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns


def Original_Problem():
    np.random.seed(1)  # 随机数的标记，这里编号为1，规定这个之后，随机数就只生成一次，之后就会固定，除非seed的参数改变。
    x = np.random.uniform(-5, 5, 35)  # 生成从-5到5的随机数记为x，共35个
    e = 2 * np.random.randn(35)  # randn返回满足标准正态分布的随机值，我们将这个随机值的二倍看做随机产生的误差，存在e中
    y = 2 * x
    plt.plot(x, y, 'ro')  # 画出没有误差的散点图
    y = 2 * x + e
    plt.plot(x, y, 'bo')  # 画出有误差的散点图

    plt.show()


def Basic_fitting():
    np.random.seed(1)
    x = np.random.uniform(-5, 5, 35)
    e = 2 * np.random.randn(35)

    y = 2 * x
    # plt.plot(x,y,'ro')
    plt.plot(x, 2 * x, 'r--')
    y = 2 * x + e

    plt.plot(x, y, 'bo')
    sns.regplot(x, y)
    # plt.show()


def un_uniform_data():
    np.random.seed(1)
    x = np.random.uniform(-5, 5, 35)
    e = 2 * np.random.randn(35)

    y = 2 * x
    # plt.plot(x,y,'ro')
    plt.plot(x, 2 * x, 'r--')
    y = 2 * x + e

    for i in range(25, 35):  # 这个for语句将后面的25到35的样本误差扩大了三倍
        y[i] += 3 * e[i]
    plt.plot(x, y, 'bo')
    plt.show()


def WLS():
    np.random.seed(1)
    x = np.random.uniform(-5, 5, 35)
    e = 2 * np.random.randn(35)
    x.sort()
    y = 2 * x

    plt.plot(x, 2 * x, 'r--')
    y = 2 * x + e

    for i in range(25, 35):
        y[i] += 3 * e[i]
    plt.plot(x, y, 'bo')
    ans = [[0], [0]]  # 用于存放结果
    lenx = len(x)

    # 首先我们需要将x由数组转化为矩阵，并给x加上一列1，新的矩阵记为x2
    x2 = np.array(x).reshape(lenx, 1)  # 先将x转化为单行的矩阵，与之前的区别在于之前x的元素为变量，现在的x2的元素为一维数组，例：
    # 原来：x=[x1,x2,x3]
    # 现在：x2=[[x1],[x2],[x3]]
    y2 = np.array(y).reshape(lenx, 1)  # 对y进行同样操作
    # 接下来要给x前面增加一列1
    # print('???')
    add = np.zeros((lenx, 1))  # 先生成具有lenx个子数组且子数组元素为一个0的二维数组，例：
    # [[0],[0],[0]]
    for i in range(lenx):  # 因为上面要求的是一列1，故而把0改成1
        add[i][0] = 1
    # print(add)
    x3 = np.hstack((add, x2))  # 用np.hstack将add和x2进行拼接，例：
    # [[1,x1],[1,x2],[1,x3]]
    matx = np.mat(x3)  # 再将x3转化为矩阵以方便进行矩阵运算，当然x3这样的array数组也可以进行矩阵运算（借助dot函数），但是比较麻烦，mat形式的直接用乘号即可）
    maty = np.mat(y2)
    ans = [[0], [0]]
    ans = np.mat(ans)  # 用来存放答案的矩阵
    t = 2  # 加权参数，对应上面第二张图公式里的分母2t^2的t
    w = np.mat(np.eye((lenx)))  # 生成一个对角阵（这里是 单位矩阵）作为w（也就是权重矩阵）
    # for i in range(lenx):  #对权重矩阵对角线上的数据进行处理

    avey = y.mean()
    minn = 20
    kk = 0
    for i in range(lenx):
        if abs(y[i] - avey) < minn:
            minn = abs(y[i] - avey)
            kk = i

    test_point = kk  # 这里要选取的是y值距离y平均值最近的样本对应的x值.
    # print(kk)
    for i in range(lenx):
        buf = x3[test_point] - matx[i, :]  # 这里的中间变量buf是一个lenx行两列的矩阵，每行第一个元素存放测试点的x值，第二个元素存放着测试点x值与x各元素的差
        w[i, i] = np.exp(buf * buf.T / -2 * t ** 2)  # 对权重矩阵的运算，即上面第二幅图的公式,这里有一个技巧，就是用矩阵的乘法来进行了平方运算
        # print(buf)
        # print('???')
        # print(buf*buf.T)                #测试数据
    ans = (matx.T * (w * matx)).I * (matx.T * (w * maty))  # 即对上面第三幅图的公式的计算
    # print('<><><><><>')
    # print(ans)
    # print(matx.T)       #测试点

    ans = ans.getA().tolist()  # 算出ans之后，将其转化为列表
    ans_x = [x[0], x[lenx - 1]]
    ans_y = [ans[0][0] + ans[1][0] * ans_x[0], ans[0][0] + ans[1][0] * ans_x[1]]
    plt.plot(ans_x, ans_y, 'g-')


WLS()
plt.show()
