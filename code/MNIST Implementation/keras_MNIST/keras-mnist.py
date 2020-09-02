from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation
from keras.layers import Conv2D,MaxPooling2D,Flatten
from keras.optimizers import SGD,Adam
from keras.utils import np_utils
from keras.datasets import mnist # keras集成了mnist数据集
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 只是防止告警 'I tensorflow/core/platform/cpu_feature_guard.cc:140] Your CPU supports instructions that this'，没有真正解决这个问题

def load_data():
    (x_train,y_train),(x_test,y_test) = mnist.load_data()
    size = 10000 # 测试集大小
    x_train = x_train[0:size]# 截取10000个样本
    y_train = y_train[0:size]# 截取10000个样本
    x_train = x_train.reshape(size,28*28) #x_train本来是10000x28x28的数组，把它转换成10000x784的二维数组
    x_test = x_test.reshape(x_test.shape[0],28*28)

    x_train = x_train.astype('float32')#将它的元素类型转换为float32,之前为uint8
    x_test = x_test.astype('float32')

    x_train = x_train/255 # x_train之前的灰度值最大为255，最小为0，这里将它们进行特征归一化，变成了在0到1之间的小数
    x_test = x_test/255

    # one-hot编码 y_train之前可以理解为10000x1的数组，每个单元素数组的值就是样本所表示的数字
    y_train = np_utils.to_categorical(y_train,10)# 把它转换成了10000x10的数组
    y_test = np_utils.to_categorical(y_test,10)

    # print(x_train.shape) # 10000 x 784
    # print(y_train[0])
    return (x_train,y_train),(x_test,y_test)

def run():
    # 加载数据
    (x_train, y_train), (x_test, y_test) = load_data()

    #　可视化一些图像
    # plt.subplot(221)
    # plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
    # plt.subplot(222)
    # plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
    # plt.subplot(223)
    # plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
    # plt.subplot(224)
    # plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
    # plt.show()

    # 定义模型
    model = Sequential()

    # 简单的多层感知机模型（可替换）
    units = 28*28
    # 定义输入层，全连接网络，输入维度是784，有633个神经元，激活函数是Sigmoid
    model.add(Dense(input_dim=units,units=units,activation='relu'))
    for i in range(2):
        # 定义隐藏层
        model.add(Dense(units=units,activation='relu'))
    #model.add(Dense(units=units, activation='relu'))
    #model.add(Dense(units=units,activation='relu'))
    # 定义输出层，有10个神经元，也就是10个输出，激活函数是Softmax
    model.add(Dense(units=10,activation='softmax'))

    # 编译模型
    # 损失函数选择交叉熵以及设置优化器
    model.compile(loss='categorical_crossentropy',optimizer=Adam(),metrics=['accuracy'])
    model.fit(x_train,y_train,batch_size=100,epochs=20)# 模型训练 定义好batch_size和epoch

    # 对模型训练的结果进行评估
    result = model.evaluate(x_train,y_train,batch_size=100) # 这里的批次数量也要加上，不然默认是32
    print('\nTrain Acc:%.2f%%' % (result[1] * 100))
    result = model.evaluate(x_test,y_test,batch_size=100)
    print('\nTest Acc:%.2f%%' % (result[1] * 100))

if __name__ == '__main__':
    run()

    # cnn结构
    # 第一个卷积层，32个卷积核，大小５x5，卷积模式SAME,激活函数relu,输入张量的大小
    # model.add(Conv2D(filters= 32, kernel_size=(5,5), padding='Same', activation='relu',input_shape=(28,28,1)))
    # model.add(Conv2D(filters= 32, kernel_size=(5,5), padding='Same', activation='relu'))
    # 池化层,池化核大小２x2
    # model.add(MaxPool2D(pool_size=(2,2)))
    # 随机丢弃四分之一的网络连接，防止过拟合
    # model.add(Dropout(0.25))  
    # model.add(Conv2D(filters= 64, kernel_size=(3,3), padding='Same', activation='relu'))
    # model.add(Conv2D(filters= 64, kernel_size=(3,3), padding='Same', activation='relu'))
    # model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    # model.add(Dropout(0.25))
    # 全连接层,展开操作，
    # model.add(Flatten())
    # 添加隐藏层神经元的数量和激活函数
    # model.add(Dense(256, activation='relu'))    
    # model.add(Dropout(0.25))
    # 输出层
    # model.add(Dense(10, activation='softmax'))  