from keras.layers import Dense
from keras.models import Sequential
from keras.datasets import mnist
from keras.utils import np_utils

#载入数据
(x_train,y_train),(x_test,y_test) = mnist.load_data()
# print('x_train:',x_train.shape,'y_train:',y_train.shape)

#数据处理
x_train = x_train.reshape(x_train.shape[0],-1)/255.0
x_test = x_test.reshape(x_test.shape[0],-1)/255.0
y_train = np_utils.to_categorical(y_train,num_classes=10)
y_test = np_utils.to_categorical(y_test,num_classes=10)

#创建模型，输入784个神经元，输出10个神经元
model = Sequential([
        Dense(units=200,input_dim=784,bias_initializer='one',activation='tanh'),
        Dense(units=100,bias_initializer='one',activation='tanh'),
        Dense(units=10,bias_initializer='one',activation='softmax')
    ])

#定义优化器，编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

#训练模型
model.fit(x_train,y_train,batch_size=32,epochs=5,verbose=2)

#评估模型
loss_train,accuracy_train = model.evaluate(x_train,y_train)
print('train loss:',loss_train,'train accuracy:',accuracy_train)
loss_test,accuracy_test = model.evaluate(x_test,y_test)
print('test loss:',loss_test,'test accuracy:',accuracy_test)
