import keras
import pickle
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
import time
import argparse


def read_data(flag,num):

    fig_w = 45 
    data_num = 60000    
    test_num = 10000
    num_classes = 10

    if (flag == 0):
      train_data = np.fromfile("mnist_train_data",dtype=np.uint8)
      test_data = np.fromfile("mnist_test_data",dtype=np.uint8)
    elif (flag == 1):
      train_data = np.load("mnist_train_data_deskewed.npy")
      test_data = np.load("mnist_test_data_deskewed.npy")
    else:
      train_data = np.load("mnist_train_data_denoised.npy")
      test_data = np.load("mnist_test_data_denoised.npy")

    y_train = np.fromfile("mnist_train_label",dtype=np.uint8)
    y_test = np.fromfile("mnist_test_label",dtype=np.uint8)
    train_data = train_data.reshape(data_num,fig_w,fig_w,1)
    test_data = test_data.reshape(test_num,fig_w,fig_w,1)
    
    train_data = train_data .astype('float32')
    test_data = test_data .astype('float32')

    train_data = train_data[:num]
    y_train = y_train[:num]

    train_data /= 255
    test_data /= 255
    y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)

    return train_data,test_data,y_train,y_test


def CNN(train_data,test_data,y_train,y_test):

    start = time.time()

    batch_size = 128
    num_classes = 10
    epochs = 12
    input_shape = (45,45,1)

    model = Sequential()
    model.add(Conv2D(32,
                     activation='relu',
                     input_shape=input_shape,
                     nb_row=3,
                     nb_col=3))
    model.add(Conv2D(64, activation='relu',
                     nb_row=3,
                     nb_col=3))
    model.add(Conv2D(128, activation='relu',
                     nb_row=3,
                     nb_col=3))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.35))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.metrics.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    model.fit(train_data, y_train, batch_size=batch_size, epochs=epochs,
              verbose=1,validation_data=(test_data, y_test))
    score = model.evaluate(test_data, y_test, verbose=0)

    return score[1],time.time()-start

def main():
    parser = argparse.ArgumentParser()
    '''
    根据参数选择dataset和training data amount
    '''
    parser.add_argument('-f', type=int, choices=[0, 1, 2],
                    help="the symbol of dataset ")
    parser.add_argument('-n', type=int, default = 60000, required=False,
                        help="training data number")
    args = parser.parse_args()


    train_data,test_data,y_train,y_test = read_data(args.f,args.n)
    score,time_value = CNN(train_data,test_data,y_train,y_test)
    print('Test accuracy:', score)
    print('Runing time: ',time_value)

if __name__ == "__main__":
    main()

