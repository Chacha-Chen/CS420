import argparse
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop,SGD,Adam
import numpy as np
import time

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
    train_data = train_data.reshape(data_num,fig_w*fig_w)
    test_data = test_data.reshape(test_num,fig_w*fig_w)
    
    train_data = train_data .astype('float32')
    test_data = test_data .astype('float32')

    train_data = train_data[:num]
    y_train = y_train[:num]

    train_data /= 255
    test_data /= 255
    y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)

    return train_data,test_data,y_train,y_test





def MLP(train_data,test_data,y_train,y_test):
	
	start = time.time()

	batch_size = 128
	num_classes = 10
	epochs = 12

	model = Sequential()
	model.add(Dense(512, activation='relu', input_shape=(45*45,)))
	model.add(Dropout(0.3))
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.3))
	model.add(Dense(10, activation='softmax'))

	model.summary()

	model.compile(loss='categorical_crossentropy',
	              optimizer=RMSprop(),
	              metrics=['accuracy'])

	history = model.fit(train_data, y_train,
	                    batch_size=batch_size,
	                    epochs=epochs,
	                    validation_data=(test_data, y_test))
	score = model.evaluate(test_data, y_test, verbose=0)
	end = time.time()

	return score[1],end-start

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
    score,time_value = MLP(train_data,test_data,y_train,y_test)
    print('Test accuracy:', score)
    print('Runing time: ',time_value)

if __name__ == "__main__":
    main()

