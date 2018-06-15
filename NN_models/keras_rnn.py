import argparse
import keras  
from keras.layers import SimpleRNN  
from keras.layers import Dense, Activation,Dropout 
from keras.models import Sequential  
from keras.optimizers import Adam  
import numpy as np 
import time

data_num = 60000 
fig_w = 45     
test_num = 10000

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
    train_data = train_data.reshape(data_num,fig_w,fig_w)
    test_data = test_data.reshape(test_num,fig_w,fig_w)
    
    train_data = train_data .astype('float32')
    test_data = test_data .astype('float32')

    train_data = train_data[:num]
    y_train = y_train[:num]

    train_data /= 255
    test_data /= 255
    y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)

    return train_data,test_data,y_train,y_test



def RNN(train_data,test_data,y_train,y_test):
	start = time.time()

	learning_rate = 0.001  
  
	n_hidden = 128  
	batch_size = 128
	num_classes = 10
	epochs = 48

	model = Sequential()  
	model.add(SimpleRNN(n_hidden,  
	               batch_input_shape=(None, 45, 45),  
	               unroll=True))

	model.add(Dropout(0.2))
	model.add(Dense(num_classes))  
	model.add(Activation('softmax'))  
	  
	adam = Adam(lr=learning_rate)  
	model.summary()  
	model.compile(optimizer=adam,  
	              loss='categorical_crossentropy',  
	              metrics=['accuracy'])  
	  
	model.fit(train_data, y_train,  
	          batch_size=batch_size,  
	          epochs=epochs,  
	          verbose=1,  
	          validation_data=(test_data, y_test))  
	  
	scores = model.evaluate(test_data, y_test, verbose=0)  
	end = time.time()
	return scores[1],end-start

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
    score,time_value = RNN(train_data,test_data,y_train,y_test)
    print('Test accuracy:', score)
    print('Runing time: ',time_value)

if __name__ == "__main__":
    main()