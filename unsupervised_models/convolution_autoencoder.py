from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras.callbacks import TensorBoard
import pickle 
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

data_num = 60000 #The number of figures
test_num = 10000 #the number of test data
fig_w = 45       #width of each figure
y_test = np.fromfile("mnist_test_label",dtype=np.uint8)
'''
#原来的数据集
(x_train, _), (x_test, _) = mnist.load_data()

#中心化的数据集
x_train = np.load("new_train.npy")
x_test = np.load("new_test.npy")

'''
#去噪+中心化的数据集
pkl_file = open('./data_train_reducedn.pkl', 'rb')
x_train = pickle.load(pkl_file)
x_train = x_train.astype('float32')
pkl_file.close()

pkl_file_2 = open('./data_test_reducedn.pkl', 'rb')
x_test = pickle.load(pkl_file_2)
x_test = x_test.astype('float32')
pkl_file_2.close()

x_train = np.reshape(x_train, (data_num, 45, 45, 1))  # adapt this if using `channels_first` image data format
x_train = x_train[:,:44,:44,:]
print (x_train.shape)
x_test = np.reshape(x_test, (test_num, 45, 45, 1))  # adapt this if using `channels_first` image data format
x_test = x_test[:,:44,:44,:]
input_img = Input(shape=(44, 44, 1))  # adapt this if using `channels_first` image data format


#encode
x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)


#decode
x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)

encoder = Model(input=input_img, output=encoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.fit(x_train, x_train,
                epochs=1,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir='/logs/convolution_autoencoder')]
				)
				
				
decoded_imgs = autoencoder.predict(x_test)
encoded_imgs = encoder.predict(x_train)
print (encoded_imgs.shape)

#print figure
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(44, 44))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n+1)
    plt.imshow(decoded_imgs[i].reshape(44, 44))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig("convolution_autoencoder.png")


