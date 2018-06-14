import numpy as np
np.random.seed(1337)  # for reproducibility
import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Input
import matplotlib.pyplot as plt
from sklearn import linear_model
from keras import regularizers
from keras.callbacks import TensorBoard
import pickle 
from sklearn.decomposition import PCA, KernelPCA
data_num = 60000 #The number of figures
test_num = 10000 #the number of test data
fig_w = 45       #width of each figure
y_test = np.fromfile("mnist_test_label",dtype=np.uint8)

'''
#原始的mnist
# download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
# X shape (60,000 28x28), y shape (10,000, )
(x_train, _), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255. - 0.5       # minmax_normalized
x_test = x_test.astype('float32') / 255. - 0.5         # minmax_normalized
x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))
print(x_train.shape)
print(x_test.shape)

#老师给的mnist
x_train = np.fromfile("mnist_train_data",dtype=np.uint8)
x_test = np.fromfile("mnist_test_data",dtype=np.uint8)
y_test = np.fromfile("mnist_test_label",dtype=np.uint8)
#y_test = keras.utils.to_categorical(y_test, num_classes=10)
#reshape the matrix
x_train = x_train.reshape(data_num,fig_w*fig_w)
x_test = x_test.reshape(test_num,fig_w*fig_w)
print (x_train.shape)
print("After reshape:",x_train.shape)
x_train = x_train/255
x_test = x_test/255

#中心化的mnist
x_train = np.load("new_train.npy")
x_train = x_train.reshape(data_num,fig_w, fig_w)
x_train = x_train.astype('float32') / 255. - 0.5

x_test = np.load("new_test.npy")
x_test = x_test.reshape(test_num,fig_w, fig_w)
x_test = x_test.astype('float32') / 255. - 0.5
x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))
'''

#去噪+中心化的
pkl_file = open('./data_train_reducedn.pkl', 'rb')
x_train = pickle.load(pkl_file)
x_train = x_train.astype('float32')
pkl_file.close()
pkl_file_2 = open('./data_test_reducedn.pkl', 'rb')
x_test = pickle.load(pkl_file_2)
x_test = x_test.astype('float32')
pkl_file_2.close()
x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))

# in order to plot in a 2D figure
encoding_dim = 2

# this is our input placeholder
input_img = Input(shape=(2025,))

# encoder layers
#encoded = Dense(1024, activation='relu')(input_img)
encoded = Dense(512, activation='relu',bias_regularizer=regularizers.l2(0.005),name = 'layer1')(input_img)
encoded = Dense(128, activation='relu',name = 'layer2')(encoded)
encoded = Dense(64, activation='relu',name = 'layer3')(encoded)
encoded = Dense(10, activation='relu',name = 'layer4')(encoded)
encoder_output = Dense(encoding_dim)(encoded)

# decoder layers
decoded = Dense(10, activation='relu',name = 'layer5')(encoder_output)
decoded = Dense(64, activation='relu',name = 'layer6')(decoded)
decoded = Dense(128, activation='relu',name = 'layer7')(decoded)
decoded = Dense(512, activation='relu',name = 'layer8')(decoded)
#decoded = Dense(1024, activation='relu')(decoded)
decoded = Dense(2025, activation='tanh',name = 'layer9')(decoded)

# construct the autoencoder model
autoencoder = Model(input=input_img, output=decoded)

# construct the encoder model for plotting
encoder = Model(input=input_img, output=encoder_output)

# compile autoencoder
autoencoder.compile(optimizer='adam', loss='mse')

# training
autoencoder.fit(x_train, x_train,
                epochs=100,
                batch_size=256,
                shuffle=True,
				validation_data=(x_test, x_test),
				callbacks=[TensorBoard(log_dir='/logs/autoencoder')])

# plotting
encoded_imgs = encoder.predict(x_test)
decoded_imgs = autoencoder.predict(x_test)

plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=y_test)
plt.colorbar()
plt.savefig("2D.png")

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
     #predict
     ax = plt.subplot(3, n, i+1+n)
     plt.imshow(decoded_imgs[i].reshape(45, 45))
     plt.gray()
     ax.get_yaxis().set_visible(False)
     ax.get_xaxis().set_visible(False)
     #original
     ax = plt.subplot(3, n, i+1+2*n)
     plt.imshow(x_test[i].reshape(45, 45))
     plt.gray()
     ax.get_yaxis().set_visible(False)
     ax.get_xaxis().set_visible(False)
plt.savefig("decode.png")

