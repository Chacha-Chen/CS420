# "mnist_train_data" is the data file which contains a 60000*45*45 matrix(data_num*fig_w*fig_w)
# "mnist_train_label" is the label file which contains a 60000*1 matrix. Each element i is a number in [0,9]. 
# The dataset is saved as binary files and should be read by Byte. Here is an example of input the dataset and save a random figure.

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.interactive(True)

data_num_train = 60000 #The number of figures
fig_w = 45       #width of each figure

X_train = np.fromfile("mnist_train_data",dtype=np.uint8)
X_test = np.fromfile("mnist_test_data",dtype=np.uint8)

# label = np.fromfile("./dataset/mnist_train_label",dtype=np.uint8)


#reshape the matrix
X_train = X_train.reshape(data_num_train,fig_w,fig_w)
X_test = X_test.reshape(10000,45,45)


from scipy.ndimage import interpolation

def moments(image):
    c0,c1 = np.mgrid[:image.shape[0],:image.shape[1]] # A trick in numPy to create a mesh grid
    totalImage = np.sum(image) #sum of pixels
    m0 = np.sum(c0*image)/totalImage #mu_x
    m1 = np.sum(c1*image)/totalImage #mu_y
    m00 = np.sum((c0-m0)**2*image)/totalImage #var(x)
    m11 = np.sum((c1-m1)**2*image)/totalImage #var(y)
    m01 = np.sum((c0-m0)*(c1-m1)*image)/totalImage #covariance(x,y)
    mu_vector = np.array([m0,m1]) # Notice that these are \mu_x, \mu_y respectively
    covariance_matrix = np.array([[m00,m01],[m01,m11]]) # Do you see a similarity between the covariance matrix
    return mu_vector, covariance_matrix


def deskew(image):
    c,v = moments(image)
    alpha = v[0,1]/v[0,0]
    affine = np.array([[1,0],[alpha,1]])
    ocenter = np.array(image.shape)/2.0
    offset = c-np.dot(affine,ocenter)
    return interpolation.affine_transform(image,affine,offset=offset)


# from mpl_toolkits.axes_grid1 import AxesGrid
# grid = AxesGrid(plt.figure(figsize=(8,15)), 141,  # similar to subplot(141)
#                     nrows_ncols=(10, 2),
#                     axes_pad=0.05,
#                     label_mode="1",
#                     )

# deskewed_data=np.zeros(shape=(500,45,45))
# for i in range(0,500):
#     deskewed_data[i]=deskew(X_train[i])
# for i in range(0,10):
#     im = grid[2*i].imshow(X_train[i])
#     im2 = grid[2*i+1].imshow(deskewed_data[i])
    


#for i in range(0,10):
#    plt.imshow(deskewed_data[i])

# print('original MNIST digits 1-100:')
# fig, axes = plt.subplots(10, 10, figsize=(10, 10))
 
# for i in range(10):
#     for j in range(10):
#         axes[i][j].imshow(X_train[j], interpolation='none')
#         axes[i][j].set_xticks([])
#         axes[i][j].set_yticks([])

# plt.draw()

# fig, axes = plt.subplots(2, 10, figsize=(10, 2))

# i=0
# for j in range(10):
#     axes[i][j].imshow(deskewed_data[j], interpolation='none')
#     axes[i][j].set_xticks([])
#     axes[i][j].set_yticks([])
    
# i=1     
# for j in range(10):
#     axes[i][j].imshow(X_train[j], interpolation='none')
#     axes[i][j].set_xticks([])
#     axes[i][j].set_yticks([])

print('Deskewing...(may take a while)')

deskewed_data_test=np.zeros(shape=(10000,45,45))

for i in range(0,10000):
    deskewed_data_test[i]=deskew(X_test[i])


deskewed_data_train=np.zeros(shape=(60000,45,45))

for i in range(0,60000):
    deskewed_data_train[i]=deskew(X_train[i])

print('MNIST deskewing completed...')

# fig, axes = plt.subplots(10, 10, figsize=(10, 10))
 
# for i in range(10):
#     for j in range(10):
#         axes[i][j].imshow(deskewed_data_train[j], interpolation='none')
#         axes[i][j].set_xticks([])
#         axes[i][j].set_yticks([])

# plt.draw()

print('Saving pickle file to directory dataset')
import pickle 
output = open('mnist_train_data_deskewed.pkl', 'wb')
pickle.dump(deskewed_data_train,output)
output.close()

output = open('mnist_test_data_deskewed.pkl', 'wb')
pickle.dump(deskewed_data_test,output)
output.close()


# data_test = pickle.load(pkl_file)
# #pprint.pprint(data1)

# pkl_file.close()

