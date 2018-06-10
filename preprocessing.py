import PIL.Image
import scipy.misc
import numpy as np
import scipy.ndimage as ndi
from skimage import morphology
import matplotlib.pyplot as plt
from PIL import Image
#import getdata
from skimage import io,color
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

data_num = 60000 #The number of figures
fig_w = 45       #width of each figure

data = np.fromfile("./data/mnist_train_data",dtype=np.uint8)
data_test = np.fromfile("./data/mnist_test_data",dtype=np.uint8)

data = data.reshape(data_num,fig_w,fig_w)
data_test = data_test.reshape(10000,45,45)

#二值化
for k in range(60000):
    for i in range(45):
        for j in range(45):
            if (data[k][i,j]<=0.4*255):
                data[k][i,j]=0
            else:
                data[k][i,j]=1

for k in range(10000):
    for i in range(45):
        for j in range(45):
            if (data_test[k][i,j]<=0.4*255):
                data_test[k][i,j]=0
            else:
                data_test[k][i,j]=1

data_bool = data.astype('bool')
data_bool_test = data_test.astype('bool')

nd_data=np.zeros(shape=(60000,45,45))
nd_data_test=np.zeros(shape=(10000,45,45))

#remove small objects
for i in range(60000):
    nd_data[i]=morphology.remove_small_objects(data_bool[i],min_size=16,connectivity=1)
    #plt.imshow(dst, cmap='gray_r')
for i in range(10000):
    nd_data_test[i]=morphology.remove_small_objects(data_bool_test[i],min_size=16,connectivity=1)

# fig, axes = plt.subplots(10, 10, figsize=(10, 10))

# for i in range(10):
#     for j in range(10):
#         index=i*10+j
#         axes[i][j].imshow(data[index], interpolation='none')
#         axes[i][j].set_xticks([])
#         axes[i][j].set_yticks([])

# fig, axes = plt.subplots(10, 10, figsize=(10, 10))

# for i in range(10):
#     for j in range(10):
#         index=i*10+j
#         axes[i][j].imshow(nd_data[index], interpolation='none')
#         axes[i][j].set_xticks([])
#         axes[i][j].set_yticks([])
    
nd_data_deskew=np.zeros(shape=(60000,45,45))
nd_data_deskew_test = np.zeros(shape=(10000,45,45))

for i in range(0,60000):
    nd_data_deskew[i]=deskew(nd_data[i])
for i in range(0,10000):
    nd_data_deskew_test[i]=deskew(nd_data_test[i])    
        
import pickle 
output = open('./data/data_train_reducedn.pkl', 'wb')
pickle.dump(nd_data_deskew,output)
output.close()

output = open('./data/data_test_reducedn.pkl', 'wb')
pickle.dump(nd_data_deskew_test,output)
output.close()


#little test

# testdata = data[:200,...]

# fig, axes = plt.subplots(10, 10, figsize=(10, 10))

# for i in range(10):
#     for j in range(10):
#         index=i*10+j
#         axes[i][j].imshow(testdata[index], interpolation='none')
#         axes[i][j].set_xticks([])
#         axes[i][j].set_yticks([])


# for k in range(200):
#     for i in range(45):
#         for j in range(45):
#             if (testdata[k][i,j]<=0.4*255):
#                 testdata[k][i,j]=0
#             else:
#                 testdata[k][i,j]=1
                
# fig, axes = plt.subplots(10, 10, figsize=(10, 10))
# for i in range(10):
#     for j in range(10):
#         index=i*10+j
#         axes[i][j].imshow(testdata[index], interpolation='none')
#         axes[i][j].set_xticks([])
#         axes[i][j].set_yticks([])
                
# testdata_bool = testdata.astype('bool')
# nd_testdata=np.zeros(shape=(200,45,45))

# for i in range(200):
#     nd_testdata[i]=morphology.remove_small_objects(testdata_bool[i],min_size=16,connectivity=1)

# fig, axes = plt.subplots(10, 10, figsize=(10, 10))
# for i in range(10):
#     for j in range(10):
#         index=i*10+j
#         axes[i][j].imshow(nd_testdata[index], interpolation='none')
#         axes[i][j].set_xticks([])
#         axes[i][j].set_yticks([])

# nd_testdata_deskew=np.zeros(shape=(200,45,45))

#for i in range(0,200):
#    nd_data_deskew[i]=deskew(nd_data_deskew[i])
    
fig, axes = plt.subplots(10, 10, figsize=(10, 10))
for i in range(10):
    for j in range(10):
        index=i*10+j
        axes[i][j].imshow(nd_data_deskew[index], interpolation='none')
        axes[i][j].set_xticks([])
        axes[i][j].set_yticks([])



