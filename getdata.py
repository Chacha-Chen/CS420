import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler


data_num = 60000 #The number of figures
fig_w = 45       #width of each figure

X_train = np.fromfile("./data/mnist_train_data",dtype=np.uint8)
y_train = np.fromfile("./data/label_train",dtype=np.uint8)
X_test = np.fromfile("./data/mnist_test_data",dtype=np.uint8)
y_test = np.fromfile("./data/label_test",dtype=np.uint8)

X_train = X_train.reshape(data_num,fig_w,fig_w)
X_test = X_test.reshape(10000,fig_w,fig_w)

X_train = X_train.reshape((X_train.shape[0], -1))
X_test = X_test.reshape((X_test.shape[0], -1))

#load data
with open('data/data_train.pkl', 'rb') as f:
    X_train_deskew = pickle.load(f)
    
with open('data/data_test.pkl', 'rb') as f:
    X_test_deskew = pickle.load(f)
    
# y_train = np.fromfile("data/label_train",dtype=np.uint8)    
# y_test = np.fromfile("data/label_test",dtype=np.uint8)
    
with open('data/data_train_reducedn.pkl', 'rb') as f:
    X_train_deskew_reducedn = pickle.load(f)
    
with open('data/data_test_reducedn.pkl', 'rb') as f:
    X_test_deskew_reducedn = pickle.load(f)

X_train_deskew = X_train_deskew.reshape((X_train_deskew.shape[0], -1))
X_test_deskew = X_test_deskew.reshape((X_test_deskew.shape[0], -1))

X_train_deskew_reducedn = X_train_deskew_reducedn.reshape((X_train_deskew_reducedn.shape[0], -1))
X_test_deskew_reducedn = X_test_deskew_reducedn.reshape((X_test_deskew_reducedn.shape[0], -1))

def one_hot(Y,length):
    NewY=[]
    for i in range(len(Y)):
        content=[]
        num = int(Y[i])
        for i in range(num):
            content.append(0)
        content.append(1)
        for i in range(num+1,length):
            content.append(0)
        NewY.append(content)
    return np.array(NewY)

y_test_onehot=one_hot(y_test,10)
y_train_onehot=one_hot(y_train,10)


scaler = StandardScaler()
scaler.fit(X_train/255)
X_train_standard=scaler.transform(X_train/255)

scaler = StandardScaler()
scaler.fit(X_test/255)
X_test_standard=scaler.transform(X_test/255)

scaler = StandardScaler()
scaler.fit(X_train_deskew/255)
X_train_deskew_standard=scaler.transform(X_train_deskew/255)

scaler = StandardScaler()
scaler.fit(X_test_deskew/255)
X_test_deskew_standard=scaler.transform(X_test_deskew/255)

scaler = StandardScaler()
scaler.fit(X_train_deskew_reducedn)
X_train_deskew_reducedn_standard=scaler.transform(X_train_deskew_reducedn)

scaler = StandardScaler()
scaler.fit(X_test_deskew_reducedn)
X_test_deskew_reducedn_standard=scaler.transform(X_test_deskew_reducedn)

#max = X_train_deskew_reducedn.max()
#min = X_train_deskew_reducedn.min()
#shred = min + 0.1*(max - min)
#X_train_deskew_reducednt = np.zeros(shape=(60000,45,45))
#
#for k in range(60000):
#    for i in range(45):
#        for j in range(45):
#            if(X_train_deskew_reducedn[k][i,j] < shred ):
#                X_train_deskew_reducednt[k][i,j] = 0
#            else:
#                X_train_deskew_reducednt[k][i,j] = 1

#(array([23832, 39794]),) is empty


#index = ~(X_train.any(axis=1)).any(axis=1)
