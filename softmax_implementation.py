
import struct
import numpy
import array
import time
import scipy.sparse
import scipy.optimize
import pickle
import numpy as np

""" The Softmax Regression class """

class SoftmaxRegression(object):

    def __init__(self, input_size, num_classes, lamda):
        
        self.input_size  = input_size  # input vector size
        self.num_classes = num_classes # number of classes
        self.lamda       = lamda       # weight decay parameter
                
        rand = numpy.random.RandomState(int(time.time()))
        
        self.theta = 0.005 * numpy.asarray(rand.normal(size = (num_classes*input_size, 1)))
    
        
    def getGroundTruth(self, labels):
        
        labels = numpy.array(labels).flatten()
        data   = numpy.ones(len(labels))
        indptr = numpy.arange(len(labels)+1)
                
        ground_truth = scipy.sparse.csr_matrix((data, labels, indptr))
        ground_truth = numpy.transpose(ground_truth.todense())
        
        return ground_truth
                
    def softmaxCost(self, theta, input, labels):
        ground_truth = self.getGroundTruth(labels)
        theta = theta.reshape(self.num_classes, self.input_size)        
        theta_x       = numpy.dot(theta, input)
        hypothesis    = numpy.exp(theta_x)      
        probabilities = hypothesis / numpy.sum(hypothesis, axis = 0)        
        cost_examples    = numpy.multiply(ground_truth, numpy.log(probabilities))
        traditional_cost = -(numpy.sum(cost_examples) / input.shape[1])        
        theta_squared = numpy.multiply(theta, theta)
        weight_decay  = 0.5 * self.lamda * numpy.sum(theta_squared)        
        cost = traditional_cost + weight_decay        
        theta_grad = -numpy.dot(ground_truth - probabilities, numpy.transpose(input))
        theta_grad = theta_grad / input.shape[1] + self.lamda * theta
        theta_grad = numpy.array(theta_grad)
        theta_grad = theta_grad.flatten()    
        return [cost, theta_grad]
                
    def softmaxPredict(self, theta, input):
        
        theta = theta.reshape(self.num_classes, self.input_size)
                
        theta_x       = numpy.dot(theta, input)
        hypothesis    = numpy.exp(theta_x)      
        probabilities = hypothesis / numpy.sum(hypothesis, axis = 0)
                
        predictions = numpy.zeros((input.shape[1], 1))
        predictions[:, 0] = numpy.argmax(probabilities, axis = 0)
        
        return predictions


input_size     = 2025    # input vector size
num_classes    = 10     # number of classes
lamda          = 0.0001 # weight decay parameter
max_iterations = 100    # number of optimization iterations

with open('data/data_train_reducedn.pkl', 'rb') as f:
    training_data = pickle.load(f)
training_data = training_data.reshape(60000,2025)
training_data = np.transpose(training_data)
training_labels = np.fromfile("data/label_train",dtype=np.uint8) 

regressor = SoftmaxRegression(input_size, num_classes, lamda)
opt_solution  = scipy.optimize.minimize(regressor.softmaxCost, regressor.theta, 
                                        args = (training_data, training_labels,), method = 'L-BFGS-B', 
                                        jac = True, options = {'maxiter': max_iterations})
opt_theta     = opt_solution.x

with open('data/data_test_reducedn.pkl', 'rb') as f:
    test_data = pickle.load(f)
    
test_data = test_data.reshape(10000,2025)
test_data = np.transpose(test_data)
test_labels = np.fromfile("data/label_test",dtype=np.uint8)
predictions = regressor.softmaxPredict(opt_theta, test_data)
correct = test_labels[:, 0] == predictions[:, 0]
print( """Accuracy :""", numpy.mean(correct))
