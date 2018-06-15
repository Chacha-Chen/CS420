wget -o ./dataset/mnist "http://cmach.sjtu.edu.cn/course/cs420/projects/mnist.zip"
unzip -a mnist.zip
rm mnist.zip
mv ./mnist/mnist_train/mnist_train_label .
mv ./mnist/mnist_train/mnist_train_data .
mv ./mnist/mnist_test/mnist_test_label .
mv ./mnist/mnist_test/mnist_test_data .
rm -r mnist
