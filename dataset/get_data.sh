wget "http://cmach.sjtu.edu.cn/course/cs420/projects/mnist.zip"
unzip -a mnist.zip
rm mnist.zip
mv ./mnist/mnist_train/mnist_train_label .
mv ./mnist/mnist_train/mnist_train_data .
mv ./mnist/mnist_test/mnist_test_label .
mv ./mnist/mnist_test/mnist_test_data .
rm -r mnist

echo Data Downloading finished🍻
echo Data preprocessing begin👀...
echo Deskewing begin👀...
python ../preprocessing/deskew.py

echo Deskewing finished🍻
echo Denoising begin👀...
echo Sorry this is super slow, 5--10min👀

python ../preprocessing/noise_reduction.py 

echo Denoising finished🍻