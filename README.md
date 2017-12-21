# CNN for MNIST

- - -

## What's this?

This is a small but complete CNN model for MNIST implemented using TensorFlow. Obviously this model is a result of a boring period. However, you can learning some tensorflow API from this toy model ;)

- - -

## To run this model, we need

* TensorFlow r1.4
* numpy
* scipy
* PIL
* matplotlib
* argparse 

- - -

## How to use

### Download dataset

You can download the dataset from [Lecun's website](http://yann.lecun.com/exdb/mnist/) and put data files into 'MNIST' directory.

You should download these four files

* train-images-idx3-ubyte
* train-labels-idx1-ubyte
* t10k-images-idx3-ubyte
* t10k-labels-idx1-ubyte

### Training

Run train.py

`py -2 train.py`

or

`python train.py`

### Test

Run test.py

`python test.py`

### Image Recognition

Run recognition.py

`python recognition.py --img [your img file path]`