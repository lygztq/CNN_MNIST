IMG_PATH = "MNIST/train-images-idx3-ubyte"
LABEL_PATH = "MNIST/train-labels-idx1-ubyte"
TEST_IMG_PATH = "MNIST/t10k-images-idx3-ubyte"
TEST_LABEL_PATH = "MNIST/t10k-labels-idx1-ubyte"
IMG_SIZE = (28,28)
USE_QUEUE_LOADING = True
BATCH_SIZE = 50
BASE_LR_RATE = 1e-5
DECAY_RATE = 0.98
STEP = 1e4
MAX_EPOCH = 20000
SAVE_EVERY = 1000
CKPT_DIR = './checkpoints'
LOG_DIR = './logs'
