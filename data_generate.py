import pickle
import numpy as np
import scipy.io as sio
datapath='/ssd/wys/data'
#'MNIST' # cifar10   stl10 #fashion_mnist #usps


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_data_one(file, s):
    batch = unpickle(file)

    data = batch[b'data']
    labels = batch[b'labels']

    print("Loading %s : %d." % (file, len(data)))
    return data, labels

def load_data(files, data_dir, label_count, image_size, img_channels, s='cifar10'):
    data, labels = load_data_one(data_dir + '/' + files[0], s)
    for f in files[1:]:
        data_n, labels_n = load_data_one(data_dir + '/' + f, s)
        data = np.append(data, data_n, axis=0)
        labels = np.append(labels, labels_n, axis=0)
    labels = np.array([[float(i == label) for i in range(label_count)] for label in labels])
    data = data.reshape([-1, img_channels, image_size, image_size])
    data = data.transpose([0, 2, 3, 1])
    data = data.reshape([-1, img_channels*image_size*image_size]).astype('float32')
    return data, labels

# Our DEC model data_genernate with tensorflow
import tensorflow as tf
FLAGS = tf.app.flags.FLAGS
def data_generate():
    # MNIST
    if FLAGS.datasource == 'MNIST':
        label1 = np.fromfile(FLAGS.datapath + '/MNIST/raw/train-labels-idx1-ubyte', dtype=np.uint8)[8:]
        input1 = np.fromfile(FLAGS.datapath + '/MNIST/raw/train-images-idx3-ubyte', dtype=np.uint8)[16:]
        label2 = np.fromfile(FLAGS.datapath + '/MNIST/raw/t10k-labels-idx1-ubyte', dtype=np.uint8)[8:]
        input2 = np.fromfile(FLAGS.datapath + '/MNIST/raw/t10k-images-idx3-ubyte', dtype=np.uint8)[16:]
        input1 = input1.reshape(int(input1.shape[0]/784), 784).astype('float32')
        input2 = input2.reshape(int(input2.shape[0]/784), 784).astype('float32')
        FLAGS.imagesize = 28
    # cifar10
    elif FLAGS.datasource == 'cifar10':
        image_size = 32
        img_channels = 3
        data_dir = FLAGS.datapath + '/CIFAR10/cifar-10-batches-py'
        meta = unpickle(data_dir + '/batches.meta')
        label_names = meta[b'label_names']
        label_count = len(label_names)
        train_files = ['data_batch_%d' % d for d in range(1, 6)]
        input1, label1 = load_data(train_files, data_dir, label_count, image_size, img_channels)
        input2, label2 = load_data(['test_batch'], data_dir, label_count, image_size, img_channels)
        FLAGS.imagesize = 32
    # stl10
    elif FLAGS.datasource == 'STL10':
        input1 = np.fromfile(FLAGS.datapath + '/STL10/stl10_binary/train_X.bin', dtype=np.uint8)
        label1 = np.fromfile(FLAGS.datapath + '/STL10/stl10_binary/train_y.bin', dtype=np.uint8)
        input2 = np.fromfile(FLAGS.datapath + '/STL10/stl10_binary/test_X.bin', dtype=np.uint8)
        label2 = np.fromfile(FLAGS.datapath + '/STL10/stl10_binary/test_y.bin', dtype=np.uint8)
        #before
        input1 = input1.reshape(5000, 27648).astype('float32')
        input2 = input2.reshape(8000, 27648).astype('float32')

        label1 = label1 - 1
        label2 = label2 - 1
        FLAGS.imagesize = 96
    # fashion-MNIST
    elif FLAGS.datasource == 'FMNIST':
        label1 = np.fromfile(FLAGS.datapath + '/FMNIST/raw/train-labels-idx1-ubyte', dtype=np.uint8)[8:]
        input1 = np.fromfile(FLAGS.datapath + '/FMNIST/raw/train-images-idx3-ubyte', dtype=np.uint8)[16:]
        label2 = np.fromfile(FLAGS.datapath + '/FMNIST/raw/t10k-labels-idx1-ubyte', dtype=np.uint8)[8:]
        input2 = np.fromfile(FLAGS.datapath + '/FMNIST/raw/t10k-images-idx3-ubyte', dtype=np.uint8)[16:]
        input1 = input1.reshape(int(input1.shape[0] / 784), 784).astype('float32')
        input2 = input2.reshape(int(input2.shape[0] / 784), 784).astype('float32')
        FLAGS.imagesize = 28
    # ['__version__', '__header__', 'ff', '__globals__']

   

    # data reshape
    input1, label1, input2, label2 = np.array(input1), np.array(label1), np.array(input2), np.array(label2)

    if len(label1.shape) > 1:
        label1 = np.argmax(label1, axis=1)
        label2 = np.argmax(label2, axis=1)

    return input1, label1, input2, label2


