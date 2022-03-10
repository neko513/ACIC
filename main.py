#Teacher-student framework
#Teacher module contains clusterings predicted by teacher models and ADI-Gate
#Student module learns a cluster model(DNN) with PW-loss
#tensorflow-
##########################Our method##################
# train data: n_train*d
# train label: n_train*1
# test data: n_test*d
# test label:n_test*1
# teacher_model_path='./Result/'+ [dataset name] + '/' + str[index] +'.txt'


################################################################
#python main.py --datasource=MNIST --batch=32 --lr=0.0008 --teacher_model_num=9
#python mainpy --datasource=cifar10  --batch=32 --lr=0.0005 --teacher_model_num=6
#python main.py --datasource=FMNIST --teacher_model_num=15 --batch=32 --lr=0.0008
#python main.py --datasource=STL10  --batch=64 --lr=0.0008 --teacher_model_num=10


################################################################
import numpy as np
from random import sample
import datetime
import os
import tensorflow as tf
from tensorflow.python.platform import flags
from utils import *
from data_generate import *
import metric
from sklearn import metrics, preprocessing
from construct_model import *
from weighted_model import *
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1

FLAGS = flags.FLAGS
flags.DEFINE_string('datasource', 'MNIST', 'MNIST/cifar10/STL10/FMNIST/usps/sim')
flags.DEFINE_integer('teacher_model_num', 100, 'The number of teacher models')
flags.DEFINE_bool('save_model', True, 'True: save model / ')
flags.DEFINE_integer('epoch_num', 200, 'updata epoch number')
flags.DEFINE_integer('epoch_flag', 0, 'From this epoch, ADIGate makes sense')
flags.DEFINE_float('lr', 0.0008, 'learning rate')
flags.DEFINE_integer('op_warm_steps', 100, 'parameter num_warmup_steps of create_optimizer, meaning the period of decay ')
flags.DEFINE_integer('random_seed', 0, 'random seed')
flags.DEFINE_integer('imagesize', 28, 'The  default size of image input data, not need to change')
flags.DEFINE_bool('mergence', True, 'True: train with training and testing / False: train only with training samples')
flags.DEFINE_integer('cluster_num', 10, 'The number of clusters')
flags.DEFINE_integer('batch', 64, 'size of minibatch') 
flags.DEFINE_string('datapath', 'Dataset', 'the rootpath of dataset')
flags.DEFINE_bool('loading', False, 'True: use existed model / False: train model')
flags.DEFINE_integer('model_index', 0 , 'the index of loading model')

# Clustering results from teacher models
prior_label=np.array([]).astype('int')

# clustering model of student module
def construct_model(data, weights, reuse=False):
    if FLAGS.datasource == 'MNIST':
        return MNIST_model(data, weights, reuse)
    if FLAGS.datasource == 'cifar10':
        return cifar_model(data, weights, reuse)
    if FLAGS.datasource == 'STL10':
        return STL_model(data, weights, reuse)
    if FLAGS.datasource == 'FMNIST':
        return fashion_model(data, weights, reuse)


# weight DNN of student module
def weighted_model(class_num):
    if FLAGS.datasource == 'MNIST':
        return MNIST_weights(class_num)
    if FLAGS.datasource == 'cifar10':
        return cifar_weights(class_num)
    if FLAGS.datasource == 'STL10':
        return STL_weights(class_num)
    if FLAGS.datasource == 'FMNIST':
        return fashion_weights(class_num)

# load clustering assignments from teacher models
def prior_load(n,p_num,label):
    # p_num: the number of teacher models
    # label : True label to present the performance of teacher models
    global prior_label
    prior_label = np.zeros((p_num,n))
    print('teacher model performance...')
    print("index cluster_num acc      nmi      ari")
    for j in range(p_num):
        prelabel=np.loadtxt('./Result/' + FLAGS.datasource + '/' + str(j) + '.txt',delimiter="," ).astype('float32')
        if len(prelabel.shape)>1:
            prelabel = np.argmax(prelabel, axis=1)
        prior_label[j] = prelabel
        print(j,"           ",len(set(prelabel)),"            ",metric.metric(label.astype('int'),prelabel.astype('int')))


#ADI_gate
def adigate(teacher_model_num, index, class_num, slabel):
    # print(weight)
    global prior_label
    n = len(index)
    pre_s = np.zeros((teacher_model_num,n,n))
    s=np.zeros((n,n))
    #read data
    for i in range(teacher_model_num):
        prelabel = prior_label[i, index]
        pre = prelabel.astype('int')
        prelabel = np.eye(class_num)[pre]
        pre_s[i] = np.dot(prelabel, np.transpose(prelabel))

    # ADIGate--weight value per batch
    weight = np.ones((teacher_model_num))/teacher_model_num
    slabel = np.dot(slabel, np.transpose(slabel))
    for i in range(teacher_model_num):
        weight[i] = 1-np.sum(np.square(pre_s[i]-slabel))/(n*n)
    weight = weight / np.sum(weight)

    for i in range(teacher_model_num):
        s = s + pre_s[i]*weight[i]

    # set diag(0)
    return s-np.identity(n)


def main():
    np.random.seed(FLAGS.random_seed)
    tf.set_random_seed(FLAGS.random_seed)
    random.seed(FLAGS.random_seed)

    print('Parameter setting...')
    teacher_model_num = FLAGS.teacher_model_num
    mini_batch = FLAGS.batch
    epoch_num = FLAGS.epoch_num
    class_num = FLAGS.cluster_num

    print('prior cluster number : %d, minibatch : %d, epoch_num: %d, cluster number: %d' % (
    teacher_model_num, mini_batch, epoch_num, class_num))
    print('Loading and processing data...')
    # starttime = datetime.datetime.now()
    input1, label1, input2, label2 = data_generate()
    if FLAGS.mergence == True:
        if label2.shape[0]:
            input1 = np.concatenate((input1, input2), axis=0)
            label1 = np.concatenate((label1, label2), axis=0)
    # input1: learning data, label1: true label of learning data, input2: test data, label2: true label of test data
    # input1 and input2 are combined as our training data.
    n = input1.shape[0]
    d = input1.shape[1]

    # dataset preprocessing
    input1 = preprocessing.scale(input1)
    print('training samples number:', n, '   samples dimension:', d, '  class num:', class_num)
    prior_load(n, teacher_model_num,label1)
    batch_num = int(n/mini_batch)

    #construct DNN model
    print('Construct and initialize DNN...')
    input_batch = tf.placeholder(tf.float32, [None, d], name='x_input')
    tag = tf.placeholder(tf.int32, name='tag')
    weights = weighted_model(class_num)
    posterior_output = construct_model(input_batch, weights)  # mXc
    pre_label = tf.argmax(posterior_output, 1)  # m*1

    s_predict = tf.matmul(posterior_output, tf.transpose(posterior_output))
    s_predict = s_predict-tf.matrix_diag(tf.diag_part(s_predict))  #tf.multiply(s_predict, np.identiaaaaty(mini_batch))
    class_similarity_batch = tf.placeholder(tf.float32, [mini_batch, mini_batch], name='Soft_similarity')

    loss = PW_loss(s_predict, class_similarity_batch, Tag=tag)
    train = create_optimizer(loss, init_lr=FLAGS.lr, num_train_steps=batch_num * epoch_num,
                                 num_warmup_steps=FLAGS.op_warm_steps, use_tpu=False)
        

    m_saver = tf.train.Saver(max_to_keep = 0)
    init = tf.global_variables_initializer()
    sess = tf.Session(config=config)
    sess.run(init)
    #loading model
    file_path = 'Model/' + FLAGS.datasource + '_batch' + str(mini_batch) + '_lr' + str(FLAGS.lr) + '_epoch_flag' + str(FLAGS.epoch_flag) + str(FLAGS.random_seed)
    if FLAGS.loading == True:
        # load ith model
        model_path = file_path +'/epoch_' + str(FLAGS.model_index)
        print('Loading trained model: %s ...' % (model_path))
        if os.path.exists(model_path+'.meta') and os.path.exists(model_path+'.index') and os.path.exists(model_path+'.data-00000-of-00001'):
            m_saver.restore(sess, model_path)
            pre_train_label = sess.run(pre_label, feed_dict={input_batch: input1})
            accuracy = metric.metric(label1, np.array(pre_train_label))
            print(accuracy)
        else:
            print('Error: %s  not existing' % (model_path))

    #training model
    else:
        print('Training DNN...')

        if FLAGS.save_model and not os.path.exists(file_path):
            os.mkdir(file_path)
            model_path = file_path + '/init'
            m_saver.save(sess, model_path)
        epoch=0
        index = np.array(range(n))
        while (epoch < epoch_num):
            print("starting a new epoch......")
            # produce_minibatch
            random.shuffle(index)
            batch_loss, cost=[],[]
            pre_train_slabel = []

            for batch_index in range(batch_num):
                index_start = batch_index * mini_batch
                index_end = batch_index * mini_batch + mini_batch
                x = input1[index[index_start:index_end]]              
                slabel = sess.run(posterior_output, feed_dict={input_batch: x, tag: mini_batch})
                s = adigate(teacher_model_num, index[index_start:index_end], 50, slabel)
                # updata per batch
                batch_loss.append(sess.run(loss, feed_dict={input_batch: x, class_similarity_batch: s, tag: mini_batch}))
                sess.run(train, feed_dict={input_batch: x, class_similarity_batch: s, tag: mini_batch})
                
            # save model
            if FLAGS.save_model:
                model_path = file_path+'/epoch_' + str(epoch)
                m_saver.save(sess, model_path)

            #predict the performance of whole dataset
            if n > mini_batch:
                iter = batch_num + 1
                index = np.array(range(n))
                for i in range(iter):
                    index_start = i * mini_batch
                    index_end = i * mini_batch + mini_batch
                    batch_loss1 = 0
                    if i < iter-1:
                        x = input1[index_start:index_end]
                        ta = mini_batch
                        slabel = sess.run(posterior_output, feed_dict={input_batch: x})
                        s = adigate(teacher_model_num, index[index_start:index_end], 50, slabel)
                        batch_loss1 =sess.run(loss, feed_dict={input_batch: x, class_similarity_batch: s, tag: ta})*ta
                        pre_train_slabel.extend(slabel)
                        
                    elif index_start<n:
                        s=np.zeros((mini_batch,mini_batch))
                        x = input1[index_start:n]
                        ta = n-index_start
                        index_end = n
                        slabel = sess.run(posterior_output, feed_dict={input_batch: x})
                        s[0:ta,0:ta] = adigate(teacher_model_num, index[index_start:index_end], 50, slabel)
                        batch_loss1 = sess.run(loss, feed_dict={input_batch: x, class_similarity_batch: s, tag: ta}) * ta
                        pre_train_slabel.extend(slabel)
                    cost.append(batch_loss1)
            else:
                pre_train_slabel.extend(sess.run(posterior_output, feed_dict={input_batch: input1}))
            pre_train_label = np.argmax(pre_train_slabel, 1)
            if FLAGS.save_model:
                label_path = file_path + '/label_epoch_'+str(epoch)+'.txt'
                np.savetxt(label_path, pre_train_label, fmt="%f", delimiter=",")
            accuracy = metric.metric(label1, np.array(pre_train_label))
            acc = accuracy[0]
            nmi = accuracy[1]
            ari = accuracy[2]

            print('epoch %i avarage loss: %f, last result: loss: %f, acc: %f, nmi: %f, ari: %f'%
                  (epoch, np.mean(batch_loss),np.sum(cost)/n,np.mean(acc), np.mean(nmi), np.mean(ari)))
            epoch = epoch + 1

if __name__ == "__main__":
    main()

