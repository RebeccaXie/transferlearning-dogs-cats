# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 00:32:55 2018

@author: Tian
"""
import tensorflow as tf
import os
import sys
import numpy as np
import argparse


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

FLAGS=None

LOGDIR = './log_lenet'
EPOCHS = 5000
ETA = 0.01
BATCH = 100

train_path = "./data/train1"
test_path = "./data/test1"

def normalize(file,label):
    #  computes (x - mean) / adjusted_stddev
    image_string = tf.read_file(file)
    image_decode = tf.image.decode_image(image_string)
    image_resize = tf.reshape(image_decode,(28,28,3))
    image = tf.image.per_image_standardization(image_resize)
    
    label = tf.one_hot(label, 2, 1, 0)
    return  image, label

# 将图像文件转为tensor
def file2Data(path,amount1,amount2):
  
    _dogs = [path+'/dogs/'+i for i in os.listdir(path+'/dogs')]
    _cats=[path+'/cats/'+i for i in os.listdir(path+'/cats')]
    train = tf.constant(_dogs+_cats)
    label1 = tf.constant([0]*amount1 +[1]*amount2)
    
   
    _data = tf.data.Dataset.from_tensor_slices((train,label1))
    _data = _data.map(normalize) 
    _data = _data.shuffle(5000)  # if you want to shuffle your data

    return _data


def main():
    
    # get batch data
    train_data = file2Data(train_path,1914,1942)
    test_data = file2Data(test_path,716,703)
    train_data = train_data.batch(BATCH)
    test_data = test_data.batch(BATCH)

    iterator = tf.data.Iterator.from_structure(train_data.output_types,
                                               train_data.output_shapes)
    img, label = iterator.get_next()

    train_init = iterator.make_initializer(train_data)  # initializer for train_data
    test_init = iterator.make_initializer(test_data)  # initializer for test_data

    
    # SOLUTION: Layer 1: Convolutional. Input = 28x28x3. with padding 2x2 Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean = 0, stddev = 0.1))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(img, conv1_W, strides=[1, 1, 1, 1], padding='SAME') + conv1_b

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)

    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Image summaries
    with tf.name_scope('Conv_1'):
        conv1_sample = tf.transpose(conv1[0,:,:,:],perm=[2,0,1])
        conv1_sample = tf.expand_dims(conv1_sample, axis = 3)
        conv1_image = tf.summary.image("conv1", conv1_sample, max_outputs= 6)

    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean= 0, stddev=0.1))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)

    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    #  Image summaries
    with tf.name_scope('Conv_2'):
        conv2_sample = tf.transpose(conv2[0,:,:,:],perm=[2,0,1])
        conv2_sample = tf.expand_dims(conv2_sample, axis=3)
        conv2_image = tf.summary.image("conv2", conv2_sample, max_outputs= 16)

    # SOLUTION: Layer 3: Input = 5x5x16. Output = 120.
    conv3_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 16, 120), mean= 0, stddev=0.1))
    conv3_b = tf.Variable(tf.zeros(120))
    conv3 = tf.nn.conv2d(conv2, conv3_W, strides=[1, 1, 1, 1], padding='VALID') + conv3_b

    # SOLUTION: Activation.
    conv3 = tf.nn.relu(conv3)

    #  Image summaries
#    with tf.name_scope('Conv_3'):
#        conv3_sample = tf.transpose(conv3[0,:,:,:],perm=[2,0,1])
#        conv3_sample = tf.expand_dims(conv3_sample, axis=3)
#        conv3_image = tf.summary.image("conv3", conv3_sample, max_outputs= 120)

    # Removes dimensions of size 1 from the shape of a tensor.
    fc0=tf.squeeze(conv3)

    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=0, stddev=0.1))
    fc1_b = tf.Variable(tf.zeros(84))
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b

    # SOLUTION: Activation.
    fc1 = tf.nn.relu(fc1)

    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 10.
    fc2_W = tf.Variable(tf.truncated_normal(shape=(84, 2), mean=0, stddev=0.1))
    fc2_b = tf.Variable(tf.zeros(2))
    z_2 = tf.matmul(fc1, fc2_W) + fc2_b
            
    # z_2 最后一个全连接层的输出
    # Step 5: define loss function
    # use cross entropy of softmax of logits as the loss function
    entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=z_2, labels=label, name='loss')
    loss = tf.reduce_mean(entropy)  # computes the mean over all the examples in the batch

    # Step 6: define training op
    # using gradient descent with learning rate of 0.01 to minimize loss
    optimizer = tf.train.AdamOptimizer(ETA).minimize(loss)

    # Step 7: calculate accuracy with test set
    preds = tf.nn.softmax(z_2)
    correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))

    streaming_accuracy, streaming_accuracy_update = tf.metrics.mean(correct_preds)


    with tf.name_scope('Accuracy'):
        train_accuracy_scalar = tf.summary.scalar('Train_Accuracy', accuracy)
        streaming_accuracy_scalar = tf.summary.scalar('Test_Accuracy', streaming_accuracy)


    # 把所有的summary合到一张图上
    train_merged = tf.summary.merge([train_accuracy_scalar])
    test_merged = tf.summary.merge([streaming_accuracy_scalar,conv1_image,conv2_image])


    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        saver = tf.train.Saver()
        train_writer = tf.summary.FileWriter(LOGDIR + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(LOGDIR + '/test', sess.graph)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        train_step = 0
        for _ in range(EPOCHS):
            sess.run(train_init)            
            
            try:
                while True:
                    _, l, summaries = sess.run([optimizer, loss, train_merged])             
                    train_writer.add_summary(summaries, global_step=train_step)
                    train_step += 1
            except tf.errors.OutOfRangeError:
                pass

            sess.run(test_init)
            try:
                while True:
                    _, summaries = sess.run([streaming_accuracy_update, test_merged])
                    test_writer.add_summary(summaries, global_step=train_step)
                    if train_step % 200 == 0 or train_step + 1 == EPOCHS:                
                        if train_step == EPOCHS:
                            print('Final test accuracy = %.1f%%' % (streaming_accuracy.eval() * 100))
                            saver.save(sess, "./model/lenet.model", global_step=train_step)
                        else:
                            print('Epoch %d: test accuracy = %.1f%%' % (train_step, streaming_accuracy.eval() * 100))
                        
            except tf.errors.OutOfRangeError:
                pass

        test_writer.close()
        train_writer.close()

if __name__ == '__main__':
    # Define paramaters for the model
    parser = argparse.ArgumentParser()
    FLAGS, unparsed = parser.parse_known_args()
#    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    main()
