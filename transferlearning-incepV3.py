# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 03:50:25 2018

Lenet --> Train
@author: Tian
"""

import tensorflow as tf
import os
import sys
import numpy as np
import argparse

TRAIN_SIZE = 3200
TEST_SIZE = 800

# 
LOGDIR = './log_incepV3'

MODEL_DIR = './model'  # inception-v3模型的文件夹
MODEL_FILE = 'tensorflow_inception_graph.pb'  # inception-v3模型文件名

# inception-v3模型参数
BOTTLENECK_TENSOR_SIZE = 2048


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

FLAGS=None

train_path = "./train"
test_path = "./test"

class Inception:
    """
    预训练好的inception-v3包含1000种分类.
    """
    # 数据层.    # botttel-neck 层.
    tensor_name_input_jpeg = "DecodeJpeg/contents:0"
    tensor_name_transfer_layer = 'pool_3/_reshape:0'

    def __init__(self):

        # 创建tensorflow计算图.
        self.graph = tf.Graph()

        # 将新的计算图设置为默认图.
        with self.graph.as_default():
            # 打开pre_trained模型.
            path = os.path.join(MODEL_DIR, MODEL_FILE)
            with tf.gfile.FastGFile(path, 'rb') as file:
                # 复制定义好的计算图到新的图中，先创建一个空的图.
                graph_def = tf.GraphDef()
                # 加载proto-buf中的模型.
                graph_def.ParseFromString(file.read())
                # 最后复制pre-def图的到默认图中.
                tf.import_graph_def(graph_def, name='')


        # 获取计算图最后一层的数据
        self.transfer_layer = self.graph.get_tensor_by_name(self.tensor_name_transfer_layer)
        # 创建会话执行图.
        self.session = tf.Session(graph=self.graph)

    def _create_feed_dict(self, image):
        if image is not None:
            # Image is passed in as a 3-dim array that is already decoded.
            feed_dict = {self.tensor_name_input_jpeg: image.eval()}
        else:
            raise ValueError("Either image or image_path must be set.")

        return feed_dict 
    
    def transfer_values(self, image):

        # Create a feed-dict for the TensorFlow graph with the input image.
        feed_dict = self._create_feed_dict(image)
        transfer_values = self.session.run(self.transfer_layer, feed_dict=feed_dict)
        # 变成一维数据输出
        transfer_values = np.squeeze(transfer_values)
        return transfer_values

    def close(self):
        self.session.close()
        
def normalize(file,label):
    #  computes (x - mean) / adjusted_stddev
    image_string = tf.read_file(file)
    image_decode = tf.image.decode_image(image_string)
    image_resize = tf.reshape(image_decode,(28,28,3))
    image = tf.image.per_image_standardization(image_resize)
    
    label = tf.one_hot(label, 2, 1, 0)
    return  image, label

# 将图像文件转为tensor
def file2Data(path,amount):
  
    train_dogs = [path+'/dogs/'+i for i in os.listdir(path+'/dogs')]
    train_cats=[path+'/cats/'+i for i in os.listdir(path+'/cats')]
    train = tf.constant(train_dogs+train_cats)
    label1 = tf.constant([0]*int(amount/2) +[1]*int(amount/2))
    
   
    train_data = tf.data.Dataset.from_tensor_slices((train,label1))
    train_data = train_data.map(normalize) 
    train_data = train_data.shuffle(5000)  # if you want to shuffle your data

    return train_data
##### ka - zai - du - qu - shu - ju 
        
#    test_dogs = [test_path+'\\dogs\\'+i for i in os.listdir(test_path+'\\dogs')]
#    test_cats = [test_path+'\\cats\\'+i for i in os.listdir(test_path+'\\cats')]
#    test = tf.constant(test_dogs + test_cats)
#    label2 = tf.constant([0]*600 +[1]*600)
#    
#    test_data = tf.data.Dataset.from_tensor_slices((test,label2))
#    test_data = test_data.map(normalize)
    


# # 使用inception-v3处理图片获取特征向量
#def image2Botteleneck(sess, image_data, image_data_tensor, bottleneck_tensor):
#
#    bottleneck_values = sess.run(bottleneck_tensor,{image_data_tensor: image_data})
#    bottleneck_values = np.squeeze(bottleneck_values)  # 将四维数组压缩成一维数组
#    
#    return bottleneck_values



def main():
    
    # get batch data
    train_data = file2Data(train_path,TRAIN_SIZE)
    test_data = file2Data(test_path,TEST_SIZE)
    train_data = train_data.batch(FLAGS.batch_size)
    test_data = test_data.batch(FLAGS.batch_size)

    iterator = tf.data.Iterator.from_structure(train_data.output_types,
                                               train_data.output_shapes)
    img, label = iterator.get_next()

    train_init = iterator.make_initializer(train_data)  # initializer for train_data
    test_init = iterator.make_initializer(test_data)  # initializer for test_data

    # 读取训练好的inception-v3模型
    model = Inception()
#    transfer_output = model.transfer_values(img)
      
    # 定义最后一层全连接层，输入：Inception-V3的bottleneck层，输出：2  
    transfer_output = tf.placeholder(tf.float32, [None, BOTTLENECK_TENSOR_SIZE], name='BottleneckInputPlaceholder')

    # Fully Connected. Input = BOTTLENECK_TENSOR_SIZE. Output = 2.   
    fc_W = tf.Variable(tf.truncated_normal(shape=(BOTTLENECK_TENSOR_SIZE, 2), mean=0, stddev=0.1))
    fc_b = tf.Variable(tf.zeros(2))
    z = tf.matmul(transfer_output, fc_W) + fc_b
            
    # z_2 最后一个全连接层的输出
    # Step 5: define loss function
    # use cross entropy of softmax of logits as the loss function
    entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=z, labels=label, name='loss')
    loss = tf.reduce_mean(entropy)  # computes the mean over all the examples in the batch

    # Step 6: define training op
    # using gradient descent with learning rate of 0.01 to minimize loss
    optimizer = tf.train.AdamOptimizer(FLAGS.eta).minimize(loss)

    # Step 7: calculate accuracy with test set
    preds = tf.nn.softmax(z)
    correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))

    streaming_accuracy, streaming_accuracy_update = tf.metrics.mean(correct_preds)


    with tf.name_scope('Accuracy'):
        train_accuracy_scalar = tf.summary.scalar('Train_Accuracy', accuracy)
        streaming_accuracy_scalar = tf.summary.scalar('Test_Accuracy', streaming_accuracy)


    # 把所有的summary合到一张图上
    train_merged = tf.summary.merge([train_accuracy_scalar])
    test_merged = tf.summary.merge([streaming_accuracy_scalar])


    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        train_writer = tf.summary.FileWriter(LOGDIR + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(LOGDIR + '/test', sess.graph)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        train_step = 0
        for _ in range(FLAGS.epochs):
            sess.run(train_init)            
            
            try:
                while True:
                    _, l, summaries = sess.run([optimizer, loss, train_merged],feed_dict={transfer_output: model.transfer_values(img)})             
                    train_writer.add_summary(summaries, global_step=train_step)
                    train_step += 1
            except tf.errors.OutOfRangeError:
                pass

            sess.run(test_init)
            try:
                while True:
                    _, summaries = sess.run([streaming_accuracy_update, test_merged],feed_dict={transfer_output: model.transfer_values(img)})
                    test_writer.add_summary(summaries, global_step=train_step)
            except tf.errors.OutOfRangeError:
                pass

        test_writer.close()
        train_writer.close()

if __name__ == '__main__':
    # Define paramaters for the model
    parser = argparse.ArgumentParser()
#    parser.add_argument("--mnist_folder", default='data') #mnist数据下载目录
    parser.add_argument("--eta",default=0.01,type=float) #学习率
    parser.add_argument("--epochs",default=50,type=int) #训练次数
    parser.add_argument("--batch_size",default=10,type=int) #每次训练批量
    parser.add_argument("--test_interval",default=10, type=int) #测试间隔
#    parser.add_argument("--log_dir",default='log_inceV3/')  #日志目录
    FLAGS, unparsed = parser.parse_known_args()
#    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    main()
