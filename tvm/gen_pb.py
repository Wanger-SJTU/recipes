import tensorflow.compat.v1 as tf
import os
from tensorflow.python.framework import graph_util

# pb_file_path = os.getcwd()



def weight(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1],padding="VALID",use_cudnn_on_gpu=False, data_format="NCHW", name="out") 

def matmul(x, W):
    return tf.nn.matmul(x, W, name="out1") 

with tf.Session(graph=tf.Graph()) as sess:
    x = tf.placeholder(tf.float32, shape=[3, 224,224], name='x')
    x = tf.expand_dims(x,axis=0)
    
    w = weight((3,3,3,3))

    out = conv2d(x, w)
    

    sess.run(tf.global_variables_initializer())

    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['out'])

    # 写入序列化的 PB 文件
    with tf.gfile.FastGFile('model.pb', mode='wb') as f:
        f.write(constant_graph.SerializeToString())