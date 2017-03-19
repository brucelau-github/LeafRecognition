import tensorflow as tf
slim = tf.contrib.slim
from PIL import Image
import sys
sys.path.append("..")
from nets.inception_v3 import *
import numpy as np
import os
import time

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    'image_dir', '', 'The path of image to inference.')
def inference(image):
    if not tf.gfile.Exists(image):
        tf.logging.fatal('File does not exist %s', image)

    checkpoint_dir = '/home/brucelau/workbench/data/checkpoints'
    input_tensor = tf.placeholder(tf.float32, [None, 299, 299, 3])
    sess = tf.Session()
    arg_scope = inception_v3_arg_scope()
    with slim.arg_scope(arg_scope):
        logits, end_points = inception_v3(
            input_tensor, is_training=False, num_classes=5)
        saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint(checkpoint_dir)
    saver.restore(sess, ckpt)
    im = Image.open(image).resize((299, 299))
    im = np.array(im) / 255.0
    im = im.reshape(-1, 299, 299, 3)
    start = time.time()
    predict_values, logit_values = sess.run(
        [end_points['Predictions'], logits], feed_dict={input_tensor:im})
    print 'a image take time {0}'.format(time.time() - start)
    return image, predict_values

def main(_):
    if not FLAGS.image_dir:
        raise ValueError('You must supply the image path with --image_dir')
    image, predict = inference(FLAGS.image_dir)
    print 'the predition is :'
    print 'norway_maple:',predict[0][0]
    print 'siberian_crab_apple:',predict[0][1]
    print 'siberian_elm:',predict[0][2]
    print 'silver_maple:',predict[0][3]
    print 'yellow_buckeye:',predict[0][4]

if __name__ == "__main__":
	tf.app.run()
