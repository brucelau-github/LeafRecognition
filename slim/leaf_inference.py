import tensorflow as tf
slim = tf.contrib.slim
from PIL import Image
import sys
sys.path.append("..")
from nets.inception_v3 import *
import numpy as np
import os
import time

class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

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
    #image_reader = ImageReader()
    #im = tf.gfile.FastGFile(image, 'rb').read()
    #im = image_reader.decode_jpeg(sess, im)
    #im = tf.image.resize_image_with_crop_or_pad(im,299,299)
    # im = tf.placeholder(dtype=tf.string)
    # image = sess.run(self._decode_jpeg,
    #                 feed_dict={tf.placeholder(dtype=tf.string): image_data})
    im = Image.open(image).resize((299, 299))
    im = np.array(im) / 255.0
    im = im.reshape(-1, 299, 299, 3)
    start = time.time()
    predict_values, logit_values = sess.run(
        [end_points['Predictions'], logits], feed_dict={input_tensor:im})
    print 'a image take time {0}'.format(time.time() - start)
    return image, predict_values


if __name__ == "__main__":
    sample_images = '/home/brucelau/workbench/data/leaf_photos/norway_maple/1249060544_0002.jpg'
    image, predict = inference(sample_images)
    print 'the score with the {0} is {1} '.format(image, predict)
