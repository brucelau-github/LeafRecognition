#!/usr/bin/python
import signal
import os,time
import tensorflow as tf
from PIL import Image
import sys
from nets.inception_v3 import *
import numpy as np
sys.path.append("..")

sess = tf.Session()

checkpoint_dir = '/home/brucelau/workbench/data/checkpoints'
input_tensor = tf.placeholder(tf.float32, [None, 299, 299, 3])
arg_scope = inception_v3_arg_scope()
with slim.arg_scope(arg_scope):
        logits, end_points = inception_v3(input_tensor, is_training=False, num_classes=5)
        saver = tf.train.Saver()
ckpt = tf.train.latest_checkpoint(checkpoint_dir)
saver.restore(sess, ckpt)

def inference(image):
	im = Image.open(image).resize((299, 299))
	im = Image.open(image).resize((299, 299))
	im = np.array(im) / 255.0
	im = im.reshape(-1, 299, 299, 3)
	start = time.time()
	predict_values, logit_values = sess.run([end_points['Predictions'], logits], feed_dict={input_tensor:im})
	print 'inference a image take time {0}'.format(time.time() - start)
	return predict_values

def receive_signal(signum, stack):
	print 'Received:', signum
	predict = inference('/home/brucelau/workbench/data/leaf_photos/norway_maple/1249060544_0003.jpg')
	outf = open("/tmp/py_server_out","w")
	outf.write(str(time.time()))
	outf.write("norway_maple:{0}".format(predict[0][0]))
	outf.write("siberian_crab_apple:{0}".format(predict[0][1]))
	outf.write("siberian_elm:{0}".format(predict[0][2]))
	outf.write("silver_maple:{0}".format(predict[0][3]))
	outf.write("yellow_buckeye:{0}".format(predict[0][4]))
	outf.close()
	print 'the predict is :'
	print 'norway_maple:',predict[0][0]
	print 'siberian_crab_apple:',predict[0][1]
	print 'siberian_elm:',predict[0][2]
	print 'silver_maple:',predict[0][3]
	print 'yellow_buckeye:',predict[0][4]
	
signal.signal(signal.SIGUSR1, receive_signal)
signal.signal(signal.SIGUSR2, receive_signal)

pidfifo = "/tmp/py_server_pid"
#os.mkfifo(pidfifo)

pidf = open(pidfifo,"w")
pidf.write(str(os.getpid()))
pidf.close()
print 'My PID is:', os.getpid()

while True:
	print 'Waiting...'
  	signal.pause()
