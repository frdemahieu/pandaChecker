import tensorflow as tf
import os
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser()
parser.add_argument(
  '--image',
  type=str,
  default=0,
  help="File to test"
)
args = parser.parse_args()


image = tf.gfile.FastGFile(args.image, 'rb').read()

label = ['blackPanda', 'redPanda']

with tf.gfile.FastGFile('tf_files/graph.pb','rb') as g:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(g.read())
    _ = tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    softmax = sess.graph.get_tensor_by_name('panda_tensor:0')
    prediction = sess.run(softmax, {'DecodeJpeg/contents:0': image})

    for item_id in range(0, len(label)):
        labelStr = label[item_id]
        score = prediction[0][item_id]
        print('%s probability : %.5f' % (labelStr,score))




