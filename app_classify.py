# Create a folder named model, download pre-trained model unzip and moved to folder "model"
# wget https://storage.googleapis.com/openimages/2017_07/classes-trainable.txt 
# wget https://storage.googleapis.com/openimages/2017_07/class-descriptions.csv 
# wget https://storage.googleapis.com/openimages/2017_07/oidv2-resnet_v1_101.ckpt.tar.gz 
# tar -xzf oidv2-resnet_v1_101.ckpt.tar.gz , move model files to "model" folder 

import tensorflow as tf
import os, json, urllib
from PIL import Image
from flask import Flask, request
from collections import OrderedDict

app = Flask(__name__)

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('labelmap', 'model/classes-trainable.txt',
                    'Labels, one per line.')
flags.DEFINE_string('dict', 'model/class-descriptions.csv',
                    'Descriptive string for each label.')
flags.DEFINE_string('checkpoint_path', 'model/',
                    'Path to checkpoint file.')
flags.DEFINE_integer('top_k', 10, 'Maximum number of results to show.')
flags.DEFINE_float('score_threshold', None, 'Score threshold.')

def LoadLabelMap(labelmap_path, dict_path):
  """Load index->mid and mid->display name maps.
  Args:
    labelmap_path: path to the file with the list of mids, describing
        predictions.
    dict_path: path to the dict.csv that translates from mids to display names.
  Returns:
    labelmap: an index to mid list
    label_dict: mid to display name dictionary
  """
  labelmap = [line.rstrip() for line in tf.gfile.GFile(labelmap_path)]
  label_dict = {}
  for line in tf.gfile.GFile(dict_path):
    words = [word.strip(' "\n') for word in line.split(',', 1)]
    label_dict[words[0]] = words[1]

  return labelmap, label_dict

# Load the Tensorflow model into memory.
g=tf.get_default_graph()
sess = tf.Session()
saver = tf.train.import_meta_graph(FLAGS.checkpoint_path+'oidv2-resnet_v1_101.ckpt.meta')
saver.restore(sess,FLAGS.checkpoint_path+'oidv2-resnet_v1_101.ckpt')
        
# load labelMap        
labelmap, label_dict = LoadLabelMap(FLAGS.labelmap, FLAGS.dict)

@app.route("/predict/<path:url>", methods=['GET']) 

def predict_labels(url):
    label_score = {}
    
    response = urllib.request.urlopen(url)
    image_data = response.read()
    
    input_values = g.get_tensor_by_name('input_values:0')
    predictions = g.get_tensor_by_name('multi_predictions:0')
    
    predictions_eval = sess.run(
    predictions, feed_dict={ input_values: [image_data] })
    
    top_k = predictions_eval.argsort()[::-1]  # indices sorted by score
    if FLAGS.top_k > 0:
        top_k = top_k[:FLAGS.top_k]
    if FLAGS.score_threshold is not None:
        top_k = [i for i in top_k
               if predictions_eval[i] >= FLAGS.score_threshold]

    for idx in top_k:
        mid = labelmap[idx]
        display_name = label_dict[mid]
        score = predictions_eval[idx]
        
        label_score[label_dict[mid]] = predictions_eval[idx]
    return str(label_score)
        
@app.route("/")
def hello():
    return "Image Classification API: \n /predict/image_url"

if __name__ == '__main__':
    app.run(host='0.0.0.0',port = 5023, debug=False, threaded=False)

