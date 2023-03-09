!pip install -q tflite-model-maker

import numpy as np
import os

from tflite_model_maker import configs
from tflite_model_maker import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import text_classifier
from tflite_model_maker.text_classifier import DataLoader

import tensorflow as tf
assert tf.__version__.startswith('2')
tf.get_logger().setLevel('ERROR')

data_file = tf.keras.utils.get_file(fname='comment-spam-extras.csv',
                                    origin='https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/code/6.5/jm_blog_comments_extras.csv',
                                    extract=False
                                    )

spec = model_spec.get('average_word_vec')
spec.num_words = 2000
spec.seq_len = 20
spec.wordvec_dim = 7

data = DataLoader.from_csv(
    filename=data_file,
    text_column='commenttext',
    label_column='spam',
    model_spec=spec,
    delimiter=',',
    shuffle=True,
    is_training=True
)

train_data, test_data = data.split(0.9)

model = text_classifier.create(train_data, model_spec=spec, epochs=50)

model.export(export_dir='/tmp/js_export', export_format=[ExportFormat.TFJS, ExportFormat.LABEL, ExportFormat.VOCAB])
!zip -r /tmp/js_export/ModelFiles.zip /tmp/js_export/