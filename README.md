# Tensorflow Spam Detector model

## Install tflite-model-maker
Install the package to create the spam detection model
```
!pip install -q tflite-model-maker
```

## Import dependencies
```py
import numpy as np
import os

from tflite_model_maker import configs
from tflite_model_maker import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import text_classifier
from tflite_model_maker.text_classifier import DataLoader
```
## Get Data From csv file
Fetch data and store it in a file named comment-spam=extras.csv. Give the url to fetch data from and assign extract to false since dta is not zipped.
```py
data_file = tf.keras.utils.get_file(fname='comment-spam-extras.csv',
                                    origin='https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/code/6.5/jm_blog_comments_extras.csv',
                                    extract=False
                                    )
```

## Model Specifications
Assign specifications to be used to train the modekl
```py
# use average_word_evec model
spec = model_spec.get('average_word_vec')
# use 2000 words for training
spec.num_words = 2000
# Give the length to be used per token
spec.seq_len = 20
# Give the dimensions used for training
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
```

## Train the Data
```py
# Use 90% for training and 10% for testing
train_data, test_data = data.split(0.9)

model = text_classifier.create(train_data, model_spec=spec, epochs=50)
```

## Export the Model
Export the data to /tmp/js_export in a tfjs format vith LABEL and VOCAB included
```py
model.export(export_dir='/tmp/js_export', export_format=[ExportFormat.TFJS, ExportFormat.LABEL, ExportFormat.VOCAB])
```

## Zip the Model for download
```
!zip -r /tmp/js_export/ModelFiles.zip /tmp/js_export/
```