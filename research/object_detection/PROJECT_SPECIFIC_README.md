# Handwriting recognition in scanned documents 

## Procedure (python3)

### 0. Before we start

Skim through the articles mentioned in the Bibliography section

### 1. Setting up tensorflow repository

* Clone this repository or the [official tensorflow models repository](https://github.com/tensorflow/models).

* Follow the [installation steps](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) given in the `research/object_detection` subdirectory.

Caveats:
* Use protobuf 3.0.0. Newer versions of protobuf does not work. Version 3.4.0 also works.
* For pycoco installation, if `make` is not available. Running `python setup.py install` is an alternative.
* ```export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim``` is critically important. The `slim` package included in the repository is different from the `tensorflow.contrib.slim` shipped with default tensorflow. For windows installations, just copying `/research/slim` to `/research/object_detection/slim` works nicely. 

### 2. Labling custom dataset

Use [labelimg](https://github.com/tzutalin/labelImg) to label the images. Export the labeled data as PascalVOC.

Or download the labeled data from [this link](https://github.com/akashchandwani/training_data)

### 3. Converting raw data to tensorflow records

* Use `/research/object_detection/my_utils/xml_to_csv.py` to convert the PascalVOC xml to csv which is a slightly modified version of [xml_to_csv.py](https://github.com/datitran/raccoon_dataset/blob/master/xml_to_csv.py)

With all exported data in `./images` simply run `python3 xml_to_csv.py`.

* Use `/research/object_detection/my_utils/generate_tfrecord.py` which is same as [generate_tfrecord.py](https://github.com/datitran/raccoon_dataset/blob/master/generate_tfrecord.py) to generate the record files.

```
python3 generate_tfrecord.py --csv_input=data/train_labels.csv --output_path=data/train.record

python3 generate_tfrecord.py --csv_input=data/test_labels.csv --output_path=data/test.record
```

Move the `*.record` files to `/research/object_detection/data`

The `/research/object_detection/data/my_labels.txt` file has the label data.

### 4. Download and configure the official model for transfer learning by fine tuning

* Download [faster_rcnn_resnet50_coco_2018_01_28](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz) through the official tensorflow distributions.

* Extract it to `/research/object_detection/`

* The `/research/object_detection/samples/configs/faster_rcnn_resnet50_coco.config` will be used to configure the training. In this repository it is already configured for our use.

Explanation of the modifications

* **Line 9**: Number of classes
* **Line 106**: Path to the pretrained checkpoint.
* **Line 121 and Line 135**: Paths to the `train.record` and `test.record` files
* **Line 123 and Line 137**: Path to the labels of classes.
* **IMPORTANT**: Do not change `batch_size`.

### 5. Train the model

```python3 train.py --logtostderr --train_dir=training/ --pipeline_config_path=samples/configs/faster_rcnn_resnet50_coco.config```

### 6. Export the model after training is complete

Checkpoints are created periodically, make sure to use the latest version.

**IMPORTANT**: When using `model.ckpt-???` make sure `*.index`, `*.data` and `*.meta` are present for that checkpoint. If not, use a different checkpoint.

```python3 export_inference_graph.py --input_type image_tensor --pipeline_config_path samples/configs/faster_rcnn_resnet50_coco.config --trained_checkpoint_prefix training/model.ckpt-379 --output_directory my_inference_graph```

### 7. Running the frozen graph in jupyter notebook

**IMPORTANT**: Make sure you run

```export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim``` 

before

`jupyter notebook`

## External links

* [Raw data](https://www.gsa.gov/real-estate/real-estate-services/leasing-policy-procedures/lease-documents)
* [Labled data with the .record files](https://github.com/akashchandwani/training_data)
* [faster_rcnn_resnet50_coco_2018_01_28 official google download link](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz)

## Bibliography

1. [Making sense of Handwritten Sections in Scanned Documents using the Azure ML Package for Computer Vision and Azure Cognitive Services](https://www.microsoft.com/developerblog/2018/05/07/handwriting-detection-and-recognition-in-scanned-documents-using-azure-ml-package-computer-vision-azure-cognitive-services-ocr/)

2. [TensorFlow Object Detection API: Youtube playlist by sentdex](https://www.youtube.com/playlist?list=PLQVvvaa0QuDcNK5GeCQnxYnSSaar2tpku)

3. [How to train your own Object Detector with TensorFlow's Object Detector API](https://towardsdatascience.com/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9)