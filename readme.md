Tensorflow implementation of RPN(Region Proposal Networks) for single class object detection. The setting of this project refer to [this paper](https://link.springer.com/chapter/10.1007/978-3-319-46475-6_28) so that it can be used for pedestrian detection(other tasks need to change parametres).

# Installtion
1. clone this repository
2. [install Tensorflow](https://www.tensorflow.org/install/)

# Datasets
The current version only supports Pascal VOC format. It means you must provide images and corresponding labels(.xml). 

Your directory structure should be similar to :
```
|-- DATASET_DIR
    |-- annotations
    |-- images
```

Then you can use ```tf_convert_data.py``` script to convert data to TF-Records:
```
python tf_convert_data.py \
    --dataset_dir=/path/to/your/DATASET_DIR \
    --output_dir=/paht/to/your/OUTPUT_DIR
```

Both training data and test data can be get in this way.

# Training
The script train_ssd_network.py is in charged of training the network. This project supports fine-tune with pre-trained VGG-16 model. You can download it from [here](https://github.com/tensorflow/models/tree/master/research/slim). Then you can begin your train with ```train_rpn.py``` script:
```
python train_rpn.py \
    --train_dir=/path/where/to/write/training/log \
    --dataset_dir=/path/to/DATASET_DIR \
    --model_path=/path/to/vgg_16.ckpt
```
It model_path is set to ```None```, the project will not use any pre-trained model and begin to train from random initialization.