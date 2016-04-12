# Weakly_detector
Tensorflow implementation of "Learning Deep Features for Discriminative Localization"

B. Zhou, A. Khosla, A. Lapedriza, A. Oliva, and A. Torralba
Learning Deep Features for Discriminative Localization.
Computer Vision and Pattern Recognition (CVPR), 2016.
[[PDF](http://arxiv.org/pdf/1512.04150.pdf)][[Project Page](http://cnnlocalization.csail.mit.edu/)]

### Results of Caltech256 Dataset
![alt tag](https://github.com/jazzsaxmafia/Weakly_detector/blob/master/results/demo.main.jpg)

### Results of Action40 Dataset
![alt tag](https://github.com/jazzsaxmafia/Weakly_detector/blob/master/results/demo.main2.jpg)
Object localization using only image-level annotation, without bounding box annotation.

* If you want to train the model using custom dataset, you need the pretrained VGG Network weights [[VGG](https://drive.google.com/file/d/0B5o40yxdA9PqOVI5dF9tN3NUc2c/view?usp=sharing)], which is used in [[code](https://github.com/jazzsaxmafia/Weakly_detector/blob/master/src/train.caltech.py#L10)].

