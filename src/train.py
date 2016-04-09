import tensorflow as tf
import numpy as np
import pandas as pd

from detector import Detector
from util import load_image
import os
import ipdb

dataset_path = '../data/JPEGImages'
trainset_path = '../data/ImageSplits/train.txt'
testset_path = '../data/ImageSplits/test.txt'
weight_path = '../data/caffe_layers_value.pickle'
model_path = '../models'
pretrained_model_path = '../models/model-0'
n_epochs = 10000
learning_rate = 0.002
weight_decay_rate = 0.0005
momentum = 0.9
batch_size = 40

train_list = pd.read_csv( trainset_path, header=None )[0].values
test_list = pd.read_csv( testset_path, header=None)[0].values
labels = np.unique(map(lambda x: '_'.join(x.split('_')[:-1]), train_list))
n_labels = len(labels)
label_dict = pd.Series( range(n_labels), index=labels)

images_tf = tf.placeholder( tf.float32, [None, 224, 224, 3], name="images")
labels_tf = tf.placeholder( tf.int64, [None], name='labels')

detector = Detector(weight_path, n_labels)

conv5, conv6, gap, output = detector.inference(images_tf)
loss_tf = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits( output, labels_tf ))
weight_decay = tf.reduce_sum(tf.pack([tf.nn.l2_loss(x) for x in tf.trainable_variables()])) * weight_decay_rate
loss_tf += weight_decay

sess = tf.InteractiveSession()
saver = tf.train.Saver( max_to_keep=50 )

optimizer = tf.train.MomentumOptimizer( learning_rate, momentum )
grads_and_vars = optimizer.compute_gradients( loss_tf )
grads_and_vars = [(tf.clip_by_value(gv[0], -5., 5.), gv[1]) for gv in grads_and_vars]
train_op = optimizer.apply_gradients( grads_and_vars )
tf.initialize_all_variables().run()

if pretrained_model_path:
    print "Pretrained"
    saver.restore(sess, pretrained_model_path)

dataset = pd.DataFrame({'image_path':train_list })
dataset['label_name'] = dataset['image_path'].map(lambda x: '_'.join(x.split('_')[:-1]))
dataset['label'] = dataset['label_name'].map(label_dict)
dataset['image_path'] = dataset['image_path'].map(lambda x: os.path.join( dataset_path, x))
dataset.index = range(len(dataset))
dataset = dataset.ix[ np.random.permutation( len(dataset) ) ]

testset = pd.DataFrame({'image_path':test_list })
testset['label_name'] = testset['image_path'].map(lambda x: '_'.join(x.split('_')[:-1]))
testset['label'] = testset['label_name'].map(label_dict)
testset['image_path'] = testset['image_path'].map(lambda x: os.path.join( dataset_path, x))
testset.index = range(len(testset))
testset = testset.ix[ np.random.permutation( len(testset) ) ][:200]

f_log = open('../results/log.txt', 'w')

iterations = 0
for epoch in range(n_epochs):
    for start, end in zip(
        range( 0, len(dataset)+batch_size, batch_size),
        range(batch_size, len(dataset)+batch_size, batch_size)):

        current_data = dataset[start:end]
        current_image_paths = current_data['image_path'].values
        current_images = np.array(map(lambda x: load_image(x), current_image_paths))

        good_index = np.array(map(lambda x: x is not None, current_images))

        current_data = current_data[good_index]
        current_images = np.stack(current_images[good_index])
        current_labels = current_data['label'].values

        _, loss_val = sess.run(
                [train_op, loss_tf],
                feed_dict={
                    images_tf: current_images,
                    labels_tf: current_labels
                    })

#        conv5_val, conv6_val, gap_val, output_val = sess.run(
#                [conv5, conv6, gap, output],
#                feed_dict={
#                    images_tf: current_images
#                    })

        iterations += 1
        if iterations % 10 == 0:
            print loss_val

    n_correct = 0
    n_data = 0
    for start, end in zip(
            range(0, len(testset)+batch_size, batch_size),
            range(batch_size, len(testset)+batch_size, batch_size)
            ):
        current_data = testset[start:end]
        current_image_paths = current_data['image_path'].values
        current_images = np.array(map(lambda x: load_image(x), current_image_paths))

        good_index = np.array(map(lambda x: x is not None, current_images))

        current_data = current_data[good_index]
        current_images = np.stack(current_images[good_index])
        current_labels = current_data['label'].values

        output_vals = sess.run(
                output,
                feed_dict={images_tf:current_images})

        label_predictions = output_vals.argmax(axis=1)
        acc = (label_predictions == current_labels).sum()

        n_correct += acc
        n_data += len(current_data)

    acc_all = n_correct / float(n_data)
    f_log.write('epoch:'+str(epoch)+'\tacc:'+str(acc_all) + '\n')
    print 'epoch:'+str(epoch)+'\tacc:'+str(acc_all) + '\n'

    saver.save( sess, os.path.join( model_path, 'model'), global_step=epoch)




