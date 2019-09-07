from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import i3d

from data.load_data import load_data
import os
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from accuracy import top_k_accuracy
import csv

_IMAGE_SIZE = 224

_SAMPLE_VIDEO_FRAMES = 80

_SAMPLE_ROOT_ = '/media/Med_6T2/mmaction/data_tools/ucf101/rawframes'

_TESTLIST_PATH_ = './data/ucf101_testlist.txt'

_VIDS_PER_CLASS_ = 100


_SAMPLE_PATHS = {
    'rgb': 'data/v_CricketShot_g04_c01_rgb.npy',
    'flow': 'data/v_CricketShot_g04_c01_flow.npy',
}

_CHECKPOINT_PATHS = {
    'rgb': 'data/checkpoints/rgb_scratch/model.ckpt',
    'rgb600': 'data/checkpoints/rgb_scratch_kin600/model.ckpt',
    'flow': 'data/checkpoints/flow_scratch/model.ckpt',
    'rgb_imagenet': 'data/checkpoints/rgb_imagenet/model.ckpt',
    'flow_imagenet': 'data/checkpoints/flow_imagenet/model.ckpt',
}

_LABEL_MAP_PATH = 'data/label_map.txt'
_LABEL_MAP_PATH_600 = 'data/label_map_600.txt'

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('eval_type', 'joint', 'rgb, rgb600, flow, or joint')
tf.flags.DEFINE_boolean('imagenet_pretrained', True, '')

def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)
    eval_type = FLAGS.eval_type

    imagenet_pretrained = FLAGS.imagenet_pretrained

    NUM_CLASSES = 400
    if eval_type == 'rgb600':
        NUM_CLASSES = 600

    if eval_type not in ['rgb', 'rgb600', 'flow', 'joint']:
        raise ValueError('Bad `eval_type`, must be one of rgb, rgb600, flow, joint')

    if eval_type == 'rgb600':
        kinetics_classes = [x.strip() for x in open(_LABEL_MAP_PATH_600)]
    else:
        kinetics_classes = [x.strip() for x in open(_LABEL_MAP_PATH)]

    if eval_type in ['rgb', 'rgb600', 'joint']:
        # RGB input has 3 channels.
        rgb_input = tf.placeholder(
            tf.float32,
            shape=(1, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3))

        with tf.variable_scope('RGB'):
            rgb_model = i3d.InceptionI3d(
                NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
            rgb_logits, _ = rgb_model(
                rgb_input, is_training=False, dropout_keep_prob=1.0)

        rgb_variable_map = {}
        for variable in tf.global_variables():

            if variable.name.split('/')[0] == 'RGB':
                if eval_type == 'rgb600':
                    rgb_variable_map[variable.name.replace(':0', '')[len('RGB/inception_i3d/'):]] = variable
                else:
                    rgb_variable_map[variable.name.replace(':0', '')] = variable

        rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)

    if eval_type in ['flow', 'joint']:
        # Flow input has only 2 channels.
        flow_input = tf.placeholder(
            tf.float32,
            shape=(1, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 2))
        with tf.variable_scope('Flow'):
            flow_model = i3d.InceptionI3d(
                NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
            flow_logits, _ = flow_model(
                flow_input, is_training=False, dropout_keep_prob=1.0)
        flow_variable_map = {}
        for variable in tf.global_variables():
            if variable.name.split('/')[0] == 'Flow':
                flow_variable_map[variable.name.replace(':0', '')] = variable
        flow_saver = tf.train.Saver(var_list=flow_variable_map, reshape=True)

    if eval_type == 'rgb' or eval_type == 'rgb600':
        model_logits = rgb_logits
    elif eval_type == 'flow':
        model_logits = flow_logits
    else:
        model_logits = rgb_logits + flow_logits
    model_predictions = tf.nn.softmax(model_logits)

    csvFile = open('transfer_ucf101.csv', 'a')
    writer = csv.writer(csvFile)
    writer.writerow(['Actnames', 'Top-1 accuracy', 'Top-5 accuracy'])

    with tf.Session() as sess:

        feed_dict = {}
        if eval_type in ['rgb', 'rgb600', 'joint']:
            if imagenet_pretrained:
                rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb_imagenet'])
            else:
                rgb_saver.restore(sess, _CHECKPOINT_PATHS[eval_type])
            tf.logging.info('RGB checkpoint restored')

            listfile = open(_TESTLIST_PATH_, 'r')
            testlist = listfile.readlines()
            actnames = []

            gt_labels = []
            results = []

            for item in tqdm(testlist):

                sample = item.split('\n')[0]
                video_name = sample.split(' ')[0]
                video_label = int(sample.split(' ')[1])
                class_name = video_name.split('/')[0]
                gt_labels.append(video_label)
                actnames.append(class_name)

                rgb_sample = load_data(os.path.join(_SAMPLE_ROOT_, video_name))
                #tf.logging.info('RGB data loaded, shape=%s', str(rgb_sample.shape))
                feed_dict[rgb_input] = rgb_sample

                out_logits, out_predictions = sess.run(
                    [model_logits, model_predictions],
                    feed_dict=feed_dict)

                #out_logits = out_logits[0]
                out_predictions = out_predictions[0]

                results.append(out_predictions)

            assert len(results) % _VIDS_PER_CLASS_ == 0
            class_num = len(results) // _VIDS_PER_CLASS_

            for idx in range(0, class_num):

                 preds = results[idx*_VIDS_PER_CLASS_ : (idx+1)*_VIDS_PER_CLASS_]
                 labels = gt_labels[idx*_VIDS_PER_CLASS_ : (idx+1)*_VIDS_PER_CLASS_]

                 top1, top5 = top_k_accuracy(preds, labels, k=(1, 5))
                 row = [actnames[idx*_VIDS_PER_CLASS_],
                   "{:.4f}".format(top1),  "{:.4f}".format(top5)]
                 writer.writerow(row)

    csvFile.close()



if __name__ == '__main__':
  tf.app.run(main)










