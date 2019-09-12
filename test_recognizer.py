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

_IMAGE_SIZE = 224

_SAMPLE_ROOT_ = '~/kinetics-i3d/data/test_data'
_ANN_PATH_ = '/media/Med_6T2/mmaction/data_tools/kinetics400/annotations/classInd.txt'

_SAMPLE_VIDEO_FRAMES = 80
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


def get_label_dict(label_path):
    ann_file = open(label_path, 'r')
    classes = ann_file.readlines()

    label_dict = {}
    for idx,name in enumerate(classes):
        name = name.split('\n')[0]
        label_dict.update({name:idx})

    return  label_dict


def get_sample_pool(root_path):

    """ Require the directory is organized into 2 levels"""
    sample_pool = []
    all_classes = os.listdir(root_path)
    all_classes.sort()

    for c_item in all_classes:
        videos = os.listdir(os.path.join(root_path, c_item))
        for v_item in videos:
            sample_pool.append(os.path.join(root_path, c_item, v_item))

    return sample_pool


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

    label_dict = get_label_dict(_ANN_PATH_)

    with tf.Session() as sess:

        feed_dict = {}
        if eval_type in ['rgb', 'rgb600', 'joint']:
            if imagenet_pretrained:
                rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb_imagenet'])
            else:
                rgb_saver.restore(sess, _CHECKPOINT_PATHS[eval_type])
            tf.logging.info('RGB checkpoint restored')

            sample_pool = get_sample_pool(_SAMPLE_ROOT_)
            gt_labels = []
            results = []

            for vid in tqdm(sample_pool):
                rgb_sample = load_data(vid)
                feed_dict[rgb_input] = rgb_sample

                out_logits, out_predictions = sess.run(
                    [model_logits, model_predictions],
                    feed_dict=feed_dict)

                out_predictions = out_predictions[0]
                class_name = vid.split('/')[-2]
                gt_labels.append(label_dict[class_name])
                results.append(out_predictions)

        top1, top5 = top_k_accuracy(results, gt_labels, k=(1, 5))
        print("Top-1 Accuracy = {:.02f}".format(top1 * 100))
        print("Top-5 Accuracy = {:.02f}".format(top5 * 100))


if __name__ == '__main__':
  tf.app.run(main)








