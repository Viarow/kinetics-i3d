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

_SAMPLE_ROOT_ = '/media/Med_6T2/mmaction/data_tools/kinetics400/rawframes_val/cleaning_shoes'

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


def plot_results(results, ap_num, full_range, grad, variable, class_id):
    colors = ['lightcoral', 'turquoise', 'yellowgreen', 'slateblue']
    fig, ax = plt.subplots()
    ax.set(xlabel=variable.lower(), ylabel='softmax score',
        title='Controlled Variable: '+variable)
    vid_num = len(results)
    assert (vid_num // ap_num) == (full_range // grad)
    point_num = vid_num // ap_num
    x = [x_idx*grad for x_idx in range(0, point_num)]
    label_list = []
    for ap_idx in range(0, ap_num):
        scores = results[ap_idx*point_num : (ap_idx+1)*point_num]
        y = [scores[i][class_id] for i in range(0, point_num)]
        plt.plot(x, y, marker='o', color=colors[ap_idx],
                 linewidth=2, markersize=6)
        label_list.append('appearance_{:1d}'.format(ap_idx+1))
    plt.legend(label_list)
    ax.grid()
    fig.savefig(variable+'.png')
    plt.show()


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

    with tf.Session() as sess:

        feed_dict = {}
        if eval_type in ['rgb', 'rgb600', 'joint']:
            if imagenet_pretrained:
                rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb_imagenet'])
            else:
                rgb_saver.restore(sess, _CHECKPOINT_PATHS[eval_type])
            tf.logging.info('RGB checkpoint restored')

            sample_pool = os.listdir(_SAMPLE_ROOT_)
            sample_pool.sort()

            results = []

            for vid_name in tqdm(sample_pool):
                rgb_sample = load_data(os.path.join(_SAMPLE_ROOT_, vid_name))
                #tf.logging.info('RGB data loaded, shape=%s', str(rgb_sample.shape))
                feed_dict[rgb_input] = rgb_sample

                # if eval_type in ['flow', 'joint']:
                #     if imagenet_pretrained:
                #         flow_saver.restore(sess, _CHECKPOINT_PATHS['flow_imagenet'])
                #     else:
                #         flow_saver.restore(sess, _CHECKPOINT_PATHS['flow'])
                #     tf.logging.info('Flow checkpoint restored')
                #     flow_sample = np.load(_SAMPLE_PATHS['flow'])
                #     tf.logging.info('Flow data loaded, shape=%s', str(flow_sample.shape))
                #     feed_dict[flow_input] = flow_sample

                out_logits, out_predictions = sess.run(
                    [model_logits, model_predictions],
                    feed_dict=feed_dict)

                out_logits = out_logits[0]
                out_predictions = out_predictions[0]

                results.append(out_predictions)

        plot_results(results, ap_num=4, full_range=360, grad=15, variable='ForTest', class_id=260)
        gt_labels = [260]*len(results)
        top1, top5 = top_k_accuracy(results, gt_labels, k=(1, 5))
        print("Top-1 Accuracy = {:.02f}".format(top1 * 100))
        print("Top-5 Accuracy = {:.02f}".format(top5 * 100))


if __name__ == '__main__':
  tf.app.run(main)

