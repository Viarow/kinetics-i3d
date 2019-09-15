from data.common_classes import Kinetics400_UCF101
from data.common_classes import Kinetics400_HMDB51
import os

_UCF101_ROOT_ = '/media/Med_6T2/mmaction/data_tools/ucf101/rawframes/'

def ucf_filelist(target_root, vid_num, filelist_path):

    listfile = open(filelist_path, 'w')

    for (ucf_name, index) in Kinetics400_UCF101:
        videos = os.listdir(os.path.join(target_root, ucf_name))
        #videos = videos.sort()
        if len(videos) > vid_num:
            videos = videos[0:vid_num]
        else:
            print("{:d} videos in ".format(len(videos)) + ucf_name +'\n')

        for v in videos:
            listfile.writelines(ucf_name + '/' + v + ' ' + "{:d}".format(index) + '\n')
        print("Generating video lists for " + ucf_name + '\n')


_HMDB51_ROOT_ = '/media/Med_6T2/mmaction/data_tools/hmdb51/rawframes/'

def hmdb_filelist(target_root, vid_num, filelist_path):

    listfile = open(filelist_path, 'w')

    for(hmdb_name, index) in Kinetics400_HMDB51:
        videos = os.listdir(os.path.join(target_root, hmdb_name))
        if len(videos) > vid_num:
            videos = videos[0:vid_num]
        else:
            print("{:d} videos in ".format(len(videos)) + hmdb_name + '\n')

        for v in videos:
            listfile.writelines(hmdb_name + '/' + v + ' ' + "{:d}".format(index) +'\n')
        print("Generating video lists for " + hmdb_name + '\n')



_Kinetics400_ROOT_ = '/media/Med_6T2/mmaction/data_tools/kinetics400/rawframes_val/'
_Kinetics400_ann_ = '/media/Med_6T2/mmaction/data_tools/kinetics400/annotations/classInd.txt'


def kinetics_filelist(origin_root, vid_num, filelist_path, target_dataset):

    assert target_dataset in ['ucf101', 'hmdb51', 'moments_in_time']

    listfile = open(filelist_path, 'w')
    classInd = open(_Kinetics400_ann_, 'r')
    kinetics_classes = classInd.readlines()

    if target_dataset == 'ucf101':

        for (ucf_name, idx) in Kinetics400_UCF101:
            kinetics_name = kinetics_classes[idx].split('\n')[0]

            videos = os.listdir(os.path.join(origin_root, kinetics_name))
            #videos = videos.sort()

            if len(videos) > vid_num:
                videos = videos[0:vid_num]
            else:
                print("{:d} videos in ".format(len(videos)) + kinetics_name +'\n')

            for v in videos:
                listfile.writelines(kinetics_name + '/' + v + ' ' + "{:d}".format(idx) + '\n')
            print("Generating video lists for " + kinetics_name + '\n')
    elif target_dataset == 'hmdb51':

        for (hmdb_name, idx) in Kinetics400_HMDB51:
            kinetics_name = kinetics_classes[idx].split('\n')[0]

            videos = os.listdir(os.path.join(origin_root, kinetics_name))
            #videos = videos.sort()

            if len(videos) > vid_num:
                videos = videos[0:vid_num]
            else:
                print("{:d} videos in ".format(len(videos)) + kinetics_name +'\n')

            for v in videos:
                listfile.writelines(kinetics_name + '/' + v + ' ' + "{:d}".format(idx) + '\n')
            print("Generating video lists for " + kinetics_name + '\n')




def generate_filelist():

    ucf_vid_num = 100
    ucf_filelist(_UCF101_ROOT_, ucf_vid_num, 'ucf101_testlist.txt')

    hmdb_vid_num = 100
    hmdb_filelist(_HMDB51_ROOT_, hmdb_vid_num, 'hmdb51_testlist.txt')

    kinetics_vid_num = 40
    kinetics_filelist(_Kinetics400_ROOT_, kinetics_vid_num, 'kinetics_ucf_testlist.txt', 'ucf101')
    kinetics_filelist(_Kinetics400_ROOT_, kinetics_vid_num, 'kinetics_hmdb_testlist.txt', 'hmdb51')


if __name__ == '__main__':
    generate_filelist()








