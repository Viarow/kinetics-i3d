from skimage import io
from imgaug import augmenters as iaa
import argparse
from tqdm import tqdm
import os


aug_dict = {
    'AdditiveGaussianNoise': iaa.AdditiveGaussianNoise(loc=0, scale=0.05*255, per_channel=False),
    'AdditiveGaussianNoise_pc': iaa.AdditiveGaussianNoise(loc=0, scale=0.05*255, per_channel=True),
    'AdditiveLaplaceNoise': iaa.AdditiveLaplaceNoise(loc=0, scale=0.05*255, per_channel=False),
    'AdditiveLaplaceNoise_pc': iaa.AdditiveLaplaceNoise(loc=0, scale=0.05*255, per_channel=True),
    'AdditivePoissonNoise': iaa.AdditivePoissonNoise(lam=16.00, per_channel=False),
    'AdditivePoissonNoise_pc': iaa.AdditivePoissonNoise(lam=16.00, per_channel=True),
    'ImpulseNoise': iaa.ImpulseNoise(p=0.05),
    'SaltAndPepper': iaa.SaltAndPepper(p=0.05),
    'GaussianBlur': iaa.GaussianBlur(sigma=0.50),
    'AverageBlur': iaa.AverageBlur(k=3),
    'AddToHueAndSaturation_p': iaa.AddToHueAndSaturation(value=25),
    'AddToHueAndSaturation_n': iaa.AddToHueAndSaturation(value=-25),
    'Grayscale': iaa.Grayscale(alpha=1.0),
    'GammaContrast': iaa.GammaContrast(gamma=0.5, per_channel=False),
    'GammaContrast_pc': iaa.GammaContrast(gamma=0.5, per_channel=True),
    'HistogramEqualization': iaa.HistogramEqualization(to_colorspace='HSV'),
    'Rotate_p': iaa.Affine(rotate=25),
    'Rotate_n': iaa.Affine(rotate=-25)
}


def parse_args():
    parser = argparse.ArgumentParser(description='Simple transform using imgaug')
    parser.add_argument('old_root', help='Root path of the original data set.')
    parser.add_argument('new_root', help='Root path of the transformed data set.')
    parser.add_argument('testlist', help='List of videos to be augmented.')
    parser.add_argument('--augmenter', help='Choose one kind of augmenter.')
    args = parser.parse_args()
    return args


def simple_transform():
    args = parse_args()
    augmenter = aug_dict[args.augmenter]
    old_root = args.old_root
    new_root = args.new_root
    ann_file = open(args.testlist)
    testlist = ann_file.readlines()

    for v_item in tqdm(testlist):
        video_name = v_item.split(' ')[0]
        imgs_old_path = os.path.join(old_root, video_name)
        imgs_new_path = os.path.join(new_root, video_name)

        if not os.path.exists(imgs_new_path):
            os.makedirs(imgs_new_path)

        imgs = os.listdir(imgs_old_path)
        for f_item in imgs:
            image = io.imread(os.path.join(imgs_old_path, f_item))
            image_aug = augmenter.augment_image(image)
            io.imsave(os.path.join(imgs_new_path, f_item), image_aug)




if __name__ == '__main__':
    simple_transform()






