from skimage import io, transform
import numpy as np
import os
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler


def Resize(img, output_size):

    assert isinstance(output_size, (int, tuple))
    h, w = img.shape[:2]
    if isinstance(output_size, int):
        if h > w:
            new_h, new_w = output_size * h / w, output_size
        else:
            new_h, new_w = output_size, output_size * w / h
    else:
        new_h, new_w = output_size

    new_h, new_w = int(new_h), int(new_w)
    img = transform.resize(img, (new_h, new_w))

    return img


def Rescale(pixels):

    pixels = np.transpose(pixels, (2, 0, 1))
    channel_num = pixels.shape[0]
    for idx in range(0, channel_num):
        channel = pixels[idx]
        scaler = MinMaxScaler((-1, 1))
        scaler.fit(channel)
        pixels[idx] = scaler.transform(channel)
    pixels = np.transpose(pixels, (1, 2, 0))
    
    return pixels


def CenterCrop(img, output_size):

    assert isinstance(output_size, (int, tuple))
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    else:
        assert len(output_size) == 2

    h, w = img.shape[:2]
    new_h, new_w = output_size

    top = int((h - new_h) / 2)
    left = int((w - new_w) / 2)

    img = img[top: top + new_h,
              left: left + new_w]

    return img


def preProcess(vid_dir):

    """ Pre-process raw frames from a single video
        Return a ndarray in shape [1, num_frames, 3, 244, 244]"""

    frames = os.listdir(vid_dir)
    frames.sort()
    #output= np.ndarray([], dtype=float)
    output = []

    for frame_id in tqdm(frames):
        image = io.imread(os.path.join(vid_dir, frame_id))
        img_resize = Resize(image, 256)
        img_rescale = Rescale(img_resize)
        img_crop = CenterCrop(img_rescale, 224)
        output.append(img_crop)
        
    output = output[0:100]
    output = np.asarray(output)
    output = np.expand_dims(output, 0)
    print(output.shape)
    return output


def main():
    vid_dir = '/media/Med_6T2/mmaction/data_tools/kinetics400/rawframes_val/cleaning_shoes/-5I-VI9TP5k_000181_000191/'
    output = preProcess(vid_dir)
    np.save('test_data.npy', output)

if __name__ == '__main__':
    main()