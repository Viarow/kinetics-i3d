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

    """Rescale pixel values of each channel to range (-1, 1)"""

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


def read_single_image(path, image_scale, input_size):

    """ Read a single image and apply composed transforms to the image.
        Return n-dimensional numpy array with shape=(224, 224, 3) and dtype=float """

    image = io.imread(path)
    img_resize = Resize(image, image_scale)
    img_rescale = Rescale(img_resize)
    img_crop = CenterCrop(img_rescale, input_size)

    return img_crop


class Sampler(object):

    def __init__(self, num_segments, new_length=32, new_step=2, temporal_jitter=False):
        self.num_segments = num_segments
        self.new_length = new_length
        self.new_step = new_step
        self.temporal_jitter = temporal_jitter
        self.old_length = self.new_length * self.new_step
        self.image_scale = 256
        self.input_size = 224


    def __call__(self, dir_path):
        record = os.listdir(dir_path)
        record.sort()
        num_frames = len(record)
        seg_indices, skip_offsets = self.sample_indices(num_frames)
        output = []
        for seg_ind in seg_indices:
            p = int(seg_ind)
            for i, ind in enumerate(range(0, self.old_length, self.new_step)):
                if p + skip_offsets[i]  < num_frames:
                    frame_ind = p + skip_offsets[i]
                else:
                    frame_ind = p
                frame = read_single_image(os.path.join(dir_path, record[frame_ind]),
                                          self.image_scale, self.input_size)
                output.append(frame)
                if p + self.new_step < num_frames:
                    p += self.new_step

        output = np.asarray(output)
        output = np.expand_dims(output, 0)

        return output



    def sample_indices(self, num_frames):

        """Sample frame indices.
               Return a list of offsets and a lost of skip_offsets"""

        average_duration = (num_frames - self.old_length + 1) // self.num_segments

        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration)
            offsets = offsets + np.random.randint(average_duration, size=self.num_segments)

        elif num_frames > max(self.num_segments, self.old_length):
            offsets = np.sort(np.random.randint(num_frames - self.old_length + 1,
                                                size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))

        if self.temporal_jitter:
            skip_offsets = np.random.randint(self.new_step,
                                             size=self.old_length // self.new_step)
        else:
            skip_offsets = np.zeros(self.old_length // self.new_step, dtype=int)

        return offsets + 1, skip_offsets  # frame index starts from 1


def load_data(dir_path):

    sampler = Sampler(num_segments=10, new_length=32, new_step=2, temporal_jitter=False)
    output = sampler(dir_path)

    return output


if __name__ == '__main__':
    load_data('/media/Med_6T2/mmaction/data_tools/kinetics400/rawframes_val/cleaning_shoes/-5I-VI9TP5k_000181_000191/')









