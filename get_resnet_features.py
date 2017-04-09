import os
from glob import glob
import numpy as np
import pydicom
import cv2
import keras.backend as K
from keras.models import Model
from models.resnet50 import ResNet50

K.set_image_dim_ordering('th')


def get_3d_data(path):
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    return np.stack([s.pixel_array for s in slices])


def zero_centering(batch):
    pass #TODO add zero centering for all three channels


def get_data_id(path):
    sample_image = get_3d_data(path)
    sample_image[sample_image == -2000] = 0

    batch = []
    cnt = 0
    dx = 40
    ds = 512

    for i in range(0, sample_image.shape[0] - 3, 3):
        tmp = []

        for j in range(3):
            img = sample_image[i + j]
            img = 255.0 / np.amax(img) * img
            img = cv2.equalizeHist(img.astype(np.uint8))
            img = img[dx: ds - dx, dx: ds - dx]
            img = cv2.resize(img, (224, 224))
            tmp.append(img)

        tmp = np.array(tmp)
        batch.append(np.array(tmp))

    batch = np.array(batch)
    return batch


def calc_features(input_path, output_path, n_iterations=100, overwrite=True):
    model = ResNet50(weights='imagenet')
    layer_name = 'avg_pool'
    intermediate_layer_model = Model(input=model.input, output=model.get_layer(layer_name).output)
    i = 0

    for folder in glob(input_path+'*'):
        print(i)

        if i >= n_iterations:
            break

        output_basename = os.path.basename(os.path.normpath(folder))
        output_name = output_path + output_basename + "_features"

        if not overwrite and os.path.exists(output_name + ".npy"):
            i += 1
            continue

        batch = get_data_id(folder)
        img = np.ndarray([len(batch),3,224,224],dtype=np.float32)
        img = batch
        intermediate_output = intermediate_layer_model.predict(img, batch_size = 20)
        np.save(output_name, intermediate_output)
        i += 1


if __name__ == '__main__':
    #input_directory = "/media/andre/USB Drive/kaggle/stage1/"
    #output_directory = "/home/andre/kaggle-dsb-2017/data/test/"
    #input_directory = "/media/andre/USB Drive/kaggle/stage2/"
    #output_directory = "/home/andre/kaggle-dsb-2017/data/stage2_resnet_features/"
    calc_features(input_directory, output_directory, n_iterations=520, overwrite=False)

