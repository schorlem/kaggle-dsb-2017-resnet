import numpy as np
import pydicom
from models.resnet50 import ResNet50
import os
import cv2
import keras.backend as K
from keras.models import Model
from glob import glob

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


def calc_features(input_path, output_path):
    model = ResNet50(weights='imagenet')
    layer_name = 'avg_pool'
    intermediate_layer_model = Model(input=model.input, output=model.get_layer(layer_name).output)

    for folder in glob(input_path+'*'):
        output_basename = os.path.basename(os.path.normpath(folder))
        output_folder = output_path + output_basename
        if not os.path.exists(output_folder): #TODO don't create new folder for each patient but add id to each filename
            os.makedirs(output_folder)

        batch = get_data_id(folder)


        img = np.ndarray([len(batch),3,224,224],dtype=np.float32)
        img = batch
        intermediate_output = intermediate_layer_model.predict(img)
        #print(output_basename, intermediate_output, intermediate_output.shape)
        np.save(output_folder + "/features.np", intermediate_output) #TODO add overwrite flag or some other method to enable batch processing of inputs

if __name__ == '__main__':
    input_directory = "/home/andre/kaggle-dsb-2017/data/sample_images/"
    output_directory = "/home/andre/kaggle-dsb-2017/data/resnet_features/"
    calc_features(input_directory, output_directory)

