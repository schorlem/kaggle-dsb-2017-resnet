import numpy as np
import pydicom
from models.resnet50 import ResNet50
import os
import cv2
import keras.backend as K
from keras.models import Model

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
    # f, plots = plt.subplots(4, 5, sharex='col', sharey='row', figsize=(10, 8))

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

        # if cnt < 20:
        #     plots[cnt // 5, cnt % 5].axis('off')
        #     plots[cnt // 5, cnt % 5].imshow(np.swapaxes(tmp, 0, 2))
        # cnt += 1

    # plt.show()
    batch = np.array(batch)
    return batch


def calc_features(input_path, output_path):
    for folder in glob.glob(input_path+'*'): #TODO figure this function out
        batch = get_data_id(folder)
        feats = net.predict(batch)
        print(feats.shape)
        np.save(folder, feats)

    batch = get_data_id(folder)

    model = ResNet50(weights='imagenet')


    img = np.ndarray([1,3,224,224],dtype=np.float32)
    img[0] = batch[0]
    x = img
    layer_name = 'avg_pool'
    intermediate_layer_model = Model(input=model.input,
                                     output=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(x)
    flat_output = intermediate_output.flatten()


if __name__ == '__main__':
    input_folder = "/home/andre/kaggle-dsb-2017/data/sample_images/"
    output_folder = "/home/andre/kaggle-dsb-2017/data/resnet_features/"
    calc_features(input_folder, output_folder)

