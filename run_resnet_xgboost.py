import numpy as np
import dicom
import pandas as pd
from models.resnet50 import ResNet50
from keras.preprocessing import image
from models.imagenet_utils import preprocess_input, decode_predictions
#import xgboost as xgb
import os
import cv2
import keras.backend as K
from keras.models import Model
import time

K.set_image_dim_ordering('th')


def get_3d_data(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    return np.stack([s.pixel_array for s in slices])


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
            print("Shape: ", img.shape)
            print(img.sum(axis=0), img.sum(axis=1))
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

if __name__ == '__main__':
    #folder = "/home/andre/kaggle-dsb-2017/data/sample_images/0a0c32c9e08cc2ea76a71649de56be6d"
    folder = "/Users/Anuar_the_great/desktop/machine_learning/dsb/sample_images/0a0c32c9e08cc2ea76a71649de56be6d"
    batch = get_data_id(folder)

    model = ResNet50(weights='imagenet')


    img = np.ndarray([1,3,224,224],dtype=np.float32)
    img[0] = batch[0]
    x = preprocess_input(img)
    layer_name = 'b1'
    intermediate_layer_model = Model(input=model.input,
                                     output=model.get_layer(layer_name).output)
    t = time.time()
    intermediate_output = intermediate_layer_model.predict(x)
    print("TIME ELAPSED FOR FORWARD PASS: ", time.time() - t)
    flat_output = intermediate_output.flatten()
    print(flat_output, len(flat_output))
    #print(flat_output)
    #img_path = 'elephant.jpg'
    #img = image.load_img(img_path, target_size=(224, 224))
    #x = image.img_to_array(img)
    #x = np.expand_dims(x, axis=0)
    #x = preprocess_input(x)

    # preds = model.predict(x)
    # print('Predicted:', decode_predictions(preds))
