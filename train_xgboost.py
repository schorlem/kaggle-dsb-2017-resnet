import pickle
import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split


def train_xgboost(input_folder, output_folder, labels):
    df = pd.read_csv(labels)
    print(df.head())

    patients = []
    for id in df['id'].tolist():
        patient = np.load(input_folder+'%s_features.npy' % str(id))
        patient = patient.reshape(patient.shape[0],patient.shape[1])
        patient = np.mean(patient, axis = 0)
        patients.append(patient)

    x = np.array(patients)
    y = df['cancer'].as_matrix()


    trn_x, val_x, trn_y, val_y = train_test_split(x, y, random_state=42, stratify=y, test_size=0.20)

    clf = xgb.XGBRegressor(max_depth=10,
                           n_estimators=1500,
                           min_child_weight=9,
                           learning_rate=0.01,
                           nthread=8,
                           subsample=0.80,
                           colsample_bytree=0.80,
                           seed=4242)

    clf.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], verbose=True, eval_metric='logloss', early_stopping_rounds=50)
    pickle.dump(clf, open(output_folder + "pima.pickle.dat", "wb"))
    return clf


if __name__ == '__main__':
    input_directory = "/home/andre/kaggle-dsb-2017/data/resnet_features/"
    output_directory = "/home/andre/kaggle-dsb-2017/data/resnet_features/"
    labels_file = "/home/andre/kaggle-dsb-2017/data/stage1_labels.csv"
    train_xgboost(input_directory, output_directory, labels_file)
