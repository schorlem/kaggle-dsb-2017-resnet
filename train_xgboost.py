import pickle
import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split


def train_xgboost(input_folder, output_folder, labels):
    df = pd.read_csv(labels)
    print(df.head())

    shapes= []
    patients = []
    for id in df['id'].tolist():
        patient = np.load(input_folder+'%s_features.npy' % str(id))
        patient = patient.reshape(patient.shape[0], patient.shape[1])
        #patient = patient.reshape(patient.shape[0]*patient.shape[1])
        patient = np.mean(patient, axis = 0)
        #print(patient.shape)
        shapes.append(patient.shape[0])
        patients.append(patient)
    #print(max(shapes), min(shapes), max(shapes)-min(shapes))

    #mean_patients = []
    #for patient in patients:
    #    mean_patient = []
    #    x = patient.shape[0] - min(shapes)
    #    n_half = x // 2
    #    n_above = n_half + x % 2
    #    n_below = patient.shape[0] - n_half
    #    if n_above+n_half > 1:
    #        mean_patient.extend(np.mean(patient[:n_above], axis=0))
    #        middle = patient[n_above:n_below]
    #        middle = middle.reshape(middle.shape[0]*middle.shape[1])
    #        mean_patient.extend(middle)
    #        mean_patient.extend(np.mean(patient[n_below:], axis=0))
    #    elif n_above+n_half == 0:
    #        mean_patient.extend(np.zeros(patient.shape[1]))
    #        middle = patient.reshape(patient.shape[0]*patient.shape[1])
    #        mean_patient.extend(middle)
    #        mean_patient.extend(np.zeros(patient.shape[1]))
    #    elif n_above+n_half == 1:
    #        middle = patient.reshape(patient.shape[0]*patient.shape[1])
    #        mean_patient.extend(middle)
    #        mean_patient.extend(np.zeros(patient.shape[1]))

    #    mean_patient = np.array(mean_patient)
    #    if mean_patient.shape[0] != 67584:
    #        print(mean_patient.shape)
    #        print(patient.shape)
    #        print(n_above, n_below)
    #    mean_patients.append(mean_patient)




    x = np.array(patients)
    #x = np.array(mean_patients)
    y = df['cancer'].as_matrix()


    trn_x, val_x, trn_y, val_y = train_test_split(x, y, random_state=42, stratify=y, test_size=0.20)

    clf = xgb.XGBRegressor(max_depth=4,
                           n_estimators=1500,
                           min_child_weight=2,
                           learning_rate=0.004,
                           nthread=4,
                           subsample=0.80,
                           colsample_bytree=0.80,
                           seed=4242)

    clf.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], verbose=True, eval_metric='logloss', early_stopping_rounds=50)

    clf2 = xgb.XGBRegressor(max_depth=4,
                           n_estimators=580,
                           min_child_weight=2,
                           learning_rate=0.004,
                           nthread=4,
                           subsample=0.80,
                           colsample_bytree=0.80,
                           seed=4242)

    #clf2.fit(x, y, verbose=True, eval_metric='logloss')

    xgb3 = xgb.XGBClassifier(learning_rate=0.005,
                         n_estimators=580,
                         max_depth=4,
                         min_child_weight=2,
                         gamma=0,
                         subsample=0.8,
                         colsample_bytree=0.8,
                         objective='binary:logistic',
                         nthread=4,
                         scale_pos_weight=1,
                         seed=4242)

    #xgb3.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], verbose=True, eval_metric='logloss', early_stopping_rounds=50)

    pickle.dump(clf, open(output_folder + "pima.pickle.dat", "wb"))
    return clf


if __name__ == '__main__':
    input_directory = "/home/andre/kaggle-dsb-2017/data/resnet_features/"
    output_directory = "/home/andre/kaggle-dsb-2017/data/resnet_features/"
    labels_file = "/home/andre/kaggle-dsb-2017/data/stage1_labels.csv"
    train_xgboost(input_directory, output_directory, labels_file)
