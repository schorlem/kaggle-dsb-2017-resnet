import pickle
import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split


def train_xgboost(input_folder, output_folder, labels):
    df = pd.read_csv(labels)
    print(df.head())

    patients = []
    for pid in df['id'].tolist():
        patient = np.load(input_folder+'%s_features.npy' % str(pid))
        patient = patient.reshape(patient.shape[0], patient.shape[1])
        patient = np.mean(patient, axis=0)
        patients.append(patient)

    x = np.array(patients)
    y = df['cancer'].as_matrix()

    trn_x, val_x, trn_y, val_y = train_test_split(x, y, random_state=42, stratify=y, test_size=0.20)

    clf = xgb.XGBClassifier(learning_rate=0.005,
                            n_estimators=403,
                            max_depth=4,
                            min_child_weight=2,
                            gamma=0,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            objective='binary:logistic',
                            nthread=4,
                            scale_pos_weight=1,
                            seed=500)

    # Fit with early stopping rounds to find best number of estimators.
    # Then change it above and run again without stopping rounds.
    clf.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], verbose=True, eval_metric='logloss', early_stopping_rounds=50)
    clf.fit(x, y, verbose=True, eval_metric='logloss')

    pickle.dump(clf, open(output_folder + "pima.pickle.dat", "wb"))
    return clf


if __name__ == '__main__':
    input_directory = "path_to_project/data/resnet_features/"
    output_directory = "path_to_project/data/resnet_features/"
    labels_file = "path_to_project/data/all_labels.csv"
    train_xgboost(input_directory, output_directory, labels_file)
