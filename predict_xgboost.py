import pickle
import pandas as pd
import numpy as np

def create_submission(input_directory, output_directory, submission_file, model_file):
    clf = pickle.load(open(model_file, "rb"))

    df = pd.read_csv(submission_file)

    patients = []
    for id in df['id'].tolist():
        patient = np.load(input_directory+'%s_features.npy' % str(id))
        patient = patient.reshape(patient.shape[0],patient.shape[1])
        patient = np.mean(patient, axis = 0)
        patients.append(patient)

    x = np.array(patients)

    pred = clf.predict(x)

    df['cancer'] = pred
    df.to_csv(output_directory+"submission.csv", index=False)
    print(df.head())


if __name__ == '__main__':
    input_directory = "/home/andre/kaggle-dsb-2017/data/resnet_features/"
    output_directory = "/home/andre/kaggle-dsb-2017/data/resnet_features/"
    submission_file = "/home/andre/kaggle-dsb-2017/data/stage1_sample_submission.csv"
    model_file = "/home/andre/kaggle-dsb-2017/data/resnet_features/pima.pickle.dat"
    create_submission(input_directory, output_directory, submission_file, model_file)
