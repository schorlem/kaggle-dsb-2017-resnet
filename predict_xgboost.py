import pickle
import pandas as pd
import numpy as np


def create_submission(input_directory, output_directory, submission_file, model_file):
    clf = pickle.load(open(model_file, "rb"))

    df = pd.read_csv(submission_file)

    patients = []
    for pid in df['id'].tolist():
        patient = np.load(input_directory+'%s_features.npy' % str(pid))
        patient = patient.reshape(patient.shape[0], patient.shape[1])
        patient = np.mean(patient, axis=0)
        patients.append(patient)

    x = np.array(patients)

    pred = 1-clf.predict_proba(x)

    df['cancer'] = pred
    df.to_csv(output_directory+"submission.csv", index=False)
    print(df.head())


if __name__ == '__main__':
    input_dir = "/path_to_project/data/stage2_resnet_features/"
    output_dir = "/path_to_project/data/stage2_resnet_features/"
    submission_data = "/path_to_project/data/stage2_sample_submission.csv"
    model_data = "/path_to_project/data/resnet_features/pima.pickle.dat"
    create_submission(input_dir, output_dir, submission_data, model_data)
