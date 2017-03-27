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

#    shapes= []
#    patients = []
#    for id in df['id'].tolist():
#        patient = np.load(input_directory+'%s_features.npy' % str(id))
#        patient = patient.reshape(patient.shape[0], patient.shape[1])
#        shapes.append(patient.shape[0])
#        patients.append(patient)
#
#    mean_patients = []
#    for patient in patients:
#        mean_patient = []
#        x = patient.shape[0] - 31
#        n_half = x // 2
#        n_above = n_half + x % 2
#        n_below = patient.shape[0] - n_half
#        if n_above+n_half > 1:
#            mean_patient.extend(np.mean(patient[:n_above], axis=0))
#            middle = patient[n_above:n_below]
#            middle = middle.reshape(middle.shape[0]*middle.shape[1])
#            mean_patient.extend(middle)
#            mean_patient.extend(np.mean(patient[n_below:], axis=0))
#        elif n_above+n_half == 0:
#            mean_patient.extend(np.zeros(patient.shape[1]))
#            middle = patient.reshape(patient.shape[0]*patient.shape[1])
#            mean_patient.extend(middle)
#            mean_patient.extend(np.zeros(patient.shape[1]))
#        elif n_above+n_half == 1:
#            middle = patient.reshape(patient.shape[0]*patient.shape[1])
#            mean_patient.extend(middle)
#            mean_patient.extend(np.zeros(patient.shape[1]))
#
#        mean_patient = np.array(mean_patient)
#        if mean_patient.shape[0] != 67584:
#            print(mean_patient.shape)
#            print(patient.shape)
#            print(n_above, n_below)
#        mean_patients.append(mean_patient)
#
#    x = np.array(mean_patients)

    pred = clf.predict(x)
    pred = pred.clip(min=0.)

    df['cancer'] = pred
    df.to_csv(output_directory+"submission.csv", index=False)
    print(df.head())


if __name__ == '__main__':
    input_directory = "/home/andre/kaggle-dsb-2017/data/resnet_features/"
    output_directory = "/home/andre/kaggle-dsb-2017/data/resnet_features/"
    submission_file = "/home/andre/kaggle-dsb-2017/data/stage1_sample_submission.csv"
    model_file = "/home/andre/kaggle-dsb-2017/data/resnet_features/pima.pickle.dat"
    create_submission(input_directory, output_directory, submission_file, model_file)
