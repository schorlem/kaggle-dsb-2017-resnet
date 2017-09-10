from sklearn.metrics import log_loss
import pandas as pd


def validate_log_loss(y_true_path, y_pred_path):
    df_true = pd.read_csv(y_true_path)
    df_pred = pd.read_csv(y_pred_path)
    y_true = df_true['cancer']
    y_pred = df_pred['cancer']

    loss = log_loss(y_true, y_pred)
    print(loss)


if __name__ == '__main__':
    y_true_file = "path_to_project/data/submission_fin/submissions_fin.csv"
    y_pred_file = "path_to_project/data/resnet_features/submission.csv"
    validate_log_loss(y_true_file, y_pred_file)