Kaggle Data Science Bowl 2017
==============================

The goal of this competitions was to find lesions in CT images of lungs that are cancerous.

This repository contains the final scripts that were used during 
stage 2 of the Kaggle DSB 2017. 
All additional scripts are located in our private gitlab repository. 

Scripts
--------

1. get_resnet_features.py loads the images applies a minimal amount of preprocessing 
 and uses the data as input for a pretrained resnet50 network. 
 The output of the last average pooling layer is extracted before fully connected layers are applied.

2. train_xgboost.py trains the xgboost classifier using the output of the resnet50 network.

3. predict_xgboost.py creates a submission file containing the predicitions.

4. log_loss_validation.py calculates the logloss of previously 
created predictions on a validation set.

