# PPP_Trainer

# Intro
A computer vision approach using OpenPose to perform action recognition.

# Commands
Model Train with Clips of Valid Actions; and Validation
`python model.py`

Model Train with Full Videos; and Validation
`python model_4_classes.py`

Model Train with Full Videos and SMOTE; and Validation
`python model_smote.py`

Model Train with Clips of Valid Actions; and Test
`python mdoel_test.py`

# Preprocess:

`prepare_data.py` \ 
`prepare_data_4_classes` to read in all json OpenPose output in `.\data`, perform windowing and feature selections, save results as `.csv`

# Data
https://umass.box.com/s/rtqvadr62phjdivecx8vhe56w19tiihy
