import pandas as pd
import numpy as np
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix

if __name__ == "__main__":

    train_file_path = './data/features_4.csv'
    test_file_path = './data/test_features_4.csv'
    # Load training data
    raw_data = pd.read_csv(train_file_path)
    raw_test = pd.read_csv(test_file_path)
    dataset = raw_data.values
    test = raw_test.values
    feature_size = 575
    X = dataset[:, 0:feature_size].astype(float)
    Y = dataset[:, feature_size]
    X_test = test[:18000, 0:feature_size].astype(float)
    Y_test = test[:18000, feature_size]

    # Data filtering
    new_X = []
    for i in range(len(X)):
        X[i][np.isnan(X[i])] = 0
        X[i][np.isinf(X[i])] = 0
        new_X.append(X[i])
    new_X = np.array(new_X)

    # Convert true labels into one hot embedding
    encoder = LabelEncoder()
    encoder_Y = encoder.fit_transform(Y)
    new_Y = np_utils.to_categorical(encoder_Y)

    encoder_test = LabelEncoder()
    encoder_Y_test = encoder_test.fit_transform(Y_test)
    Y_test = np_utils.to_categorical(encoder_Y_test)



    # Split into training and testing data
    X_train, X_val, Y_train, Y_val = train_test_split(new_X, new_Y, test_size=0.3, random_state=87)

    #print("X_train shape: ", X_train.shape)
    #print("Y_train shape: ", Y_train.shape)
    #print("X_val shape: ", X_val.shape)
    #print("Y_val shape: ", Y_val.shape)
    #print("X_test: ", X_test.shape)
    #print("Y_test: ", Y_test.shape)

    # our network with several densely connected layers
    model = Sequential()
    model.add(Dense(units=128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(units=64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(units=16, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(units=4, activation='softmax'))

    # Training
    model.compile(optimizer=Adam(0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, Y_train, batch_size=32, epochs=1000, verbose=0, validation_data=(X_val, Y_val))

    # evaluate the model
    scores = model.evaluate(X_test, Y_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    # Confution Matrix and Classification Report
    Y_pred = model.predict(X_test)
    Y_pred = np.argmax(Y_pred, axis = 1)
    Y_test = np.argmax(Y_test, axis = 1)
    target_names = ['Forehand Drive', 'Backhand Drive', 'Forehand Smash', 'Not Valid']

    print("***************************")
    print('Confusion Matrix with Test Data')
    print(confusion_matrix(Y_test, Y_pred))
    print("***************************")
    print('Classification Report with Test Data')
    print(classification_report(Y_test, Y_pred, target_names=target_names))

