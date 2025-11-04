import requests
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

MODEL_FILENAME = 'penguins_model.joblib'
ENCODER_FILENAME = 'penguins_label_encoder.joblib'
SCALER_FILENAME = 'penguins_scaler.joblib'
DATA_FILE_PATH = '../penguins_data.csv'



def download_data(data_file):
    # download the dataset if it doesn't exist already
    try:
        with open(data_file, 'rb') as file:
            return pd.read_csv(file)
    except FileNotFoundError:
        url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/refs/heads/master/penguins.csv'
        response = requests.get(url)
        with open(data_file, 'wb') as file:
            file.write(response.content)


def clean_data(data_file):

    # TODO: your code here
    pass


def preprocess_data(features, labels, scaler_file, encoder_file):

    # TODO: your code here
    pass


def train_model(X_train, y_train, X_test, y_test, model_file):
    # train a model and save it
    clf = LogisticRegression() 
    clf.fit(X_train, y_train)
    # print the model's accuracy
    accuracy = clf.score(X_test, y_test)
    print(f'Model accuracy: {accuracy:.2f}')
    joblib.dump(clf, model_file)
    

def predict_new_data(X_new, model_file, scaler_file, encoder_file):
    # load the model and make predictions
    # return a string of the predicted class
    clf = joblib.load(model_file)
    scaler = joblib.load(scaler_file)
    scaled_data = scaler.transform(X_new)
    prediction = clf.predict(scaled_data)
    prediction_prob = clf.predict_proba(scaled_data)[0][prediction[0]]
    # decode the prediction
    encoder = joblib.load(encoder_file)
    predicted_class = encoder.inverse_transform(prediction)
    return [predicted_class[0], prediction_prob]


def run_training_pipeline():
    # load data
    download_data(DATA_FILE_PATH)
    
    # clean data
    features, labels = clean_data(DATA_FILE_PATH)
    
    # preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(features, labels, SCALER_FILENAME, ENCODER_FILENAME)
    
    # train model
    train_model(X_train, y_train, X_test, y_test, MODEL_FILENAME)


if __name__ == "__main__":
    run_training_pipeline()
    print(predict_new_data([[40, 17, 190, 3500]], MODEL_FILENAME, SCALER_FILENAME, ENCODER_FILENAME))
    print(predict_new_data([[80, 10, 250, 2500]], MODEL_FILENAME, SCALER_FILENAME, ENCODER_FILENAME))
