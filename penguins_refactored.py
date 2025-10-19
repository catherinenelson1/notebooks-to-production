import requests


def download_data():
    # download the dataset if it doesn't exist already
    try:
        with open(DATA_FILE_PATH, 'rb') as file:
            return pd.read_csv(file)
    except FileNotFoundError:
        url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/refs/heads/master/penguins.csv'
        response = requests.get(url)
        with open(DATA_FILE_PATH, 'wb') as file:
            file.write(response.content)


import pandas as pd

def clean_data():
    # drop rows with missing values
    # return numpy arrays for features and labels
    df = pd.read_csv(DATA_FILE_PATH)
    df = df.dropna(subset=['species', 'bill_length_mm', 'bill_depth_mm', 
                           'flipper_length_mm', 'body_mass_g'])
    features = df[['bill_length_mm', 'bill_depth_mm', 
                   'flipper_length_mm', 'body_mass_g']].to_numpy()
    labels = df['species'].to_numpy()
    return features, labels


from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

import joblib


def preprocess_data(features, labels):
    # encode categorical variables
    # scale numerical features
    # split the data into train/test features and labels
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    joblib.dump(scaler, SCALER_FILENAME)
    encoder = LabelEncoder()
    labels_encoded = encoder.fit_transform(labels)
    joblib.dump(encoder, ENCODER_FILENAME)
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels_encoded, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


from sklearn.linear_model import LogisticRegression


def train_model(X_train, y_train, X_test, y_test):
    # train a model and save it
    clf = LogisticRegression() 
    clf.fit(X_train, y_train)
    # print the model's accuracy
    accuracy = clf.score(X_test, y_test)
    print(f'Model accuracy: {accuracy:.2f}')
    joblib.dump(clf, MODEL_FILENAME)
    

def predict_new_data(X_new):
    # load the model and make predictions
    # return a string of the predicted class
    clf = joblib.load(MODEL_FILENAME)
    scaler = joblib.load(SCALER_FILENAME)
    scaled_data = scaler.transform(X_new)
    prediction = clf.predict(scaled_data)
    prediction_prob = clf.predict_proba(scaled_data)[0][prediction[0]]
    # decode the prediction
    encoder = joblib.load(ENCODER_FILENAME)
    predicted_class = encoder.inverse_transform(prediction)
    return [predicted_class[0], prediction_prob]


def run_training_pipeline():
    # load data
    download_data()
    
    # clean data
    features, labels = clean_data()
    
    # preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(features, labels)
    
    # train model
    train_model(X_train, y_train, X_test, y_test)


MODEL_FILENAME = 'penguins_model.joblib'
ENCODER_FILENAME = 'penguins_label_encoder.joblib'
SCALER_FILENAME = 'penguins_scaler.joblib'
DATA_FILE_PATH = 'penguins_data.csv'

if __name__ == "__main__":
    run_training_pipeline()
    print(predict_new_data([[40, 17, 190, 3500]]))
    print(predict_new_data([[80, 10, 250, 2500]]))
