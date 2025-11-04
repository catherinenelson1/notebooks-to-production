"""
Penguin Species Classification Pipeline

This module provides a complete pipeline for training and using a logistic regression
model to classify penguin species based on physical measurements.
"""

import logging
import sys
from pathlib import Path
from typing import Tuple, List

import requests
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('penguins_pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants
MODEL_FILENAME = 'penguins_lr_model.joblib'
ENCODER_FILENAME = 'penguins_label_encoder.joblib'
SCALER_FILENAME = 'penguins_scaler.joblib'
DATA_FILE_PATH = 'penguins_data.csv'
DATA_URL = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/refs/heads/master/penguins.csv'
FEATURE_COLUMNS = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
TARGET_COLUMN = 'species'
RANDOM_STATE = 42
TEST_SIZE = 0.2


def download_data(data_file: str = DATA_FILE_PATH, url: str = DATA_URL) -> None:
    """
    Download the penguin dataset if it doesn't exist locally.
    
    Args:
        data_file: Path to save the dataset
        url: URL to download the dataset from
        
    Raises:
        requests.RequestException: If download fails
        IOError: If file cannot be written
    """
    data_path = Path(data_file)
    
    if data_path.exists():
        logger.info(f"Dataset already exists at {data_file}")
        return
    
    try:
        logger.info(f"Downloading dataset from {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with open(data_file, 'wb') as file:
            file.write(response.content)
        
        logger.info(f"Dataset successfully downloaded to {data_file}")
        
    except requests.RequestException as e:
        logger.error(f"Failed to download dataset: {e}")
        raise
    except IOError as e:
        logger.error(f"Failed to write dataset to file: {e}")
        raise


def load_and_clean_data(data_file: str = DATA_FILE_PATH) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and clean the penguin dataset.
    
    Args:
        data_file: Path to the dataset CSV file
        
    Returns:
        Tuple of (features, labels) as numpy arrays
        
    Raises:
        FileNotFoundError: If data file doesn't exist
        ValueError: If required columns are missing
        pd.errors.EmptyDataError: If file is empty
    """
    try:
        logger.info(f"Loading data from {data_file}")
        df = pd.read_csv(data_file)
        
        if df.empty:
            raise ValueError("Dataset is empty")
        
        logger.info(f"Dataset loaded with {len(df)} rows")
        
        # Check for required columns
        required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Log missing values
        missing_counts = df[required_columns].isnull().sum()
        if missing_counts.any():
            logger.warning(f"Missing values found:\n{missing_counts[missing_counts > 0]}")
        
        # Drop rows with missing values
        original_len = len(df)
        df = df.dropna(subset=required_columns)
        rows_dropped = original_len - len(df)
        
        if rows_dropped > 0:
            logger.info(f"Dropped {rows_dropped} rows with missing values")
        
        if df.empty:
            raise ValueError("No data remaining after dropping missing values")
        
        # Extract features and labels
        features = df[FEATURE_COLUMNS].to_numpy()
        labels = df[TARGET_COLUMN].to_numpy()
        
        logger.info(f"Cleaned dataset: {len(df)} samples, {features.shape[1]} features")
        logger.info(f"Class distribution:\n{pd.Series(labels).value_counts().to_dict()}")
        
        return features, labels
        
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {data_file}")
        raise
    except pd.errors.EmptyDataError as e:
        logger.error(f"Data file is empty: {data_file}")
        raise
    except Exception as e:
        logger.error(f"Error loading or cleaning data: {e}")
        raise


def preprocess_data(
    features: np.ndarray,
    labels: np.ndarray,
    scaler_file: str = SCALER_FILENAME,
    encoder_file: str = ENCODER_FILENAME,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess features and labels: scale features and encode labels.
    
    Args:
        features: Raw feature array
        labels: Raw label array
        scaler_file: Path to save the fitted scaler
        encoder_file: Path to save the fitted encoder
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
        
    Raises:
        ValueError: If input arrays are invalid
        IOError: If model files cannot be saved
    """
    try:
        logger.info("Starting data preprocessing")
        
        if features.size == 0 or labels.size == 0:
            raise ValueError("Features or labels array is empty")
        
        if len(features) != len(labels):
            raise ValueError(f"Features and labels have different lengths: {len(features)} vs {len(labels)}")
        
        # Scale features
        logger.info("Scaling features")
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Save scaler
        joblib.dump(scaler, scaler_file)
        logger.info(f"Scaler saved to {scaler_file}")
        
        # Encode labels
        logger.info("Encoding labels")
        encoder = LabelEncoder()
        labels_encoded = encoder.fit_transform(labels)
        
        # Save encoder
        joblib.dump(encoder, encoder_file)
        logger.info(f"Encoder saved to {encoder_file}")
        logger.info(f"Label mapping: {dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))}")
        
        # Split data
        logger.info(f"Splitting data (test_size={test_size}, random_state={random_state})")
        X_train, X_test, y_train, y_test = train_test_split(
            features_scaled, labels_encoded,
            test_size=test_size,
            random_state=random_state,
            stratify=labels_encoded
        )
        
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_file: str = MODEL_FILENAME,
    encoder_file: str = ENCODER_FILENAME
) -> LogisticRegression:
    """
    Train a logistic regression model and evaluate it.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        model_file: Path to save the trained model
        encoder_file: Path to the label encoder for report
        
    Returns:
        Trained LogisticRegression model
        
    Raises:
        ValueError: If training data is invalid
        IOError: If model file cannot be saved
    """
    try:
        logger.info("Starting model training")
        
        if X_train.size == 0 or y_train.size == 0:
            raise ValueError("Training data is empty")
        
        # Train model
        model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        model.fit(X_train, y_train)
        
        # Evaluate on training set
        train_accuracy = model.score(X_train, y_train)
        logger.info(f"Training accuracy: {train_accuracy:.4f}")
        
        # Evaluate on test set
        test_accuracy = model.score(X_test, y_test)
        logger.info(f"Test accuracy: {test_accuracy:.4f}")
        
        # Generate predictions and detailed metrics
        y_pred = model.predict(X_test)
        
        # Load encoder for class names
        try:
            encoder = joblib.load(encoder_file)
            target_names = encoder.classes_
        except Exception:
            target_names = None
            logger.warning("Could not load encoder for target names")
        
        # Classification report
        logger.info("Classification Report:")
        report = classification_report(y_test, y_pred, target_names=target_names)
        logger.info(f"\n{report}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"Confusion Matrix:\n{cm}")
        
        # Save model
        joblib.dump(model, model_file)
        logger.info(f"Model saved to {model_file}")
        
        return model
        
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise


def predict(
    X_new: np.ndarray,
    model_file: str = MODEL_FILENAME,
    scaler_file: str = SCALER_FILENAME,
    encoder_file: str = ENCODER_FILENAME
) -> List[Tuple[str, float]]:
    """
    Make predictions on new data.
    
    Args:
        X_new: New feature data (2D array, shape: [n_samples, n_features])
        model_file: Path to the trained model
        scaler_file: Path to the fitted scaler
        encoder_file: Path to the fitted encoder
        
    Returns:
        List of tuples (predicted_class, probability) for each sample
        
    Raises:
        FileNotFoundError: If model files don't exist
        ValueError: If input data is invalid
    """
    try:
        logger.info(f"Making predictions on {len(X_new)} samples")
        
        # Validate input
        if X_new.ndim != 2:
            raise ValueError(f"X_new must be 2D array, got shape {X_new.shape}")
        
        if X_new.shape[1] != len(FEATURE_COLUMNS):
            raise ValueError(f"Expected {len(FEATURE_COLUMNS)} features, got {X_new.shape[1]}")
        
        # Load artifacts
        logger.info("Loading model artifacts")
        model = joblib.load(model_file)
        scaler = joblib.load(scaler_file)
        encoder = joblib.load(encoder_file)
        
        # Scale features
        X_scaled = scaler.transform(X_new)
        
        # Make predictions
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)
        
        # Decode predictions
        predicted_classes = encoder.inverse_transform(predictions)
        
        # Get probability of predicted class
        results = []
        for i, pred_class in enumerate(predicted_classes):
            pred_idx = predictions[i]
            pred_prob = probabilities[i][pred_idx]
            results.append((pred_class, pred_prob))
            logger.info(f"Sample {i+1}: {pred_class} (probability: {pred_prob:.4f})")
        
        return results
        
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise


def run_training_pipeline(data_file: str = DATA_FILE_PATH) -> None:
    """
    Execute the complete training pipeline.
    
    Args:
        data_file: Path to the dataset file
        
    Raises:
        Exception: If any step of the pipeline fails
    """
    try:
        logger.info("=" * 60)
        logger.info("Starting Penguin Classification Training Pipeline")
        logger.info("=" * 60)
        
        # Step 1: Download data
        download_data(data_file)
        
        # Step 2: Load and clean data
        features, labels = load_and_clean_data(data_file)
        
        # Step 3: Preprocess data
        X_train, X_test, y_train, y_test = preprocess_data(features, labels)
        
        # Step 4: Train model
        train_model(X_train, y_train, X_test, y_test)
        
        logger.info("=" * 60)
        logger.info("Training pipeline completed successfully!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"Training pipeline failed: {e}")
        logger.error("=" * 60)
        raise


def main():
    """Main entry point for the script."""
    try:
        # Run training pipeline
        run_training_pipeline()
        
        # Make example predictions
        logger.info("\n" + "=" * 60)
        logger.info("Making example predictions")
        logger.info("=" * 60)
        
        # Example 1: Adelie-like measurements
        example_1 = np.array([[40, 17, 190, 3500]])
        logger.info(f"Example 1 - Features: {example_1[0].tolist()}")
        results_1 = predict(example_1)
        print(f"\nPrediction 1: {results_1[0][0]} (confidence: {results_1[0][1]:.2%})")
        
        # Example 2: Different measurements
        example_2 = np.array([[50, 15, 220, 4500]])
        logger.info(f"Example 2 - Features: {example_2[0].tolist()}")
        results_2 = predict(example_2)
        print(f"Prediction 2: {results_2[0][0]} (confidence: {results_2[0][1]:.2%})")
        
    except Exception as e:
        logger.error(f"Application failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

