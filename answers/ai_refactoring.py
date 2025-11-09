#!/usr/bin/env python3
"""
Penguin Species Classification using Logistic Regression

This script trains a logistic regression model to classify penguin species
based on physical measurements (bill length, bill depth, flipper length, body mass).
"""

import argparse
import logging
import pickle
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PenguinClassifier:
    """Logistic Regression classifier for penguin species prediction."""
    
    FEATURE_COLUMNS = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
    TARGET_COLUMN = 'species'
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the classifier.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.model = LogisticRegression(random_state=random_state, max_iter=1000)
        self.scaler = StandardScaler()
        self.encoder = LabelEncoder()
        self.is_fitted = False
        
    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load penguin data from CSV file.
        
        Args:
            data_path: Path to the CSV file
            
        Returns:
            DataFrame with loaded data
        """
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} rows")
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the data.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Preprocessing data")
        initial_rows = len(df)
        
        # Drop rows with missing values in required columns
        required_columns = [self.TARGET_COLUMN] + self.FEATURE_COLUMNS
        df_clean = df.dropna(subset=required_columns)
        
        # Select only relevant columns
        df_clean = df_clean[required_columns]
        
        rows_dropped = initial_rows - len(df_clean)
        if rows_dropped > 0:
            logger.info(f"Dropped {rows_dropped} rows with missing values")
        
        logger.info(f"Final dataset: {len(df_clean)} rows")
        logger.info(f"Species distribution:\n{df_clean[self.TARGET_COLUMN].value_counts()}")
        
        return df_clean
    
    def prepare_features(
        self, 
        df: pd.DataFrame, 
        fit: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract and scale features, encode labels.
        
        Args:
            df: Preprocessed DataFrame
            fit: Whether to fit the scaler and encoder (True for training)
            
        Returns:
            Tuple of (scaled_features, encoded_labels)
        """
        logger.info("Preparing features and labels")
        
        # Extract features and target
        X = df[self.FEATURE_COLUMNS].to_numpy()
        y = df[self.TARGET_COLUMN].to_numpy()
        
        # Scale features
        if fit:
            X_scaled = self.scaler.fit_transform(X)
            logger.info("Fitted StandardScaler on features")
        else:
            X_scaled = self.scaler.transform(X)
        
        # Encode labels
        if fit:
            y_encoded = self.encoder.fit_transform(y)
            logger.info(f"Fitted LabelEncoder. Classes: {self.encoder.classes_}")
        else:
            y_encoded = self.encoder.transform(y)
        
        return X_scaled, y_encoded
    
    def train(
        self, 
        data_path: str, 
        test_size: float = 0.2
    ) -> dict:
        """
        Train the logistic regression model.
        
        Args:
            data_path: Path to the training data CSV
            test_size: Proportion of data to use for testing
            
        Returns:
            Dictionary with training metrics
        """
        logger.info("Starting training pipeline")
        
        # Load and preprocess data
        df = self.load_data(data_path)
        df_clean = self.preprocess_data(df)
        
        # Prepare features
        X, y = self.prepare_features(df_clean, fit=True)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        logger.info(f"Split data: {len(X_train)} train, {len(X_test)} test samples")
        
        # Train model
        logger.info("Training logistic regression model")
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = self.model.score(X_test, y_test)
        
        logger.info(f"Training complete. Test accuracy: {accuracy:.4f}")
        
        # Generate detailed metrics
        report = classification_report(
            y_test, 
            y_pred, 
            target_names=self.encoder.classes_,
            output_dict=True
        )
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Log results
        logger.info("\nClassification Report:")
        logger.info("\n" + classification_report(
            y_test, 
            y_pred, 
            target_names=self.encoder.classes_
        ))
        logger.info(f"\nConfusion Matrix:\n{conf_matrix}")
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix.tolist(),
            'test_samples': len(X_test),
            'train_samples': len(X_train)
        }
    
    def predict(
        self, 
        features: np.ndarray, 
        return_proba: bool = False
    ) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            features: Array of shape (n_samples, 4) with feature values
            return_proba: Whether to return probability predictions
            
        Returns:
            Array of predicted species or probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        # Ensure input is 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make predictions
        if return_proba:
            predictions = self.model.predict_proba(features_scaled)
        else:
            predictions_encoded = self.model.predict(features_scaled)
            predictions = self.encoder.inverse_transform(predictions_encoded)
        
        return predictions
    
    def save_model(self, model_path: str) -> None:
        """
        Save the trained model and preprocessors to disk.
        
        Args:
            model_path: Path where to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'encoder': self.encoder,
            'random_state': self.random_state
        }
        
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to the saved model
        """
        logger.info(f"Loading model from {model_path}")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.encoder = model_data['encoder']
        self.random_state = model_data['random_state']
        self.is_fitted = True
        
        logger.info(f"Model loaded. Classes: {self.encoder.classes_}")
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from model coefficients.
        
        Returns:
            DataFrame with feature importance for each class
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before getting feature importance")
        
        coefficients = self.model.coef_
        
        importance_df = pd.DataFrame(
            coefficients.T,
            index=self.FEATURE_COLUMNS,
            columns=self.encoder.classes_
        )
        
        return importance_df


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Train or use a penguin species classifier'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to training data CSV'
    )
    train_parser.add_argument(
        '--output',
        type=str,
        default='penguin_model.pkl',
        help='Path to save the trained model'
    )
    train_parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proportion of data to use for testing'
    )
    train_parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random state for reproducibility'
    )
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions')
    predict_parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model'
    )
    predict_parser.add_argument(
        '--bill-length',
        type=float,
        required=True,
        help='Bill length in mm'
    )
    predict_parser.add_argument(
        '--bill-depth',
        type=float,
        required=True,
        help='Bill depth in mm'
    )
    predict_parser.add_argument(
        '--flipper-length',
        type=float,
        required=True,
        help='Flipper length in mm'
    )
    predict_parser.add_argument(
        '--body-mass',
        type=float,
        required=True,
        help='Body mass in grams'
    )
    predict_parser.add_argument(
        '--proba',
        action='store_true',
        help='Return probability predictions'
    )
    
    # Feature importance command
    importance_parser = subparsers.add_parser(
        'importance', 
        help='Show feature importance'
    )
    importance_parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model'
    )
    
    args = parser.parse_args()
    
    if args.command == 'train':
        # Train a new model
        classifier = PenguinClassifier(random_state=args.random_state)
        metrics = classifier.train(args.data, test_size=args.test_size)
        classifier.save_model(args.output)
        
        print("\nTraining Summary:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Train samples: {metrics['train_samples']}")
        print(f"  Test samples: {metrics['test_samples']}")
        print(f"  Model saved to: {args.output}")
        
    elif args.command == 'predict':
        # Load model and make prediction
        classifier = PenguinClassifier()
        classifier.load_model(args.model)
        
        features = np.array([[
            args.bill_length,
            args.bill_depth,
            args.flipper_length,
            args.body_mass
        ]])
        
        prediction = classifier.predict(features, return_proba=args.proba)
        
        print("\nPrediction:")
        if args.proba:
            print("Probabilities:")
            for species, prob in zip(classifier.encoder.classes_, prediction[0]):
                print(f"  {species}: {prob:.4f}")
        else:
            print(f"  Species: {prediction[0]}")
    
    elif args.command == 'importance':
        # Show feature importance
        classifier = PenguinClassifier()
        classifier.load_model(args.model)
        
        importance_df = classifier.get_feature_importance()
        
        print("\nFeature Importance (Coefficients):")
        print(importance_df.to_string())
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

