"""
Baseline Experiment Utilities for Classical ML

This module provides a comprehensive framework for comparing ML models
with consistent evaluation, timing, and reporting.

Example:
    >>> from baseline_utils import BaselineExperiment
    >>> import xgboost as xgb
    >>> from sklearn.datasets import load_breast_cancer
    >>>
    >>> data = load_breast_cancer()
    >>> exp = BaselineExperiment(
    ...     X=data.data, y=data.target,
    ...     task='classification',
    ...     feature_names=list(data.feature_names)
    ... )
    >>> exp.add_default_models()
    >>> exp.run()
    >>> exp.report()
"""

import numpy as np
import pandas as pd
from time import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass
import json
import warnings

from sklearn.model_selection import (
    train_test_split, cross_val_score,
    KFold, StratifiedKFold
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    warnings.warn("XGBoost not available. Install with: pip install xgboost")


@dataclass
class ModelResult:
    """
    Stores results from a single model evaluation.

    Attributes:
        name: Model display name
        metrics: Dictionary of metric names to values
        cv_scores: Cross-validation scores array
        train_time: Training time in seconds
        inference_time: Inference time in seconds
        feature_importance: Feature importance array (if available)
        model: The trained model object

    Example:
        >>> result = ModelResult(
        ...     name='XGBoost',
        ...     metrics={'accuracy': 0.95, 'f1': 0.94},
        ...     cv_scores=np.array([0.94, 0.95, 0.96, 0.93, 0.95]),
        ...     train_time=1.5,
        ...     inference_time=0.01
        ... )
        >>> print(result.cv_scores.mean())
        0.946
    """
    name: str
    metrics: Dict[str, float]
    cv_scores: np.ndarray
    train_time: float
    inference_time: float
    feature_importance: Optional[np.ndarray] = None
    model: Any = None

    def __repr__(self) -> str:
        return f"ModelResult(name='{self.name}', metrics={self.metrics})"

    def to_dict(self) -> Dict:
        """
        Convert to dictionary for JSON serialization.

        Returns:
            Dictionary with model results (excludes model object).
        """
        return {
            'name': self.name,
            'metrics': self.metrics,
            'cv_mean': float(self.cv_scores.mean()),
            'cv_std': float(self.cv_scores.std()),
            'train_time': self.train_time,
            'inference_time': self.inference_time
        }


class BaselineExperiment:
    """
    A reusable framework for comparing ML models on tabular data.

    Features:
        - Automatic cross-validation
        - Consistent metrics across models
        - Training and inference timing
        - Feature importance extraction
        - Visualization and reporting

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        task: 'classification' or 'regression'
        feature_names: Optional list of feature names
        cv_folds: Number of cross-validation folds
        test_size: Proportion of data for test set
        random_state: Random seed for reproducibility
        scale_features: Whether to scale features for linear models

    Example:
        >>> from sklearn.datasets import fetch_california_housing
        >>> housing = fetch_california_housing()
        >>>
        >>> exp = BaselineExperiment(
        ...     X=housing.data, y=housing.target,
        ...     task='regression',
        ...     feature_names=list(housing.feature_names)
        ... )
        >>> exp.add_default_models()
        >>> exp.run()
        >>> report = exp.report()
        >>> best = exp.get_best_model()
        >>> print(f"Best model: {best.name}")
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task: str = 'classification',
        feature_names: Optional[List[str]] = None,
        cv_folds: int = 5,
        test_size: float = 0.2,
        random_state: int = 42,
        scale_features: bool = True
    ):
        """Initialize the experiment."""
        # Validate task
        if task not in ['classification', 'regression']:
            raise ValueError(f"task must be 'classification' or 'regression', got '{task}'")

        self.X = X.astype(np.float32)
        self.y = y
        self.task = task
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X.shape[1])]
        self.cv_folds = cv_folds
        self.test_size = test_size
        self.random_state = random_state
        self.scale_features = scale_features

        # Model storage
        self.models: Dict[str, Any] = {}
        self.needs_scaling: Dict[str, bool] = {}

        # Results storage
        self.results: List[ModelResult] = []

        # Train/test split
        self._setup_data()

        # Set up cross-validation
        if task == 'classification':
            self.cv = StratifiedKFold(
                n_splits=cv_folds, shuffle=True, random_state=random_state
            )
        else:
            self.cv = KFold(
                n_splits=cv_folds, shuffle=True, random_state=random_state
            )

    def _setup_data(self) -> None:
        """Split data into train and test sets with optional scaling."""
        # Stratify for classification
        stratify = self.y if self.task == 'classification' else None

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify
        )

        # Prepare scaled versions for linear models
        if self.scale_features:
            self.scaler = StandardScaler()
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)
        else:
            self.X_train_scaled = self.X_train
            self.X_test_scaled = self.X_test

    def add_model(
        self,
        name: str,
        model: Any,
        needs_scaling: bool = False
    ) -> 'BaselineExperiment':
        """
        Add a model to the experiment.

        Args:
            name: Display name for the model
            model: sklearn-compatible model instance
            needs_scaling: Whether this model needs scaled features

        Returns:
            self (for method chaining)

        Example:
            >>> exp.add_model('XGBoost', xgb.XGBClassifier())
            >>> exp.add_model('LR', LogisticRegression(), needs_scaling=True)
        """
        self.models[name] = model
        self.needs_scaling[name] = needs_scaling
        return self

    def add_default_models(self) -> 'BaselineExperiment':
        """
        Add default baseline models appropriate for the task.

        For classification: XGBoost, Random Forest, Logistic Regression
        For regression: XGBoost, Random Forest, Ridge Regression

        Returns:
            self (for method chaining)
        """
        if self.task == 'classification':
            # XGBoost
            if XGB_AVAILABLE:
                self.add_model(
                    'XGBoost',
                    xgb.XGBClassifier(
                        n_estimators=100,
                        max_depth=6,
                        learning_rate=0.1,
                        random_state=self.random_state,
                        verbosity=0,
                        n_jobs=-1
                    ),
                    needs_scaling=False
                )

            # Random Forest
            self.add_model(
                'Random Forest',
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=16,
                    n_jobs=-1,
                    random_state=self.random_state
                ),
                needs_scaling=False
            )

            # Logistic Regression
            self.add_model(
                'Logistic Regression',
                LogisticRegression(
                    max_iter=1000,
                    random_state=self.random_state,
                    n_jobs=-1
                ),
                needs_scaling=True
            )

        else:  # regression
            # XGBoost
            if XGB_AVAILABLE:
                self.add_model(
                    'XGBoost',
                    xgb.XGBRegressor(
                        n_estimators=100,
                        max_depth=6,
                        learning_rate=0.1,
                        random_state=self.random_state,
                        verbosity=0,
                        n_jobs=-1
                    ),
                    needs_scaling=False
                )

            # Random Forest
            self.add_model(
                'Random Forest',
                RandomForestRegressor(
                    n_estimators=100,
                    max_depth=16,
                    n_jobs=-1,
                    random_state=self.random_state
                ),
                needs_scaling=False
            )

            # Ridge Regression
            self.add_model(
                'Ridge Regression',
                Ridge(alpha=1.0, random_state=self.random_state),
                needs_scaling=True
            )

        return self

    def _get_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate metrics appropriate for the task."""
        if self.task == 'classification':
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
            }

            # ROC-AUC for binary classification
            if y_proba is not None and len(np.unique(y_true)) == 2:
                try:
                    if y_proba.ndim > 1:
                        metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
                    else:
                        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
                except Exception:
                    pass
        else:
            metrics = {
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred),
                'r2': r2_score(y_true, y_pred)
            }

        return metrics

    def _evaluate_model(self, name: str, model: Any) -> ModelResult:
        """Evaluate a single model with CV, timing, and metrics."""
        # Select appropriate data
        if self.needs_scaling.get(name, False):
            X_train = self.X_train_scaled
            X_test = self.X_test_scaled
        else:
            X_train = self.X_train
            X_test = self.X_test

        # CV scoring
        if self.task == 'classification':
            scoring = 'accuracy'
        else:
            scoring = 'neg_root_mean_squared_error'

        cv_scores = cross_val_score(
            model, X_train, self.y_train,
            cv=self.cv, scoring=scoring, n_jobs=-1
        )

        # Training
        start_time = time()
        model.fit(X_train, self.y_train)
        train_time = time() - start_time

        # Inference
        start_time = time()
        y_pred = model.predict(X_test)
        inference_time = time() - start_time

        # Get probabilities if available
        y_proba = None
        if self.task == 'classification' and hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(X_test)
            except Exception:
                pass

        # Calculate metrics
        metrics = self._get_metrics(self.y_test, y_pred, y_proba)

        # Feature importance
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            feature_importance = np.abs(model.coef_).flatten()

        return ModelResult(
            name=name,
            metrics=metrics,
            cv_scores=cv_scores if self.task == 'classification' else -cv_scores,
            train_time=train_time,
            inference_time=inference_time,
            feature_importance=feature_importance,
            model=model
        )

    def run(self, verbose: bool = True) -> 'BaselineExperiment':
        """
        Run the experiment on all added models.

        Args:
            verbose: Whether to print progress

        Returns:
            self (for method chaining)
        """
        if not self.models:
            raise ValueError("No models added. Use add_model() or add_default_models().")

        self.results = []

        for name, model in self.models.items():
            if verbose:
                print(f"Evaluating: {name}...")

            result = self._evaluate_model(name, model)
            self.results.append(result)

            if verbose:
                if self.task == 'classification':
                    print(f"  CV Accuracy: {result.cv_scores.mean():.4f} (+/- {result.cv_scores.std():.4f})")
                else:
                    print(f"  CV RMSE: {result.cv_scores.mean():.4f} (+/- {result.cv_scores.std():.4f})")

        return self

    def get_best_model(self) -> ModelResult:
        """
        Get the best performing model.

        Returns:
            ModelResult for the best model

        Raises:
            ValueError: If no results available
        """
        if not self.results:
            raise ValueError("No results yet. Run the experiment first!")

        if self.task == 'classification':
            return max(self.results, key=lambda r: r.metrics['accuracy'])
        else:
            return min(self.results, key=lambda r: r.metrics['rmse'])

    def report(self) -> pd.DataFrame:
        """
        Generate a comparison report DataFrame.

        Returns:
            DataFrame with model comparison
        """
        if not self.results:
            raise ValueError("No results yet. Run the experiment first!")

        data = []
        for result in self.results:
            row = {
                'Model': result.name,
                'CV Mean': result.cv_scores.mean(),
                'CV Std': result.cv_scores.std(),
                **result.metrics,
                'Train Time (s)': result.train_time,
                'Inference Time (s)': result.inference_time
            }
            data.append(row)

        df = pd.DataFrame(data)

        # Sort by primary metric
        if self.task == 'classification':
            df = df.sort_values('accuracy', ascending=False)
        else:
            df = df.sort_values('rmse', ascending=True)

        return df

    def save_results(self, filepath: str) -> None:
        """
        Save results to JSON file.

        Args:
            filepath: Path to save JSON file
        """
        data = {
            'experiment_time': datetime.now().isoformat(),
            'task': self.task,
            'n_samples': len(self.X),
            'n_features': self.X.shape[1],
            'cv_folds': self.cv_folds,
            'results': [r.to_dict() for r in self.results]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


def quick_baseline(
    X: np.ndarray,
    y: np.ndarray,
    task: str = 'classification',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Quick baseline comparison with default settings.

    Args:
        X: Feature matrix
        y: Target vector
        task: 'classification' or 'regression'
        verbose: Print progress

    Returns:
        DataFrame with model comparison

    Example:
        >>> from sklearn.datasets import load_iris
        >>> data = load_iris()
        >>> report = quick_baseline(data.data, data.target)
    """
    exp = BaselineExperiment(X=X, y=y, task=task)
    exp.add_default_models()
    exp.run(verbose=verbose)
    return exp.report()
