"""
Preprocessing Pipeline for Machine Learning
============================================

A production-ready data preprocessing pipeline designed for ML workflows.
Handles missing values, categorical encoding, and feature scaling with
proper train/test separation to avoid data leakage.

This module is part of the DGX Spark AI Curriculum - Module 2.

Example Usage:
    >>> from preprocessing_pipeline import Preprocessor
    >>>
    >>> # Initialize preprocessor
    >>> preprocessor = Preprocessor(
    ...     numeric_features=['age', 'income'],
    ...     categorical_features=['education', 'job_type'],
    ...     ordinal_mappings={'education': {'High School': 0, 'Bachelor': 1, 'Master': 2}},
    ...     scaling='standard',
    ...     impute_strategy='median'
    ... )
    >>>
    >>> # Fit on training data, transform both
    >>> X_train_processed = preprocessor.fit_transform(X_train)
    >>> X_test_processed = preprocessor.transform(X_test)
    >>>
    >>> # Save for production use
    >>> preprocessor.save('preprocessor.pkl')

Author: Professor SPARK
Date: 2024
"""

from typing import Dict, List, Optional, Union, Any
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path


class Preprocessor:
    """
    A reusable data preprocessing pipeline for machine learning.

    This class handles common preprocessing tasks:
    - Missing value imputation (mean, median, mode, or constant)
    - Categorical encoding (one-hot or ordinal/label encoding)
    - Feature scaling (standard, minmax, or robust)
    - Adding missing value indicators
    - Log transformations for skewed features

    The preprocessor follows sklearn's fit/transform pattern to prevent
    data leakage: fit on training data, transform on all data.

    Attributes:
        numeric_features: List of numeric column names
        categorical_features: List of categorical column names
        ordinal_mappings: Dict mapping column names to ordinal encodings
        scaling: Scaling method ('standard', 'minmax', 'robust', or None)
        impute_strategy: Imputation strategy ('mean', 'median', or 'mode')
        add_missing_indicators: Whether to add binary missing indicators
        log_features: Features to apply log transformation to

    Example:
        >>> preprocessor = Preprocessor(
        ...     numeric_features=['age', 'income', 'score'],
        ...     categorical_features=['gender', 'department'],
        ...     scaling='standard'
        ... )
        >>> train_processed = preprocessor.fit_transform(train_df)
        >>> test_processed = preprocessor.transform(test_df)
    """

    def __init__(
        self,
        numeric_features: List[str],
        categorical_features: Optional[List[str]] = None,
        ordinal_mappings: Optional[Dict[str, Dict[str, int]]] = None,
        scaling: Optional[str] = 'standard',
        impute_strategy: str = 'median',
        add_missing_indicators: bool = False,
        log_features: Optional[List[str]] = None,
        handle_unknown_categories: str = 'ignore'
    ):
        """
        Initialize the Preprocessor.

        Args:
            numeric_features: List of numeric column names to process.
            categorical_features: List of categorical column names for one-hot encoding.
                                 Columns in ordinal_mappings are excluded from one-hot.
            ordinal_mappings: Dict of {column_name: {category: int}} for ordinal encoding.
                            Example: {'education': {'High School': 0, 'Bachelor': 1}}
            scaling: Scaling method - 'standard' (z-score), 'minmax' ([0,1]),
                    'robust' (IQR-based), or None for no scaling.
            impute_strategy: How to fill missing numeric values - 'mean', 'median', or 'mode'.
            add_missing_indicators: If True, add binary columns indicating missing values.
            log_features: List of features to apply log1p transformation to.
            handle_unknown_categories: How to handle unseen categories - 'ignore' (zeros),
                                      'error' (raise exception).

        Raises:
            ValueError: If scaling or impute_strategy is invalid.
        """
        # Validate inputs
        valid_scaling = ['standard', 'minmax', 'robust', None]
        if scaling not in valid_scaling:
            raise ValueError(f"scaling must be one of {valid_scaling}, got '{scaling}'")

        valid_impute = ['mean', 'median', 'mode']
        if impute_strategy not in valid_impute:
            raise ValueError(f"impute_strategy must be one of {valid_impute}, got '{impute_strategy}'")

        # Store configuration
        self.numeric_features = list(numeric_features)
        self.categorical_features = list(categorical_features) if categorical_features else []
        self.ordinal_mappings = ordinal_mappings or {}
        self.scaling = scaling
        self.impute_strategy = impute_strategy
        self.add_missing_indicators = add_missing_indicators
        self.log_features = list(log_features) if log_features else []
        self.handle_unknown_categories = handle_unknown_categories

        # These will be learned during fit()
        self._numeric_impute_values: Dict[str, float] = {}
        self._categorical_impute_values: Dict[str, str] = {}
        self._categorical_categories: Dict[str, List[str]] = {}
        self._scale_params: Dict[str, Dict[str, np.ndarray]] = {}
        self._feature_names: List[str] = []
        self._is_fitted: bool = False

    def fit(self, df: pd.DataFrame) -> 'Preprocessor':
        """
        Learn preprocessing parameters from training data.

        This method computes:
        - Imputation values (mean/median for numeric, mode for categorical)
        - Categorical vocabulary (unique values per category)
        - Scaling parameters (mean/std, min/max, or median/IQR)

        Args:
            df: Training DataFrame. Must contain all specified features.

        Returns:
            self: Returns the fitted preprocessor for method chaining.

        Raises:
            ValueError: If required columns are missing from df.
        """
        # Validate all required columns exist
        required_cols = set(self.numeric_features + self.categorical_features)
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in dataframe: {missing_cols}")

        # 1. Learn imputation values for numeric features
        for col in self.numeric_features:
            if self.impute_strategy == 'median':
                self._numeric_impute_values[col] = df[col].median()
            elif self.impute_strategy == 'mean':
                self._numeric_impute_values[col] = df[col].mean()
            else:  # mode
                self._numeric_impute_values[col] = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 0

        # 2. Learn imputation values and categories for categorical features
        for col in self.categorical_features:
            # Mode for imputation
            mode_values = df[col].mode()
            self._categorical_impute_values[col] = mode_values.iloc[0] if len(mode_values) > 0 else 'Unknown'

            # Unique categories (excluding NaN)
            self._categorical_categories[col] = sorted(df[col].dropna().unique().tolist())

        # 3. Learn scaling parameters
        # First, create a temporary imputed version for scaling calculations
        df_temp = df.copy()
        for col in self.numeric_features:
            df_temp[col] = df_temp[col].fillna(self._numeric_impute_values[col])

        # Apply log transform before computing scale params
        for col in self.log_features:
            if col in df_temp.columns:
                df_temp[col] = np.log1p(np.maximum(df_temp[col], 0))

        X = df_temp[self.numeric_features].values.astype(np.float64)

        if self.scaling == 'standard':
            self._scale_params = {
                'center': X.mean(axis=0),
                'scale': X.std(axis=0)
            }
        elif self.scaling == 'minmax':
            self._scale_params = {
                'center': X.min(axis=0),
                'scale': X.max(axis=0) - X.min(axis=0)
            }
        elif self.scaling == 'robust':
            q75 = np.percentile(X, 75, axis=0)
            q25 = np.percentile(X, 25, axis=0)
            self._scale_params = {
                'center': np.median(X, axis=0),
                'scale': q75 - q25
            }

        # Prevent division by zero
        if self.scaling:
            self._scale_params['scale'] = np.where(
                self._scale_params['scale'] == 0,
                1.0,
                self._scale_params['scale']
            )

        # 4. Build output feature names list
        self._feature_names = []

        # Numeric features (with optional missing indicators)
        for col in self.numeric_features:
            self._feature_names.append(col)
            if self.add_missing_indicators:
                self._feature_names.append(f'{col}_missing')

        # Ordinal encoded features
        for col in self.ordinal_mappings:
            self._feature_names.append(f'{col}_encoded')

        # One-hot encoded features (exclude ordinal columns)
        for col in self.categorical_features:
            if col not in self.ordinal_mappings:
                for category in self._categorical_categories[col]:
                    self._feature_names.append(f'{col}_{category}')

        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply preprocessing transformations to data.

        Uses parameters learned during fit() to transform new data.
        This ensures consistent preprocessing between train and test sets.

        Args:
            df: DataFrame to transform. Must contain all required features.

        Returns:
            pd.DataFrame: Transformed DataFrame with processed features.

        Raises:
            ValueError: If preprocessor hasn't been fitted yet.
            ValueError: If unknown categories found and handle_unknown_categories='error'.
        """
        if not self._is_fitted:
            raise ValueError("Preprocessor not fitted! Call fit() or fit_transform() first.")

        result = df.copy()

        # 1. Add missing indicators (before imputation)
        if self.add_missing_indicators:
            for col in self.numeric_features:
                result[f'{col}_missing'] = result[col].isna().astype(int)

        # 2. Impute missing values - numeric
        for col in self.numeric_features:
            result[col] = result[col].fillna(self._numeric_impute_values[col])

        # 3. Impute missing values - categorical
        for col in self.categorical_features:
            result[col] = result[col].fillna(self._categorical_impute_values[col])

        # 4. Apply log transformation
        for col in self.log_features:
            if col in result.columns:
                result[col] = np.log1p(np.maximum(result[col], 0))

        # 5. Ordinal encoding
        for col, mapping in self.ordinal_mappings.items():
            result[f'{col}_encoded'] = result[col].map(mapping)
            # Handle unknown categories
            unknown_mask = result[f'{col}_encoded'].isna()
            if unknown_mask.any():
                if self.handle_unknown_categories == 'error':
                    unknown_vals = result.loc[unknown_mask, col].unique()
                    raise ValueError(f"Unknown categories in '{col}': {unknown_vals}")
                else:
                    # Use -1 for unknown ordinal values
                    result[f'{col}_encoded'] = result[f'{col}_encoded'].fillna(-1)
            result = result.drop(col, axis=1)

        # 6. One-hot encoding
        for col in self.categorical_features:
            if col in self.ordinal_mappings:
                continue  # Already handled as ordinal

            # Create dummy columns for each known category
            for category in self._categorical_categories[col]:
                result[f'{col}_{category}'] = (result[col] == category).astype(int)

            # Check for unknown categories
            known_cats = set(self._categorical_categories[col])
            actual_cats = set(result[col].dropna().unique())
            unknown_cats = actual_cats - known_cats

            if unknown_cats and self.handle_unknown_categories == 'error':
                raise ValueError(f"Unknown categories in '{col}': {unknown_cats}")

            result = result.drop(col, axis=1)

        # 7. Scale numeric features
        if self.scaling:
            for i, col in enumerate(self.numeric_features):
                center = self._scale_params['center'][i]
                scale = self._scale_params['scale'][i]
                result[col] = (result[col] - center) / scale

        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit preprocessor and transform data in one step.

        Convenience method that calls fit() then transform().

        Args:
            df: Training DataFrame to fit on and transform.

        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        return self.fit(df).transform(df)

    def get_feature_names(self) -> List[str]:
        """
        Get names of all output features after transformation.

        Returns:
            List[str]: Feature names in order they appear in transformed output.

        Raises:
            ValueError: If preprocessor hasn't been fitted yet.
        """
        if not self._is_fitted:
            raise ValueError("Preprocessor not fitted! Call fit() first.")
        return self._feature_names.copy()

    def get_params(self) -> Dict[str, Any]:
        """
        Get all learned parameters for inspection or debugging.

        Returns:
            Dict containing imputation values, categories, and scale parameters.
        """
        return {
            'numeric_impute_values': self._numeric_impute_values.copy(),
            'categorical_impute_values': self._categorical_impute_values.copy(),
            'categorical_categories': {k: v.copy() for k, v in self._categorical_categories.items()},
            'scale_params': {k: v.copy() for k, v in self._scale_params.items()} if self._scale_params else None,
            'is_fitted': self._is_fitted
        }

    def save(self, path: Union[str, Path]) -> None:
        """
        Save preprocessor to disk for later use.

        Args:
            path: File path to save to (should end in .pkl).

        Example:
            >>> preprocessor.save('models/preprocessor.pkl')
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'Preprocessor':
        """
        Load a saved preprocessor from disk.

        Args:
            path: File path to load from.

        Returns:
            Preprocessor: The loaded preprocessor instance.

        Example:
            >>> preprocessor = Preprocessor.load('models/preprocessor.pkl')
        """
        with open(path, 'rb') as f:
            return pickle.load(f)

    def inverse_transform_numeric(
        self,
        data: np.ndarray,
        feature_idx: Optional[int] = None
    ) -> np.ndarray:
        """
        Inverse transform scaled numeric features back to original scale.

        Useful for interpreting model predictions or coefficients.

        Args:
            data: Scaled data to inverse transform.
            feature_idx: If provided, only inverse transform this feature index.
                        Otherwise, inverse transforms all numeric features.

        Returns:
            np.ndarray: Data in original scale.
        """
        if not self.scaling:
            return data

        if feature_idx is not None:
            center = self._scale_params['center'][feature_idx]
            scale = self._scale_params['scale'][feature_idx]
            result = data * scale + center
        else:
            result = data * self._scale_params['scale'] + self._scale_params['center']

        return result

    def __repr__(self) -> str:
        """String representation of the preprocessor."""
        fitted_str = "fitted" if self._is_fitted else "not fitted"
        return (
            f"Preprocessor(\n"
            f"  numeric_features={self.numeric_features},\n"
            f"  categorical_features={self.categorical_features},\n"
            f"  ordinal_mappings={list(self.ordinal_mappings.keys())},\n"
            f"  scaling='{self.scaling}',\n"
            f"  impute_strategy='{self.impute_strategy}',\n"
            f"  status={fitted_str}\n"
            f")"
        )


# Standalone utility functions
def detect_feature_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Automatically detect numeric and categorical features in a DataFrame.

    Args:
        df: DataFrame to analyze.

    Returns:
        Dict with 'numeric' and 'categorical' keys containing feature lists.

    Example:
        >>> feature_types = detect_feature_types(df)
        >>> print(feature_types['numeric'])
        ['age', 'income', 'score']
    """
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()

    return {
        'numeric': numeric_features,
        'categorical': categorical_features
    }


def create_preprocessing_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a report summarizing data quality issues in a DataFrame.

    Args:
        df: DataFrame to analyze.

    Returns:
        DataFrame with columns: feature, dtype, missing_count, missing_pct, unique_values

    Example:
        >>> report = create_preprocessing_report(df)
        >>> print(report[report['missing_pct'] > 0])
    """
    report_data = []

    for col in df.columns:
        missing_count = df[col].isna().sum()
        missing_pct = (missing_count / len(df)) * 100
        unique_count = df[col].nunique()

        report_data.append({
            'feature': col,
            'dtype': str(df[col].dtype),
            'missing_count': missing_count,
            'missing_pct': round(missing_pct, 2),
            'unique_values': unique_count,
            'sample_values': str(df[col].dropna().head(3).tolist())
        })

    return pd.DataFrame(report_data)


if __name__ == "__main__":
    # Demo usage
    print("Preprocessing Pipeline Demo")
    print("=" * 50)

    # Create sample data
    np.random.seed(42)
    n = 100

    demo_data = pd.DataFrame({
        'age': np.random.randint(18, 70, n).astype(float),
        'income': np.random.lognormal(10, 0.5, n),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n),
        'department': np.random.choice(['Sales', 'Engineering', 'Marketing'], n),
        'target': np.random.randint(0, 2, n)
    })

    # Add some missing values
    demo_data.loc[np.random.choice(n, 10), 'age'] = np.nan
    demo_data.loc[np.random.choice(n, 15), 'income'] = np.nan

    print("\nOriginal data shape:", demo_data.shape)
    print("\nMissing values:")
    print(demo_data.isnull().sum())

    # Create and fit preprocessor
    preprocessor = Preprocessor(
        numeric_features=['age', 'income'],
        categorical_features=['education', 'department'],
        ordinal_mappings={
            'education': {'High School': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3}
        },
        scaling='standard',
        impute_strategy='median',
        add_missing_indicators=True
    )

    # Process data
    processed = preprocessor.fit_transform(demo_data.drop('target', axis=1))

    print("\nProcessed data shape:", processed.shape)
    print("\nProcessed columns:", list(processed.columns))
    print("\nSample of processed data:")
    print(processed.head())

    print("\n" + "=" * 50)
    print("Demo complete!")
