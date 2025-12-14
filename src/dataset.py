"""
Dataset loader module for HHL project.

This module provides a flexible interface for loading and managing datasets.
Users can easily add their own datasets by creating a new loader function.

Usage:
    from dataset import load_dataset
    
    X, y, metadata = load_dataset('IRIS')
    X, y, metadata = load_dataset('CUSTOM', file_path='data/my_data.csv', 
                                   target_col='target', feature_cols=['feat1', 'feat2'])
"""

import numpy as np
import os
from pathlib import Path
from typing import Tuple, Optional, Dict, List, Union
import warnings

# Try to import optional dependencies
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from sklearn.datasets import load_iris, make_regression, fetch_openml
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


class DatasetMetadata:
    """Metadata about a loaded dataset."""
    def __init__(self, name: str, num_samples: int, num_features: int, 
                 feature_names: Optional[List[str]] = None,
                 target_name: Optional[str] = None,
                 description: Optional[str] = None):
        self.name = name
        self.num_samples = num_samples
        self.num_features = num_features
        self.feature_names = feature_names
        self.target_name = target_name
        self.description = description
    
    def __repr__(self):
        return (f"DatasetMetadata(name='{self.name}', "
                f"samples={self.num_samples}, features={self.num_features})")


def load_iris_dataset(target_index: int = 3, **kwargs) -> Tuple[np.ndarray, np.ndarray, DatasetMetadata]:
    """
    Load IRIS dataset from sklearn.
    
    Args:
        target_index: Index of feature to use as target (0-3)
        **kwargs: Additional arguments (ignored)
    
    Returns:
        Tuple of (X, y, metadata)
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for IRIS dataset. Install with: pip install scikit-learn")
    
    iris = load_iris()
    X_full = iris.data
    feature_names = iris.feature_names
    
    # Use 3 features to predict the 4th feature
    feature_indices = [i for i in range(4) if i != target_index]
    X = X_full[:, feature_indices]
    y = X_full[:, target_index]
    
    metadata = DatasetMetadata(
        name='IRIS',
        num_samples=X.shape[0],
        num_features=X.shape[1],
        feature_names=[feature_names[i] for i in feature_indices],
        target_name=feature_names[target_index],
        description='Iris flower dataset from sklearn'
    )
    
    return X, y, metadata


def load_titanic_dataset(features: List[str] = None, target: str = 'Survived', **kwargs) -> Tuple[np.ndarray, np.ndarray, DatasetMetadata]:
    """
    Load Titanic dataset from various sources.
    
    Args:
        features: List of feature column names to use
        target: Name of target column
        **kwargs: Additional arguments (ignored)
    
    Returns:
        Tuple of (X, y, metadata)
    """
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas is required for Titanic dataset. Install with: pip install pandas")
    
    if features is None:
        features = ['Pclass', 'Sex', 'Age', 'Fare']
    
    # Try to load from common locations
    titanic_paths = [
        'data/titanic.csv',
        'titanic.csv',
        'data/Titanic.csv',
        'Titanic.csv'
    ]
    
    titanic_df = None
    for path in titanic_paths:
        if os.path.exists(path):
            titanic_df = pd.read_csv(path)
            break
    
    if titanic_df is None:
        # Try seaborn
        if SEABORN_AVAILABLE:
            try:
                titanic_df = sns.load_dataset('titanic')
                titanic_df.columns = titanic_df.columns.str.capitalize()
                if 'Pclass' not in titanic_df.columns and 'Class' in titanic_df.columns:
                    titanic_df.rename(columns={'Class': 'Pclass'}, inplace=True)
            except:
                pass
        
        # Try OpenML
        if titanic_df is None and SKLEARN_AVAILABLE:
            try:
                titanic_df = fetch_openml('titanic', version=1, as_frame=True, return_X_y=False)
            except:
                pass
    
    if titanic_df is None:
        raise FileNotFoundError("Could not find Titanic dataset. Place it in data/titanic.csv or install seaborn.")
    
    # Preprocess
    titanic_df = titanic_df.copy()
    
    # Normalize column names
    col_mapping = {col: col.capitalize() for col in titanic_df.columns if col != col.capitalize()}
    if col_mapping:
        titanic_df.rename(columns=col_mapping, inplace=True)
    
    # Handle missing values
    if 'Age' in titanic_df.columns:
        titanic_df['Age'] = pd.to_numeric(titanic_df['Age'], errors='coerce')
        titanic_df['Age'].fillna(titanic_df['Age'].median(), inplace=True)
    
    if 'Fare' in titanic_df.columns:
        titanic_df['Fare'] = pd.to_numeric(titanic_df['Fare'], errors='coerce')
        titanic_df['Fare'].fillna(titanic_df['Fare'].median(), inplace=True)
    
    # Encode categorical
    if 'Sex' in titanic_df.columns:
        titanic_df['Sex'] = titanic_df['Sex'].map({'male': 0, 'female': 1, 'Male': 0, 'Female': 1})
        titanic_df['Sex'] = pd.to_numeric(titanic_df['Sex'], errors='coerce')
        titanic_df['Sex'].fillna(0, inplace=True)
    
    # Select features
    available_features = []
    for f in features:
        f_clean = f.strip()
        if f_clean in titanic_df.columns:
            available_features.append(f_clean)
        elif f_clean.capitalize() in titanic_df.columns:
            available_features.append(f_clean.capitalize())
    
    if not available_features:
        raise ValueError(f"None of the requested features {features} found in dataset.")
    
    # Convert to numeric
    for feat in available_features:
        titanic_df[feat] = pd.to_numeric(titanic_df[feat], errors='coerce')
    
    X = titanic_df[available_features].values.astype(float)
    
    # Select target
    target_col = None
    if target in titanic_df.columns:
        target_col = target
    elif target.capitalize() in titanic_df.columns:
        target_col = target.capitalize()
    
    if target_col is None:
        raise ValueError(f"Target '{target}' not found in dataset.")
    
    y = pd.to_numeric(titanic_df[target_col], errors='coerce').values.astype(float)
    
    # Remove NaN values
    valid_mask = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
    X = X[valid_mask]
    y = y[valid_mask]
    
    metadata = DatasetMetadata(
        name='TITANIC',
        num_samples=X.shape[0],
        num_features=X.shape[1],
        feature_names=available_features,
        target_name=target,
        description='Titanic passenger survival dataset'
    )
    
    return X, y, metadata


def load_synthetic_dataset(n_samples: int = 150, n_features: int = 3, 
                          noise: float = 0.1, random_state: int = 42, **kwargs) -> Tuple[np.ndarray, np.ndarray, DatasetMetadata]:
    """
    Generate synthetic regression dataset.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        noise: Noise level
        random_state: Random seed
        **kwargs: Additional arguments (ignored)
    
    Returns:
        Tuple of (X, y, metadata)
    """
    np.random.seed(random_state)
    
    if SKLEARN_AVAILABLE:
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            noise=noise,
            random_state=random_state,
            bias=1.0
        )
    else:
        X = np.random.randn(n_samples, n_features)
        true_coef = np.random.randn(n_features)
        y = X @ true_coef + np.random.randn(n_samples) * noise + 1.0
    
    metadata = DatasetMetadata(
        name='SYNTHETIC',
        num_samples=n_samples,
        num_features=n_features,
        feature_names=[f'feature_{i}' for i in range(n_features)],
        target_name='target',
        description=f'Synthetic regression dataset (n={n_samples}, features={n_features})'
    )
    
    return X, y, metadata


def load_fish_dataset(target_col: str = 'Weight', feature_cols: List[str] = None, **kwargs) -> Tuple[np.ndarray, np.ndarray, DatasetMetadata]:
    """
    Load Fish dataset from CSV file.
    
    Args:
        target_col: Name of target column (default: 'Weight')
        feature_cols: List of feature column names (default: all except Species and target)
        **kwargs: Additional arguments (ignored)
    
    Returns:
        Tuple of (X, y, metadata)
    """
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas is required for Fish dataset. Install with: pip install pandas")
    
    file_path = kwargs.get('file_path', 'data/Fish.csv')
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Fish dataset not found at {file_path}")
    
    df = pd.read_csv(file_path)
    
    # Encode Species as numeric if it exists
    if 'Species' in df.columns:
        species_map = {species: i for i, species in enumerate(df['Species'].unique())}
        df['Species'] = df['Species'].map(species_map)
    
    # Select features
    if feature_cols is None:
        # Use all columns except target and Species (if already encoded)
        feature_cols = [col for col in df.columns if col != target_col]
    
    # Ensure target exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset. Available: {list(df.columns)}")
    
    # Convert to numeric
    for col in feature_cols + [target_col]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove NaN values
    df = df.dropna(subset=feature_cols + [target_col])
    
    X = df[feature_cols].values.astype(float)
    y = df[target_col].values.astype(float)
    
    metadata = DatasetMetadata(
        name='FISH',
        num_samples=X.shape[0],
        num_features=X.shape[1],
        feature_names=feature_cols,
        target_name=target_col,
        description='Fish species dataset from CSV'
    )
    
    return X, y, metadata


def load_housing_dataset(target_col: str = 'MEDV', feature_cols: List[str] = None, **kwargs) -> Tuple[np.ndarray, np.ndarray, DatasetMetadata]:
    """
    Load Boston Housing dataset from CSV file.
    
    Args:
        target_col: Name of target column (default: 'MEDV' for median value)
        feature_cols: List of feature column names (default: all except target)
        **kwargs: Additional arguments (ignored)
    
    Returns:
        Tuple of (X, y, metadata)
    """
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas is required for Housing dataset. Install with: pip install pandas")
    
    file_path = kwargs.get('file_path', 'data/housing.csv')
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Housing dataset not found at {file_path}")
    
    # Try to read with different separators
    df = None
    for sep in [' ', ',', '\t']:
        try:
            df = pd.read_csv(file_path, sep=sep, header=None)
            if df.shape[1] > 1:
                break
        except:
            continue
    
    if df is None:
        raise ValueError(f"Could not parse housing dataset from {file_path}")
    
    # Standard Boston Housing column names (if not provided)
    if df.shape[1] == 14:  # Standard Boston Housing format
        column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 
                       'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
        df.columns = column_names
    
    # Select features
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col != target_col]
    
    # Ensure target exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Available: {list(df.columns)}")
    
    # Convert to numeric
    for col in feature_cols + [target_col]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove NaN values
    df = df.dropna(subset=feature_cols + [target_col])
    
    X = df[feature_cols].values.astype(float)
    y = df[target_col].values.astype(float)
    
    metadata = DatasetMetadata(
        name='HOUSING',
        num_samples=X.shape[0],
        num_features=X.shape[1],
        feature_names=feature_cols,
        target_name=target_col,
        description='Boston Housing dataset from CSV'
    )
    
    return X, y, metadata


def load_csv_dataset(file_path: str, target_col: str, feature_cols: List[str] = None, 
                    sep: str = ',', **kwargs) -> Tuple[np.ndarray, np.ndarray, DatasetMetadata]:
    """
    Generic CSV dataset loader.
    
    Args:
        file_path: Path to CSV file
        target_col: Name of target column
        feature_cols: List of feature column names (default: all except target)
        sep: CSV separator (default: ',')
        **kwargs: Additional arguments passed to pd.read_csv
    
    Returns:
        Tuple of (X, y, metadata)
    """
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas is required for CSV datasets. Install with: pip install pandas")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at {file_path}")
    
    df = pd.read_csv(file_path, sep=sep, **kwargs)
    
    # Select features
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col != target_col]
    
    # Ensure target exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Available: {list(df.columns)}")
    
    # Convert to numeric
    for col in feature_cols + [target_col]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove NaN values
    df = df.dropna(subset=feature_cols + [target_col])
    
    X = df[feature_cols].values.astype(float)
    y = df[target_col].values.astype(float)
    
    dataset_name = Path(file_path).stem.upper()
    
    metadata = DatasetMetadata(
        name=dataset_name,
        num_samples=X.shape[0],
        num_features=X.shape[1],
        feature_names=feature_cols,
        target_name=target_col,
        description=f'Custom dataset loaded from {file_path}'
    )
    
    return X, y, metadata


# Registry of available datasets
_DATASET_REGISTRY = {
    'IRIS': load_iris_dataset,
    'TITANIC': load_titanic_dataset,
    'SYNTHETIC': load_synthetic_dataset,
    'FISH': load_fish_dataset,
    'HOUSING': load_housing_dataset,
    'CSV': load_csv_dataset,
    'CUSTOM': load_csv_dataset,  # Alias for CSV
}


def list_available_datasets() -> List[str]:
    """Return list of available dataset names."""
    return list(_DATASET_REGISTRY.keys())


def load_dataset(dataset_name: str, **kwargs) -> Tuple[np.ndarray, np.ndarray, DatasetMetadata]:
    """
    Load a dataset by name.
    
    Args:
        dataset_name: Name of dataset (case-insensitive). Options:
            - 'IRIS': Iris flower dataset
            - 'TITANIC': Titanic passenger dataset
            - 'SYNTHETIC': Synthetic regression dataset
            - 'FISH': Fish species dataset (from data/Fish.csv)
            - 'HOUSING': Boston Housing dataset (from data/housing.csv)
            - 'CSV' or 'CUSTOM': Load from custom CSV file
        **kwargs: Additional arguments passed to the dataset loader
    
    Returns:
        Tuple of (X, y, metadata) where:
            - X: Feature matrix (n_samples, n_features)
            - y: Target vector (n_samples,)
            - metadata: DatasetMetadata object
    
    Examples:
        # Load IRIS dataset
        X, y, meta = load_dataset('IRIS', target_index=3)
        
        # Load custom CSV
        X, y, meta = load_dataset('CUSTOM', 
                                  file_path='data/my_data.csv',
                                  target_col='target',
                                  feature_cols=['feat1', 'feat2'])
        
        # Load Fish dataset with custom target
        X, y, meta = load_dataset('FISH', target_col='Length1')
    """
    dataset_name = dataset_name.upper()
    
    if dataset_name not in _DATASET_REGISTRY:
        available = ', '.join(list_available_datasets())
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {available}")
    
    loader_func = _DATASET_REGISTRY[dataset_name]
    
    try:
        X, y, metadata = loader_func(**kwargs)
        return X, y, metadata
    except Exception as e:
        raise RuntimeError(f"Error loading dataset '{dataset_name}': {e}") from e


def register_dataset(name: str, loader_func):
    """
    Register a custom dataset loader.
    
    Args:
        name: Name of the dataset (will be converted to uppercase)
        loader_func: Function that takes **kwargs and returns (X, y, metadata)
    
    Example:
        def my_custom_loader(**kwargs):
            # Load your dataset
            X = ...
            y = ...
            metadata = DatasetMetadata(...)
            return X, y, metadata
        
        register_dataset('MY_DATASET', my_custom_loader)
        X, y, meta = load_dataset('MY_DATASET', param1=value1)
    """
    name = name.upper()
    _DATASET_REGISTRY[name] = loader_func
    print(f"Registered dataset: {name}")


# Example: How to add a custom dataset
if __name__ == '__main__':
    print("Available datasets:", list_available_datasets())
    print("\nExample usage:")
    print("  from dataset import load_dataset")
    print("  X, y, meta = load_dataset('IRIS')")
    print("  X, y, meta = load_dataset('CUSTOM', file_path='data/my_data.csv', target_col='target')")
