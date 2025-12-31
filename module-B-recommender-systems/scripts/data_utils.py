"""
Data utilities for Recommender Systems module.

This module provides functions for downloading, processing, and preparing
recommendation datasets for training and evaluation.

Professor SPARK's Note:
    "Good data preparation is 80% of the work in ML. These utilities handle
    the boring stuff so you can focus on the fun part - building models!"
"""

import os
import zipfile
import urllib.request
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# Dataset Download and Loading
# =============================================================================

def download_movielens(
    size: str = '100k',
    data_dir: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download and load MovieLens dataset.

    Args:
        size: Dataset size - '100k', '1m', '10m', or '25m'
        data_dir: Directory to save data. Defaults to ~/.cache/movielens/

    Returns:
        Tuple of (ratings_df, movies_df) DataFrames

    Example:
        >>> ratings, movies = download_movielens('100k')
        >>> print(f"Loaded {len(ratings)} ratings for {movies['title'].nunique()} movies")
        Loaded 100000 ratings for 1682 movies

    Note:
        First call downloads the dataset (~5MB for 100k).
        Subsequent calls load from cache.
    """
    # Validate size
    valid_sizes = {'100k', '1m', '10m', '25m'}
    if size.lower() not in valid_sizes:
        raise ValueError(f"size must be one of {valid_sizes}, got '{size}'")

    size = size.lower()

    # Setup directories
    if data_dir is None:
        data_dir = Path.home() / '.cache' / 'movielens'
    else:
        data_dir = Path(data_dir)

    data_dir.mkdir(parents=True, exist_ok=True)

    # URLs for different dataset sizes
    urls = {
        '100k': 'https://files.grouplens.org/datasets/movielens/ml-100k.zip',
        '1m': 'https://files.grouplens.org/datasets/movielens/ml-1m.zip',
        '10m': 'https://files.grouplens.org/datasets/movielens/ml-10m.zip',
        '25m': 'https://files.grouplens.org/datasets/movielens/ml-25m.zip',
    }

    # Check if already downloaded
    zip_path = data_dir / f'ml-{size}.zip'
    extract_dir = data_dir / f'ml-{size}'

    if not extract_dir.exists():
        print(f"Downloading MovieLens {size.upper()} dataset...")
        urllib.request.urlretrieve(urls[size], zip_path)

        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)

        # Clean up zip file
        zip_path.unlink()
        print("Done!")

    # Load data based on format (varies by size)
    if size == '100k':
        ratings_df, movies_df = _load_ml100k(extract_dir)
    elif size == '1m':
        ratings_df, movies_df = _load_ml1m(extract_dir)
    else:  # 10m, 25m
        ratings_df, movies_df = _load_ml_large(extract_dir)

    return ratings_df, movies_df


def _load_ml100k(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load MovieLens 100K format."""
    # Ratings
    ratings_df = pd.read_csv(
        data_dir / 'ml-100k' / 'u.data',
        sep='\t',
        names=['user_id', 'item_id', 'rating', 'timestamp'],
        engine='python'
    )

    # Convert to 0-indexed
    ratings_df['user_id'] = ratings_df['user_id'] - 1
    ratings_df['item_id'] = ratings_df['item_id'] - 1

    # Movies
    movies_df = pd.read_csv(
        data_dir / 'ml-100k' / 'u.item',
        sep='|',
        encoding='latin-1',
        names=['item_id', 'title', 'release_date', 'video_release', 'imdb_url',
               'unknown', 'Action', 'Adventure', 'Animation', 'Children',
               'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
               'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
               'Sci-Fi', 'Thriller', 'War', 'Western'],
        engine='python'
    )
    movies_df['item_id'] = movies_df['item_id'] - 1

    # Create genres column
    genre_cols = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                  'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
                  'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                  'Thriller', 'War', 'Western']

    def get_genres(row):
        return '|'.join([g for g in genre_cols if row[g] == 1])

    movies_df['genres'] = movies_df.apply(get_genres, axis=1)
    movies_df = movies_df[['item_id', 'title', 'genres']]

    return ratings_df, movies_df


def _load_ml1m(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load MovieLens 1M format."""
    # Ratings
    ratings_df = pd.read_csv(
        data_dir / 'ml-1m' / 'ratings.dat',
        sep='::',
        names=['user_id', 'item_id', 'rating', 'timestamp'],
        engine='python'
    )

    # Create 0-indexed mapping
    user_map = {u: i for i, u in enumerate(ratings_df['user_id'].unique())}
    item_map = {m: i for i, m in enumerate(ratings_df['item_id'].unique())}

    ratings_df['user_id'] = ratings_df['user_id'].map(user_map)
    ratings_df['item_id'] = ratings_df['item_id'].map(item_map)

    # Movies
    movies_df = pd.read_csv(
        data_dir / 'ml-1m' / 'movies.dat',
        sep='::',
        names=['item_id', 'title', 'genres'],
        engine='python',
        encoding='latin-1'
    )
    movies_df['item_id'] = movies_df['item_id'].map(item_map)
    movies_df = movies_df.dropna()
    movies_df['item_id'] = movies_df['item_id'].astype(int)

    return ratings_df, movies_df


def _load_ml_large(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load MovieLens 10M/25M format."""
    # Find the actual directory name
    subdirs = [d for d in data_dir.iterdir() if d.is_dir()]
    actual_dir = subdirs[0] if subdirs else data_dir

    # Ratings
    ratings_df = pd.read_csv(
        actual_dir / 'ratings.csv',
        usecols=['userId', 'movieId', 'rating', 'timestamp']
    )
    ratings_df.columns = ['user_id', 'item_id', 'rating', 'timestamp']

    # Create 0-indexed mapping
    user_map = {u: i for i, u in enumerate(ratings_df['user_id'].unique())}
    item_map = {m: i for i, m in enumerate(ratings_df['item_id'].unique())}

    ratings_df['user_id'] = ratings_df['user_id'].map(user_map)
    ratings_df['item_id'] = ratings_df['item_id'].map(item_map)

    # Movies
    movies_df = pd.read_csv(actual_dir / 'movies.csv')
    movies_df.columns = ['item_id', 'title', 'genres']
    movies_df['item_id'] = movies_df['item_id'].map(item_map)
    movies_df = movies_df.dropna()
    movies_df['item_id'] = movies_df['item_id'].astype(int)

    return ratings_df, movies_df


# =============================================================================
# Data Splitting
# =============================================================================

def train_test_split_by_time(
    ratings_df: pd.DataFrame,
    test_ratio: float = 0.2,
    val_ratio: float = 0.0
) -> Tuple[pd.DataFrame, ...]:
    """
    Split ratings by timestamp (more realistic than random split).

    Args:
        ratings_df: DataFrame with ratings and timestamps
        test_ratio: Fraction of data for testing
        val_ratio: Fraction of data for validation

    Returns:
        Tuple of (train_df, test_df) or (train_df, val_df, test_df)

    Example:
        >>> train, test = train_test_split_by_time(ratings, test_ratio=0.2)
        >>> print(f"Train: {len(train)}, Test: {len(test)}")

    Note:
        Time-based splits are more realistic because you can't use
        future information to predict past behavior!
    """
    sorted_df = ratings_df.sort_values('timestamp')
    n = len(sorted_df)

    test_start = int(n * (1 - test_ratio - val_ratio))
    val_start = int(n * (1 - test_ratio))

    train_df = sorted_df.iloc[:test_start].copy()

    if val_ratio > 0:
        val_df = sorted_df.iloc[test_start:val_start].copy()
        test_df = sorted_df.iloc[val_start:].copy()
        return train_df, val_df, test_df
    else:
        test_df = sorted_df.iloc[test_start:].copy()
        return train_df, test_df


def leave_one_out_split(
    ratings_df: pd.DataFrame,
    by_time: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Leave-one-out split: last item per user for testing.

    This is the standard evaluation protocol for implicit feedback.

    Args:
        ratings_df: DataFrame with ratings
        by_time: If True, use most recent item. If False, random.

    Returns:
        Tuple of (train_df, test_df)

    Example:
        >>> train, test = leave_one_out_split(ratings)
        >>> # test has exactly one item per user
        >>> assert test.groupby('user_id').size().max() == 1
    """
    if by_time:
        # Get index of last item per user (by timestamp)
        test_idx = ratings_df.groupby('user_id')['timestamp'].idxmax()
    else:
        # Random item per user
        test_idx = ratings_df.groupby('user_id').sample(n=1).index

    test_df = ratings_df.loc[test_idx].copy()
    train_df = ratings_df.drop(test_idx).copy()

    return train_df, test_df


# =============================================================================
# PyTorch Datasets
# =============================================================================

class RatingsDataset(Dataset):
    """
    PyTorch Dataset for explicit ratings.

    Args:
        ratings_df: DataFrame with user_id, item_id, rating columns

    Example:
        >>> dataset = RatingsDataset(train_df)
        >>> user, item, rating = dataset[0]
        >>> loader = DataLoader(dataset, batch_size=1024, shuffle=True)
    """

    def __init__(self, ratings_df: pd.DataFrame):
        self.users = torch.LongTensor(ratings_df['user_id'].values)
        self.items = torch.LongTensor(ratings_df['item_id'].values)
        self.ratings = torch.FloatTensor(ratings_df['rating'].values)

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.users[idx], self.items[idx], self.ratings[idx]


class ImplicitDataset(Dataset):
    """
    PyTorch Dataset for implicit feedback with negative sampling.

    Args:
        ratings_df: DataFrame with user_id, item_id columns
        num_items: Total number of items
        num_negatives: Number of negative samples per positive

    Example:
        >>> dataset = ImplicitDataset(train_df, num_items=1682, num_negatives=4)
        >>> # Each sample includes 1 positive + 4 negatives
        >>> user, items, labels = dataset[0]
    """

    def __init__(
        self,
        ratings_df: pd.DataFrame,
        num_items: int,
        num_negatives: int = 4
    ):
        self.users = ratings_df['user_id'].values
        self.items = ratings_df['item_id'].values
        self.num_items = num_items
        self.num_negatives = num_negatives

        # Build user->items set for fast negative sampling
        self.user_items: Dict[int, set] = {}
        for user, item in zip(self.users, self.items):
            if user not in self.user_items:
                self.user_items[user] = set()
            self.user_items[user].add(item)

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        user = self.users[idx]
        pos_item = self.items[idx]

        # Sample negative items
        neg_items = []
        user_positives = self.user_items[user]

        while len(neg_items) < self.num_negatives:
            neg_item = np.random.randint(0, self.num_items)
            if neg_item not in user_positives:
                neg_items.append(neg_item)

        # Combine positive and negatives
        all_items = [pos_item] + neg_items
        labels = [1.0] + [0.0] * self.num_negatives

        return (
            torch.LongTensor([user] * (1 + self.num_negatives)),
            torch.LongTensor(all_items),
            torch.FloatTensor(labels)
        )


class TripletDataset(Dataset):
    """
    Dataset for Bayesian Personalized Ranking (BPR) training.

    Returns (user, positive_item, negative_item) triplets.

    Example:
        >>> dataset = TripletDataset(train_df, num_items=1682)
        >>> user, pos_item, neg_item = dataset[0]
    """

    def __init__(self, ratings_df: pd.DataFrame, num_items: int):
        self.users = ratings_df['user_id'].values
        self.items = ratings_df['item_id'].values
        self.num_items = num_items

        # Build user->items set
        self.user_items: Dict[int, set] = {}
        for user, item in zip(self.users, self.items):
            if user not in self.user_items:
                self.user_items[user] = set()
            self.user_items[user].add(item)

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        user = self.users[idx]
        pos_item = self.items[idx]

        # Sample negative item
        neg_item = np.random.randint(0, self.num_items)
        while neg_item in self.user_items[user]:
            neg_item = np.random.randint(0, self.num_items)

        return (
            torch.LongTensor([user]),
            torch.LongTensor([pos_item]),
            torch.LongTensor([neg_item])
        )


# =============================================================================
# Data Analysis Utilities
# =============================================================================

def compute_statistics(ratings_df: pd.DataFrame) -> Dict:
    """
    Compute comprehensive statistics about the ratings dataset.

    Example:
        >>> stats = compute_statistics(ratings)
        >>> print(f"Sparsity: {stats['sparsity']:.2%}")
        Sparsity: 93.70%
    """
    num_users = ratings_df['user_id'].nunique()
    num_items = ratings_df['item_id'].nunique()
    num_ratings = len(ratings_df)

    sparsity = 1 - (num_ratings / (num_users * num_items))

    user_activity = ratings_df.groupby('user_id').size()
    item_popularity = ratings_df.groupby('item_id').size()

    return {
        'num_users': num_users,
        'num_items': num_items,
        'num_ratings': num_ratings,
        'sparsity': sparsity,
        'avg_ratings_per_user': user_activity.mean(),
        'avg_ratings_per_item': item_popularity.mean(),
        'min_ratings_per_user': user_activity.min(),
        'max_ratings_per_user': user_activity.max(),
        'min_ratings_per_item': item_popularity.min(),
        'max_ratings_per_item': item_popularity.max(),
        'rating_mean': ratings_df['rating'].mean(),
        'rating_std': ratings_df['rating'].std(),
    }


def print_dataset_info(ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> None:
    """
    Print a nicely formatted summary of the dataset.

    Example:
        >>> print_dataset_info(ratings, movies)
        ═══════════════════════════════════════════════
        MovieLens Dataset Summary
        ═══════════════════════════════════════════════
        ...
    """
    stats = compute_statistics(ratings_df)

    print("═" * 50)
    print("MovieLens Dataset Summary")
    print("═" * 50)
    print(f"Users:              {stats['num_users']:,}")
    print(f"Items:              {stats['num_items']:,}")
    print(f"Ratings:            {stats['num_ratings']:,}")
    print(f"Sparsity:           {stats['sparsity']:.2%}")
    print("─" * 50)
    print(f"Avg ratings/user:   {stats['avg_ratings_per_user']:.1f}")
    print(f"Avg ratings/item:   {stats['avg_ratings_per_item']:.1f}")
    print(f"Rating mean:        {stats['rating_mean']:.2f}")
    print(f"Rating std:         {stats['rating_std']:.2f}")
    print("─" * 50)

    # Rating distribution
    print("Rating distribution:")
    rating_counts = ratings_df['rating'].value_counts().sort_index()
    for rating, count in rating_counts.items():
        pct = count / len(ratings_df) * 100
        bar = "█" * int(pct / 2)
        print(f"  {rating:.0f} stars: {count:>6,} ({pct:>5.1f}%) {bar}")

    # Top genres
    if 'genres' in movies_df.columns:
        print("─" * 50)
        print("Top genres:")
        all_genres = []
        for genres in movies_df['genres'].dropna():
            all_genres.extend(genres.split('|'))
        genre_counts = pd.Series(all_genres).value_counts().head(5)
        for genre, count in genre_counts.items():
            if genre:  # Skip empty
                print(f"  {genre}: {count}")

    print("═" * 50)


# =============================================================================
# Utility Functions
# =============================================================================

def create_interaction_matrix(
    ratings_df: pd.DataFrame,
    num_users: int,
    num_items: int,
    binary: bool = False
) -> np.ndarray:
    """
    Create user-item interaction matrix.

    Args:
        ratings_df: DataFrame with user_id, item_id, rating
        num_users: Number of users
        num_items: Number of items
        binary: If True, use 1 for any interaction. If False, use ratings.

    Returns:
        Matrix of shape (num_users, num_items)

    Example:
        >>> matrix = create_interaction_matrix(ratings, 943, 1682, binary=True)
        >>> print(f"Shape: {matrix.shape}, Non-zeros: {matrix.sum()}")
    """
    matrix = np.zeros((num_users, num_items), dtype=np.float32)

    for _, row in ratings_df.iterrows():
        user = int(row['user_id'])
        item = int(row['item_id'])
        value = 1.0 if binary else row['rating']
        matrix[user, item] = value

    return matrix


def get_user_history(
    ratings_df: pd.DataFrame,
    user_id: int,
    movies_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Get a user's rating history with movie titles.

    Example:
        >>> history = get_user_history(ratings, user_id=0, movies_df=movies)
        >>> print(history[['title', 'rating']].head())
    """
    user_ratings = ratings_df[ratings_df['user_id'] == user_id].copy()

    if movies_df is not None:
        user_ratings = user_ratings.merge(movies_df, on='item_id', how='left')

    return user_ratings.sort_values('rating', ascending=False)


if __name__ == "__main__":
    # Quick test
    print("Testing data utilities...")
    ratings, movies = download_movielens('100k')
    print_dataset_info(ratings, movies)

    train, test = train_test_split_by_time(ratings, test_ratio=0.2)
    print(f"\nTrain/test split: {len(train):,} / {len(test):,}")

    dataset = RatingsDataset(train)
    print(f"Dataset size: {len(dataset)}")

    print("\n✅ All tests passed!")
