# Data Files for Module B: Recommender Systems

This module uses the **MovieLens** dataset, a classic benchmark for recommender systems research.

## Dataset: MovieLens 100K

We use the MovieLens 100K dataset, which contains:
- **100,000 ratings** (1-5 stars)
- **943 users**
- **1,682 movies**
- Ratings collected from 1997-1998

### Automatic Download

The notebooks automatically download the dataset using the `download_movielens()` function from our utilities. No manual download required!

```python
from scripts.data_utils import download_movielens

# Downloads to ~/.cache/movielens/ by default
ratings_df, movies_df = download_movielens(size='100k')
```

### Data Format

#### ratings.csv
| Column | Type | Description |
|--------|------|-------------|
| user_id | int | User identifier (0-942) |
| item_id | int | Movie identifier (0-1681) |
| rating | float | Rating value (1.0-5.0) |
| timestamp | int | Unix timestamp |

#### movies.csv
| Column | Type | Description |
|--------|------|-------------|
| item_id | int | Movie identifier |
| title | str | Movie title with year |
| genres | str | Pipe-separated genre list |

### Dataset Statistics

```
Users:              943
Items:            1,682
Ratings:        100,000
Sparsity:         93.7%  (most user-item pairs have no rating)
Avg ratings/user:   106
Avg ratings/item:    59
Rating distribution:
  1 star:    6,110 (6.1%)
  2 stars:  11,370 (11.4%)
  3 stars:  27,145 (27.1%)
  4 stars:  34,174 (34.2%)
  5 stars:  21,201 (21.2%)
```

### Larger Datasets (Optional)

For more challenging experiments, you can use larger versions:

| Dataset | Ratings | Users | Items | Size |
|---------|---------|-------|-------|------|
| ML-100K | 100K | 943 | 1,682 | 5 MB |
| ML-1M | 1M | 6,040 | 3,706 | 25 MB |
| ML-10M | 10M | 69,878 | 10,677 | 265 MB |
| ML-25M | 25M | 162,541 | 62,423 | 650 MB |

```python
# To use larger datasets:
ratings_df, movies_df = download_movielens(size='1m')  # or '10m', '25m'
```

### DGX Spark Consideration

With 128GB unified memory, you can easily work with ML-25M entirely in GPU memory. For the learning exercises, ML-100K provides fast iteration while teaching the same concepts.

## Generated Data

Some notebooks also generate synthetic data for specific exercises:
- `synthetic_interactions.csv` - Generated implicit feedback data
- `user_features.csv` - Synthetic user feature vectors
- `item_features.csv` - Synthetic item feature vectors

These are created on-the-fly by the notebooks and don't need manual preparation.

## Citation

If you use MovieLens in research:

```bibtex
@article{harper2015movielens,
  title={The MovieLens Datasets: History and Context},
  author={Harper, F. Maxwell and Konstan, Joseph A.},
  journal={ACM Transactions on Interactive Intelligent Systems},
  volume={5},
  number={4},
  pages={19:1--19:19},
  year={2015},
  publisher={ACM}
}
```

## Resources

- [MovieLens Website](https://movielens.org/)
- [GroupLens Research](https://grouplens.org/datasets/movielens/)
