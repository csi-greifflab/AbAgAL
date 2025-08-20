import os
import argparse
import pandas as pd
import numpy as np


def prepare_ab_ag_dataset(
    df_binding: pd.DataFrame,
    random_seed: int,
    error_rate: float,
    ab_train_frac: float = 0.5,
    ag_train_frac: float = 0.8,
) -> pd.DataFrame:
    """
    Splits antibody-antigen binding data into train/test for both Ab and Ag,
    deduplicates Ab-Ag pairs, and injects label errors into a fraction of
    cases where both splits are training.

    Parameters
    ----------
    df_binding : pd.DataFrame
        Input DataFrame with columns ['AbCDR3Seq', 'AbSlideSeq', 'AgSeq', 'BindClass'].
    random_seed : int
        Seed for reproducible shuffling.
    error_rate : float
        Fraction of eligible rows (AbTrain & AgTrain) to flip BindClass.
    train_frac : float
        Fraction for both antibody and antigen training splits (default 0.8).

    Returns
    -------
    pd.DataFrame
        DataFrame of unique Ab-Ag pairs with columns:
            ['AbSlideSeq', 'AgSeq', 'BindClass', 'AbTrain', 'AgTrain'].
    """
    rng = np.random.RandomState(random_seed)
    df = df_binding.copy()

    # 1) Split by antibody
    ab_ids = df['AbCDR3Seq'].unique()
    num_ab_train = int(len(ab_ids) * ab_train_frac)
    ab_shuffled = rng.permutation(ab_ids)
    df['AbTrain'] = df['AbCDR3Seq'].isin(ab_shuffled[:num_ab_train])

    # 2) Deduplicate Ab-Ag pairs
    df_pairs = (
        df[['AbSlideSeq', 'AgSeq', 'BindClass', 'AbTrain']]
        .drop_duplicates()
    )
    pair_counts = (
        df_pairs.groupby(['AbSlideSeq', 'AgSeq'])['BindClass']
        .transform('size')
    )
    df_unique = df_pairs[pair_counts == 1].reset_index(drop=True)

    # 3) Split by antigen (same fraction)
    ag_ids = df_unique['AgSeq'].unique()
    num_ag_train = int(len(ag_ids) * ag_train_frac)
    ag_shuffled = rng.permutation(ag_ids)
    df_unique['AgTrain'] = df_unique['AgSeq'].isin(ag_shuffled[:num_ag_train])

    # 4) Inject errors
    eligible_idx = df_unique.index[df_unique['AbTrain'] & df_unique['AgTrain']]
    num_errors = int(len(eligible_idx) * error_rate)
    if num_errors > 0:
        error_idxs = rng.choice(eligible_idx, size=num_errors, replace=False)
        df_unique.loc[error_idxs, 'BindClass'] = (
            ~df_unique.loc[error_idxs, 'BindClass']
        )

    return df_unique


def main():
    parser = argparse.ArgumentParser(
        description='Prepare Ab-Ag dataset with label noise'
    )
    parser.add_argument(
        '--seed', type=int, default=0,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--err_rate', type=float, default=0.05,
        help='Fraction of eligible training labels to flip'
    )
    parser.add_argument(
        '--input', '-i', type=str, required=True,
        help='Input TSV path with columns AbCDR3Seq, AbSlideSeq, AgSeq, BindClass'
    )
    parser.add_argument(
        '--output', '-o', type=str, required=True,
        help='Output TSV path'
    )
    args = parser.parse_args()

    # Load binding data
    df_binding = pd.read_csv(args.input, sep='\t')

    # Process dataset
    df_final = prepare_ab_ag_dataset(
        df_binding,
        random_seed=args.seed,
        error_rate=args.err_rate
    )

    return df_final


if __name__ == '__main__':
    df_final = main()
