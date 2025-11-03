import os
import pandas as pd
from zipfile import ZipFile


def extract_zip(zip_path: str, extract_to: str) -> None:
    """Extracts a ZIP file to the specified directory."""
    with ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_to)


def load_data(raw_dir: str) -> pd.DataFrame:
    """Loads and filters raw antigen-antibody binding data."""
    # Read raw files
    df1 = pd.read_csv(
        os.path.join(raw_dir, '1ADQ_A_5kHeroes-H1.txt'),
        sep='\t'
    )

    # Extract and read pooled complete data
    pooled_zip = os.path.join(raw_dir, '1ADQ_AV2PooledComplete.zip')
    extract_zip(pooled_zip, raw_dir)
    pooled_txt = os.path.join(raw_dir, '1ADQ_AV2PooledComplete.txt')
    df2 = pd.read_csv(pooled_txt, sep='\t')
    # Clean up temporary file
    if os.path.exists(pooled_txt):
        os.remove(pooled_txt)

    # Read stats file
    df3 = pd.read_csv(
        os.path.join(raw_dir, '1adq_with_stats.tsv'),
        sep='\t'
    )

    # Merge and filter
    df_merged = (
        pd.merge(df2, df3, how='inner', on='Antigen')
        .query('Nin_3_point < 1 and isInCore and inter_3point < 1 and Best == True')
        [['CDR3', 'Slide', 'Antigen', 'Energy']]
    )

    # Average duplicates
    return (
        df_merged
        .groupby(['CDR3', 'Slide', 'Antigen'], as_index=False)
        .mean()
    )


def build_mutation_map(df: pd.DataFrame) -> pd.Series:
    """Constructs mapping from Antigen names to mutated sequences."""
    antigen_list = df['Antigen'].unique().tolist()

    # Parse mutation tokens
    mutations = (
        '+'.join([x.split('+', 1)[1] for x in antigen_list if '+' in x])
        .split('+')
    )
    tokens = sorted({tok[:-1] for tok in mutations if tok})

    # Build base sequence and position map
    tokens_df = pd.DataFrame({'token': tokens})
    tokens_df['AA'] = tokens_df['token'].str[0]
    tokens_df['pos'] = tokens_df['token'].str[1:].astype(int)
    tokens_df = tokens_df.sort_values('pos').reset_index(drop=True)

    base_sequence = ''.join(tokens_df['AA'].tolist())
    pos_index = {row['pos']: idx for idx, row in tokens_df.iterrows()}

    def apply_mutations(key: str) -> str:
        seq = list(base_sequence)
        for mut in key.split('+')[1:]:
            pos = int(mut[1:-1])
            aa = mut[-1]
            seq[pos_index[pos]] = aa
        return ''.join(seq)

    return pd.Series({key: apply_mutations(key) for key in antigen_list})


def prepare_final_df(df: pd.DataFrame, mutation_map: pd.Series) -> pd.DataFrame:
    """Assembles final DataFrame with Ab sequences and binding class."""
    df = df.rename(columns={'CDR3': 'AbCDR3Seq', 'Slide': 'AbSlideSeq'})
    df['AgSeq'] = df['Antigen'].map(mutation_map)
    df = df.drop_duplicates(['AbCDR3Seq', 'AgSeq'])

    median_energy = df['Energy'].median()
    df['BindClass'] = df['Energy'] <= median_energy

    return df[['AbCDR3Seq', 'AbSlideSeq', 'AgSeq', 'BindClass']]


def main():
    RAW_DIR = os.path.join('data', 'raw')
    PROCESSED_DIR = os.path.join('data', 'processed')
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    df = load_data(RAW_DIR)
    mutation_map = build_mutation_map(df)
    final_df = prepare_final_df(df, mutation_map)

    output_path = os.path.join(PROCESSED_DIR, 'ab_ag_binding.tsv')
    final_df.to_csv(output_path, sep='\t', index=False)
    print(f'Output saved to {output_path}')


if __name__ == '__main__':
    main()
