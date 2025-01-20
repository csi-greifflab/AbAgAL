import pandas as pd
from zipfile import ZipFile
import os
from sklearn.model_selection import train_test_split
import math
from tqdm import tqdm


RANDOM_STATE = 0


def main():

    os.makedirs('../../Data/Processed/', exist_ok=True)
    
    df1 = pd.read_csv('../../Data/Raw/1ADQ_A_5kHeroes-H1.txt', sep='\t')
    ZipFile('../../Data/Raw/1ADQ_AV2PooledComplete.zip', 'r').extractall()
    df2 = pd.read_csv('1ADQ_AV2PooledComplete.txt', sep='\t')
    os.remove('1ADQ_AV2PooledComplete.txt')
    df3 = pd.read_csv('../../Data/Raw/1adq_with_stats.tsv', sep='\t')
    
    df_abag = df2[df2.CDR3.isin(df1.CDR3)][['CDR3', 'Slide', 'Antigen', 'Energy']].drop_duplicates().reset_index(drop=True)
    Ag_lst = list(df_abag.groupby(by='Antigen', as_index=False).count().Antigen)
    mut_lst = '+'.join(['+'.join(x.split('+')[1:]) for x in Ag_lst]).split('+')
    mut_lst = list(set([x[:-1] for x in mut_lst]))

    df4 = pd.DataFrame(mut_lst)
    df4['AA'] = df4.apply(lambda x: x[0][0], axis=1)
    df4['position'] = df4.apply(lambda x: int(x[0][1:]), axis=1)
    df4 = df4.sort_values(by='position').reset_index(drop=True)
    BaseAgSeq = ''.join(df4.AA)
    pos_dct = {df4.loc[i].position: i for i in range(df4.shape[0])}

    # Ag mutant variant ID to Ag mutant variant sequence
    def newseq(seq, mutations, pos_dct):
        mutations = mutations.split('+')[1:]
        for x in mutations:
            mut = x[-1]
            pos = int(x[1:-1])
            seq = seq[:pos_dct[pos]]+mut+seq[pos_dct[pos]+1:]
        return seq
    
    Ag_dct = {}
    for mutations in tqdm(Ag_lst):
        Ag_dct[mutations] = newseq(seq = BaseAgSeq, mutations = mutations, pos_dct = pos_dct)
    df_abag['AgSeq'] = df_abag.apply(lambda x: Ag_dct[x.Antigen], axis=1)
    df_abag.columns = ['CDR3', 'AbSeq', 'Ag', 'Energy', 'AgSeq']
    df_abag = df_abag.sort_values(by=['Ag', 'AbSeq']).reset_index(drop=True)
    
    df_filtered = df2[df2.CDR3.isin(df1.CDR3)].reset_index(drop=True)
    df_merged = pd.merge(df_filtered, df3, how="inner", on='Antigen').query('Nin_3_point < 1 and isInCore and inter_3point < 1 and Best == True')
    df_ab = df_abag[['CDR3', 'Energy']].groupby(by='CDR3', as_index=False).mean()
    df_ab = df_ab.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)[['CDR3']]
    
    total_rows = len(df_ab)
    train_rows = int(total_rows * 0.5)
    test_rows = total_rows - train_rows
    roles = ['train'] * train_rows + ['test'] * test_rows
    df_ab['absplit'] = roles
    df_merged2 = pd.merge(df_merged, df_ab, how="inner", on='CDR3')
    df4 = df_merged2.groupby(by='Antigen', as_index=False).count()
    Ag_lst = list(df4.Antigen)
    mut_lst = '+'.join(['+'.join(x.split('+')[1:]) for x in Ag_lst]).split('+')
    mut_lst = list(set([x[:-1] for x in mut_lst]))
    df5 = pd.DataFrame(mut_lst)
    df5['AA'] = df5.apply(lambda x: x[0][0], axis=1)
    df5['position'] = df5.apply(lambda x: int(x[0][1:]), axis=1)
    df5 = df5.sort_values(by='position').reset_index(drop=True)
    BaseAgSeq = ''.join(df5.AA)
    pos_dct = {df5.loc[i].position: i for i in range(df5.shape[0])}
    
    def newseq(seq, mutations, pos_dct):
        mutations = mutations.split('+')[1:]
        for x in mutations:
            mut = x[-1]
            pos = int(x[1:-1])
            seq = seq[:pos_dct[pos]]+mut+seq[pos_dct[pos]+1:]
        return seq
    
    Ag_dct = {}
    for mutations in tqdm(Ag_lst):
        Ag_dct[mutations] = newseq(seq = BaseAgSeq, mutations = mutations, pos_dct = pos_dct)
    df3_Ag = df_merged2.copy()
    df3_Ag['AgSeq'] = df3_Ag.apply(lambda x: Ag_dct[x.Antigen], axis=1)
    ags = df3_Ag[['AgSeq']].drop_duplicates().reset_index(drop=True)
    ags = ags.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    total_rows = len(ags)
    ag80 = math.ceil(total_rows * 0.8)
    ag20 = total_rows - ag80
    agsplit = ['train'] * ag80 + ['test'] * ag20
    ags['agsplit'] = agsplit
    df3_Ag = pd.merge(df3_Ag, ags, how="inner", on='AgSeq')
    
    def assign_total_split(x):
    	if x['agsplit'] == 'train' and x['absplit'] == 'train':
    		return 'train'
    	if x['agsplit'] == 'train' and x['absplit'] == 'test':
    		return 'testAB'
    	if x['agsplit'] == 'test' and x['absplit'] == 'train':
    		return 'testAG'
    	if x['agsplit'] == 'test' and x['absplit'] == 'test':
    		return 'test'
    		
    df_out = df3_Ag.copy()
    df_out['total_split'] = df3_Ag.apply(assign_total_split, axis=1)
    t = df_out.drop_duplicates(['CDR3', 'Slide', 'Antigen'])
    t = t.drop_duplicates(['CDR3', 'Energy', 'Antigen'])
    df_out = t[['Slide', 'Energy', 'numMut', 'inter_comman_3point_len', 'AgSeq', 'total_split']]
    med = df_out['Energy'].median()
    df_out = df_out.assign(BindClass=df_out['Energy'].apply(lambda x: x <= med))
    df_out.to_csv('../../Data/Processed/filteredCDR_AG_agab.tsv', sep='\t', index=None)
    df_out.head()


if __name__ == "__main__":
    main()
    