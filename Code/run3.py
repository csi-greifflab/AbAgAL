#!/usr/bin/env python

import sys
import os
import torch
import pandas as pd
from QBC import query_by_committee_iter, random_iter
import time
import logging
from tqdm import tqdm


class AbAgConvArgs:
    def __init__(self):
        self.train_batch_size = 256
        self.val_batch_size = 1024
        self.epochs = 100
        self.eps = 1e-07
        self.lr = 1e-02
        self.log_interval = 10
        self.patience = 3


if __name__ == "__main__":
    
    experiment_frac = float(sys.argv[1])
    experiment_name = sys.argv[2]
    print(f"Running experiment: {experiment_name}")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)
    
    committee_size = 5
    base_antigens_count = 5
    exp_cnt = 100
    dis_quantile = 0.9
    
    data_dir = '../../Data/Processed/'
    f_data = 'filteredCDR_AG_agab.tsv'
    output_dir = '../Results/'
    os.makedirs(output_dir, exist_ok=True)
    df_res = pd.DataFrame(columns=['iter', 'binding_ratio', 'exp_num', 'type', 'AgSeq', 'roc_auc_val', 
                                   'roc_aucs_test', 'roc_aucs_testAB', 'roc_aucs_testAG', 'dis_quantile'])
    logging.basicConfig(filename= experiment_name+'_experiment.log', level=logging.INFO, format='%(asctime)s - %(message)s')

    for random_state in range(exp_cnt):

        df_data = pd.read_csv(data_dir + f_data, sep='\t')
        df_data = df_data.groupby('AgSeq', group_keys=False).apply(lambda x: x.sample(frac=experiment_frac, random_state=random_state))
        df_train = df_data.copy()

        iterations = df_data[df_data.total_split=='train'].AgSeq.drop_duplicates().shape[0] - base_antigens_count
        
        start_time = time.time()
        print("Experiment: %d, Disagreement Quantile: %.3f" % (random_state, dis_quantile))
        
        # Run QBC experiment
        df_ags_QBC = query_by_committee_iter(dataset=df_train, committee_size=committee_size, iterations=iterations,
                                               base_antigens_count=base_antigens_count, training_args=AbAgConvArgs(),
                                               device=device, random_state=random_state, dis_quantile=dis_quantile)
        
        # Run Random experiment
        df_ags_random = random_iter(dataset=df_train, committee_size=committee_size, iterations=iterations,
                                      base_antigens_count=base_antigens_count, training_args=AbAgConvArgs(),
                                      device=device, random_state=random_state)

        df_ags_QBC['exp_num'] = random_state
        df_ags_QBC['type'] = 'QBC'
        df_ags_QBC['dis_quantile'] = dis_quantile
        df_ags_random['exp_num'] = random_state
        df_ags_random['type'] = 'random'
        df_res = pd.concat([df_res, df_ags_QBC], ignore_index=True)
        df_res = pd.concat([df_res, df_ags_random], ignore_index=True)
        df_res['ags_number'] = df_res.apply(lambda x: x.iter + base_antigens_count, axis=1)
        df_res.to_csv(output_dir+'df_'+experiment_name+'_'+str(experiment_frac)+'.tsv', sep='\t', index=None)
        torch.cuda.empty_cache()
        
        end_time = time.time()
        experiment_time = end_time - start_time
        logging.info(f"Experiment completed in {experiment_time:.2f} seconds")