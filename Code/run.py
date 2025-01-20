#!/usr/bin/env python

import sys
import os

sys.path.append('../Code/')
import Methods
import importlib

importlib.reload(Methods)
from Methods import *

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    iterations = 40 
    base_antigens_count = 1
    exp_cnt = 100

    data_dir = '../Data/'
    f_train = 'df_train.tsv'
    df_train = pd.read_csv(data_dir + f_train, sep='\t').sample(frac=0.2, random_state=0)
    output_dir = '../Results/'
    os.makedirs(output_dir, exist_ok = True)
    
    df0 = pd.DataFrame(columns = ['iter_cnt', 'base_ag_cnt', 'exp_cnt'])
    df0.loc[len(df0)] = [iterations, base_antigens_count, exp_cnt]
    df0.to_csv(output_dir+'exp_parameters.tsv', sep='\t', index=None)

    #### Random part for comparison
    print("******************************************************")
    print("Random")
    print("******************************************************")
    df3 = pd.DataFrame(columns=['AgSeq', 'iter', 'binding_ratio', 'roc_auc', 'exp_num', 'type'])
    for random_state in range(exp_cnt):
        print("Experiment:", random_state)
        roc_aucs, df_ags = random(dataset=df_train, iterations=iterations,
                                      base_antigens_count=base_antigens_count, training_args=AbAgConvArgs(),
                                      device=device, random_state=random_state)
            
        df_ags['exp_num'] = random_state
        df_ags['type'] = 'random'
        df3 = pd.concat([df3,df_ags], ignore_index=True)
    df3.to_csv(output_dir+'df_random.tsv', sep='\t', index=None)       
    
    #### Gradient approach part
    print("******************************************************")
    print("Gradient approach")
    print("******************************************************")
    df3 = pd.DataFrame(columns=['AgSeq', 'iter', 'binding_ratio', 'roc_auc', 'exp_num', 'type'])
    for random_state in range(exp_cnt):
        print("Experiment:", random_state)
        for threshold in [0.5,0.8]:
            roc_aucs, df_ags = gradient(dataset=df_train, iterations=iterations,
                                      base_antigens_count=base_antigens_count, training_args=AbAgConvArgs(),
                                      device=device, random_state=random_state, threshold=threshold)
            df_ags['exp_num'] = random_state
            df_ags['type'] = 'gradient_' + str( int(threshold*100) )
            df3 = pd.concat([df3,df_ags], ignore_index=True)
    df3.to_csv(output_dir+'df_grad.tsv', sep='\t', index=None)       
    
    #### Alignments approach part
    print("******************************************************")
    print("Alignments approach")
    print("******************************************************")
    df3 = pd.DataFrame(columns=['AgSeq', 'iter', 'binding_ratio', 'roc_auc', 'exp_num', 'type'])
    for random_state in range(exp_cnt):
        print("Experiment:", random_state)
        roc_aucs, df_ags = aligns(dataset=df_train, iterations=iterations,
                                      base_antigens_count=base_antigens_count, training_args=AbAgConvArgs(),
                                      device=device, random_state=random_state)
        df_ags['exp_num'] = random_state
        df_ags['type'] = 'aligns'
        df3 = pd.concat([df3,df_ags], ignore_index=True)
    df3.to_csv(output_dir+'df_aligns.tsv', sep='\t', index=None)        
    
    #### Hamming approach part
    print("******************************************************")
    print("Hamming approach")
    print("******************************************************")
    df3 = pd.DataFrame(columns=['AgSeq', 'iter', 'binding_ratio', 'roc_auc', 'exp_num', 'type'])
    for random_state in range(exp_cnt):
        print("Experiment:", random_state)
        roc_aucs, df_ags = hamming(dataset=df_train, iterations=iterations,
                                      base_antigens_count=base_antigens_count, training_args=AbAgConvArgs(),
                                      device=device, random_state=random_state)
        df_ags['exp_num'] = random_state
        df_ags['type'] = 'hamming'
        df3 = pd.concat([df3,df_ags], ignore_index=True)
    df3.to_csv(output_dir+'df_hamming.tsv', sep='\t', index=None)      
    
    #### Minimum Hamming approach part
    print("******************************************************")
    print("Hamming min approach") 
    print("******************************************************")
    df3 = pd.DataFrame(columns=['AgSeq', 'iter', 'binding_ratio', 'roc_auc', 'exp_num', 'type'])
    for random_state in range(exp_cnt):
        print("Experiment:", random_state)
        roc_aucs, df_ags = hamming_min(dataset=df_train, iterations=iterations,
                                      base_antigens_count=base_antigens_count, training_args=AbAgConvArgs(),
                                      device=device, random_state=random_state)
        df_ags['exp_num'] = random_state
        df_ags['type'] = 'hamming_min'
        df3 = pd.concat([df3,df_ags], ignore_index=True)
    df3.to_csv(output_dir+'df_hamming_min.tsv', sep='\t', index=None)        
    
    #### New gradient approach part
    print("******************************************************")
    print("Gradient with 0-1 labels approach")    
    print("******************************************************")
    df3 = pd.DataFrame(columns=['AgSeq', 'iter', 'binding_ratio', 'roc_auc', 'exp_num', 'type'])
    for random_state in range(exp_cnt):
        print("Experiment:", random_state)
        roc_aucs, df_ags = gradient1(dataset=df_train, iterations=iterations,
                                      base_antigens_count=base_antigens_count, training_args=AbAgConvArgs(),
                                      device=device, random_state=random_state)
        df_ags['exp_num'] = random_state
        df_ags['type'] = 'gradient1'
        df3 = pd.concat([df3,df_ags], ignore_index=True)
    df3.to_csv(output_dir+'df_grad1.tsv', sep='\t', index=None)       

    #### Gradient with confounding labels approach part
    print("******************************************************")
    print("Gradient with confounding labels approach") 
    print("******************************************************")
    df3 = pd.DataFrame(columns=['AgSeq', 'iter', 'binding_ratio', 'roc_auc', 'exp_num', 'type'])
    for random_state in range(exp_cnt):
        print("Experiment:", random_state)
        roc_aucs, df_ags = gradient2(dataset=df_train, iterations=iterations,
                                      base_antigens_count=base_antigens_count, training_args=AbAgConvArgs(),
                                      device=device, random_state=random_state)
        df_ags['exp_num'] = random_state
        df_ags['type'] = 'gradient2'
        df3 = pd.concat([df3,df_ags], ignore_index=True)
    df3.to_csv(output_dir+'df_grad2.tsv', sep='\t', index=None)       

    #### Gradient with respect to the input approach part
    print("******************************************************")
    print("Gradient with respect to the input approach")  
    print("******************************************************")
    df3 = pd.DataFrame(columns=['AgSeq', 'iter', 'binding_ratio', 'roc_auc', 'exp_num', 'type'])
    for random_state in range(exp_cnt):
        print("Experiment:", random_state)
        roc_aucs, df_ags = gradient3(dataset=df_train, iterations=iterations,
                                      base_antigens_count=base_antigens_count, training_args=AbAgConvArgs(),
                                      device=device, random_state=random_state)
        df_ags['exp_num'] = random_state
        df_ags['type'] = 'gradient3'
        df3 = pd.concat([df3,df_ags], ignore_index=True)
    df3.to_csv(output_dir+'df_grad3.tsv', sep='\t', index=None)       
    
    #### Gradient with respect to the model approach part
    print("******************************************************")
    print("Gradient with respect to the model approach")    
    print("******************************************************")
    df3 = pd.DataFrame(columns=['AgSeq', 'iter', 'binding_ratio', 'roc_auc', 'exp_num', 'type'])
    for random_state in range(exp_cnt):
        print("Experiment:", random_state)
        roc_aucs, df_ags = gradient4(dataset=df_train, iterations=iterations,
                                      base_antigens_count=base_antigens_count, training_args=AbAgConvArgs(),
                                      device=device, random_state=random_state)
        df_ags['exp_num'] = random_state
        df_ags['type'] = 'gradient4'
        df3 = pd.concat([df3,df_ags], ignore_index=True)
    df3.to_csv(output_dir+'df_grad4.tsv', sep='\t', index=None)       
