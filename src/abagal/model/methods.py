from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
import pandas as pd
import typing as tp
import Levenshtein
from scipy.spatial import distance
from Bio import Align
from tqdm import tqdm

import importlib

import model.abagal
importlib.reload(model.abagal)
from model.abagal import *
from model.qbc import train, train_committee


class AbAgConvArgs:
    def __init__(self):
        self.train_batch_size = 64
        self.val_batch_size = 1024
        self.test_batch_size = 1024
        self.epochs = 100
        self.eps = 1e-07
        self.lr = 1e-02
        self.gamma = 0.7
        self.no_cuda = False
        self.no_mps = False
        self.dry_run = False
        self.seed = 0
        self.log_interval = 10
        self.save_model = False
        self.patience = 3


def train_confounding(args, model, device, train_loader, validation_loader, optimizer, criterion, epochs, verbose=False):
    patience = 0
    for epoch in range(epochs):
        if patience == args.patience:
            break
        model.train(True)
        running_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.type(torch.LongTensor).to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            running_loss += loss.item()
            optimizer.step()
        avg_loss = running_loss / (batch_idx + 1)
        running_vloss = 0.0
        model.eval()
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels = vdata
                voutputs = model(vinputs)
                vloss = criterion(voutputs, vlabels.type(torch.LongTensor).to(device))
                running_vloss += vloss
        avg_vloss = running_vloss / (i + 1)
        if epoch == 0:
            best_vloss = avg_vloss
        elif avg_vloss < best_vloss:
            best_vloss = avg_vloss
            patience = 0
        else:
            patience += 1

def committee_train_confounding(dataset: pd.DataFrame, committee: tp.List[AbAgConvNet], antigen_base_list: tp.List[str], training_args, device, random_state):
    dataset['split_train_val'] = dataset['total_split']
    dataset_filtered_ags = dataset[dataset.AgSeq.isin(antigen_base_list)]
    dataset_filtered_ags = dataset_filtered_ags.copy()
    dataset_filtered_ags.loc[:, 'split_train_val'] = dataset_filtered_ags['total_split']
    train_indices = dataset_filtered_ags[dataset_filtered_ags['total_split'] == 'train'].index
    train_indices, val_indices = train_test_split(train_indices, test_size=0.2, random_state=random_state)
    dataset_filtered_ags.loc[train_indices, 'split_train_val'] = 'train'
    dataset_filtered_ags.loc[val_indices, 'split_train_val'] = 'val'
    df_split = {}
    for split_type in ['train', 'val']:
        df_split[split_type] = dataset_filtered_ags[dataset_filtered_ags.split_train_val == split_type]
    for split_type in ['test', 'testAB', 'testAG']:
        df_split[split_type] = dataset[dataset.split_train_val == split_type]
    datasets = {}
    loaders = {}
    for split_type, df in df_split.items():
        datasets[split_type] = AbAgDataset(df=df, device=device)
        batch_size = training_args.train_batch_size if split_type == 'train' else training_args.val_batch_size
        loaders[split_type] = torch.utils.data.DataLoader(dataset=datasets[split_type], batch_size=batch_size, num_workers=0, pin_memory=False)
    committee_optimizers = [torch.optim.Adam(model.parameters(), eps=training_args.eps, lr=training_args.lr) for model in committee]
    criterion = nn.NLLLoss()

    for i, model in enumerate(committee):
        train_confounding(args=training_args, model=model, device=device, train_loader=loaders['train'],
                      validation_loader=loaders['val'], optimizer=committee_optimizers[i], criterion=criterion,
                      epochs=training_args.epochs, verbose=False)
    
    return committee


def gradient(dataset: pd.DataFrame, iterations: int, base_antigens_count, training_args: AbAgConvArgs, device: str, random_state: int, threshold = 0.5, option = 'threshold', metrics = 'mean') -> tp.List[float]: 
    """
    Runs a set of gradient-based approaches for several iterations to select antigens for the model.
    """
    df_antigens = dataset[dataset.total_split=='train'][['AgSeq']].drop_duplicates().reset_index(drop=True)
    antigen_list = list(df_antigens.sample(frac=1.0, random_state=random_state).AgSeq)
    antigen_base_list = antigen_list[:base_antigens_count]
    antigen_add_list = antigen_list[base_antigens_count:]
    
    torch.manual_seed(random_state)
    net = AbAgConvNet().to(device)
    nets, df_train_ags = train_committee(dataset, [net], antigen_base_list, training_args,
                                            device, random_state, 0, -1)
    net = nets[0]
    criterion = F.binary_cross_entropy_with_logits
    
    for k in tqdm(range(iterations)):
        gradients = []
        for n in range(len(antigen_add_list)):
            model = net.eval()
            df3 = dataset.loc[(dataset.total_split=='train') & (dataset.AgSeq == antigen_add_list[n])]
            
            if df3.shape[0] > 100:
                df3 = df3.sample(n=100, random_state=random_state)
                
            dataset_new_antigen = AbAgDataset(df=df3, device=device)
            loader = torch.utils.data.DataLoader(dataset=dataset_new_antigen, batch_size=training_args.train_batch_size)
            optimizer = torch.optim.Adam(model.parameters(), eps=training_args.eps, lr=training_args.lr)
            grad = []
            
            if option == 'model_grad':
                for data, _ in loader:
                    for id, item in enumerate(data):
                        optimizer.zero_grad()
                        out = model(item)
                        out.backward()
                        batch_grad_t = model.fc2.weight.grad.detach().cpu().numpy()
                        batch_grad_norm = sum([i**2 for i in batch_grad_t[0]])**(1/2)
                        grad.append(batch_grad_norm)

            elif option == 'both_labels':
                for data, _ in loader:
                    optimizer.zero_grad()
                    out = model(data).flatten()
                    preds0 = torch.zeros(len(out)).to(device)
                    loss0 = criterion(out.flatten(), preds0, reduction='sum')
                    loss0.backward()
                    batch_grad_t0 = model.fc2.weight.grad.detach().cpu().numpy()
                    
                    optimizer.zero_grad()
                    out = model(data).flatten()
                    preds1 = torch.ones(len(out)).to(device)
                    loss1 = criterion(out.flatten(), preds1, reduction='sum')
                    loss1.backward()
                    batch_grad_t1 = model.fc2.weight.grad.detach().cpu().numpy()
                    grad.append(distance.euclidean(batch_grad_t0[0], batch_grad_t1[0]))
                
            elif option == 'threshold':
                for data, _ in loader:
                    optimizer.zero_grad()
                    out = model(data).flatten()
                    preds = torch.sigmoid(out)
                    target = (preds > threshold).float().detach()
                    loss = criterion(out.flatten(), target, reduction='sum')
                    loss.backward()
                    batch_grad_t = model.fc2.weight.grad.detach().cpu().numpy()
                    batch_grad_norm = sum([i**2 for i in batch_grad_t[0]])**(1/2)
                    grad.append(batch_grad_norm)
                    
            else:
                raise ValueError(f"Unknown option: {method}")
                
            if metrics == 'mean':
                gradients.append(sum(grad)/len(grad))
            elif metrics == 'max':
                gradients.append(max([abs(i) for i in grad]))
            else: 
                raise ValueError(f"Unknown metrics option: {metrics}")
        
        new_antigen = antigen_add_list.pop(np.argmax(gradients))
        antigen_base_list.append(new_antigen)
        
        torch.manual_seed(random_state)
        nets, df_train_ags_iter = train_committee(dataset, [net], antigen_base_list, training_args,
                                            device, random_state, k+1, -1)
        net = nets[0]
        df_train_ags = pd.concat([df_train_ags, df_train_ags_iter], ignore_index=True)
    return df_train_ags

def gradient_confounding(dataset: pd.DataFrame, iterations: int, base_antigens_count, training_args: AbAgConvArgs, device: str, random_state: int, metrics = 'mean') -> tp.List[float]: 

    df_antigens = dataset[dataset.total_split=='train'][['AgSeq']].drop_duplicates().reset_index(drop=True)
    antigen_list = list(df_antigens.sample(frac=1.0, random_state=random_state).AgSeq)
    antigen_base_list = antigen_list[:base_antigens_count]
    antigen_add_list = antigen_list[base_antigens_count:]
    
    torch.manual_seed(random_state)
    net = AbAgConvNet().to(device)
    nets, df_train_ags = train_committee(dataset, [net], antigen_base_list, training_args,
                                            device, random_state, 0, -1)
    net = nets[0]

    net_confounding = AbAgConvNet_confounding().to(device)
    nets_confounding = committee_train_confounding(dataset, [net_confounding], antigen_base_list, training_args,
                                            device, random_state)
    net_confounding = nets_confounding[0]

    criterion = nn.NLLLoss()
    
    for k in tqdm(range(iterations)):
        gradients = []
        for n in range(len(antigen_add_list)):
            model_confounding = net_confounding.eval()
            df3 = dataset.loc[(dataset.total_split=='train') & (dataset.AgSeq == antigen_add_list[n])]
            
            if df3.shape[0] > 100:
                df3 = df3.sample(n=100, random_state=random_state)
            dataset_new_antigen = AbAgDataset(df=df3, device=device)
            loader = torch.utils.data.DataLoader(dataset=dataset_new_antigen, batch_size=training_args.train_batch_size)
            optimizer_confounding = torch.optim.Adam(model_confounding.parameters(), eps=training_args.eps, lr=training_args.lr)
            grad = []
            for data, _ in loader:
                optimizer_confounding.zero_grad()
                out = model_confounding(data)
                preds = out
                loss = torch.sum(out)*(-1)
                loss.backward()
                batch_grad_t = model_confounding.fc2.weight.grad.detach().cpu().numpy()
                batch_grad_norm = sum([i**2 for i in batch_grad_t[0]])**(1/2)
                grad.append(batch_grad_norm)
                
            if metrics == 'mean':
                gradients.append(sum(grad)/len(grad))
            elif metrics == 'max':
                gradients.append(max([abs(i) for i in grad]))
            else: 
                raise ValueError(f"Unknown metrics option: {metrics}")
        
        new_antigen = antigen_add_list.pop(np.argmax(gradients))
        antigen_base_list.append(new_antigen)

        nets, df_train_ags_iter = train_committee(dataset, [net], antigen_base_list, 
                                                         training_args, device, random_state, k+1, -1)
        net = nets[0]
        nets_confounding = committee_train_confounding(dataset, [net_confounding], antigen_base_list, training_args,
                                            device, random_state)
        net_confounding = nets_confounding[0]

        df_train_ags = pd.concat([df_train_ags, df_train_ags_iter], ignore_index=True)
    return df_train_ags
    
def gradient_input(dataset: pd.DataFrame, iterations: int, base_antigens_count, training_args: AbAgConvArgs, device: str, random_state: int, metrics = 'mean') -> tp.List[float]: 
    """
    Runs a gradient-based approach which counts gradient with respect to the input for several iterations to select antigens for the model.
    """
    df_antigens = dataset[dataset.total_split=='train'][['AgSeq']].drop_duplicates().reset_index(drop=True)
    antigen_list = list(df_antigens.sample(frac=1.0, random_state=random_state).AgSeq)
    antigen_base_list = antigen_list[:base_antigens_count]
    antigen_add_list = antigen_list[base_antigens_count:]
    
    torch.manual_seed(random_state)
    net = AbAgConvNet().to(device)
    nets, df_train_ags = train_committee(dataset, [net], antigen_base_list, training_args,
                                            device, random_state, 0, -1)
    net = nets[0]
    
    torch.manual_seed(random_state)
    net_grad = AbAgConvNet_grad().to(device)
    nets_grad, df_train_ags_grad = train_committee(dataset, [net_grad], antigen_base_list, training_args,
                                            device, random_state, 0, -1)
    net_grad = nets_grad[0]
    criterion = F.binary_cross_entropy_with_logits
    
    for k in tqdm(range(iterations)):
        gradients = []
        for n in range(len(antigen_add_list)):
            model_grad = net_grad.eval()
            df3 = dataset[(dataset.total_split=='train') & (dataset.AgSeq == antigen_add_list[n])]
            
            if df3.shape[0] > 100:
                df3 = df3.sample(n=100, random_state=random_state)
                
            dataset_new_antigen = AbAgDataset(df=df3, device=device)
            loader = torch.utils.data.DataLoader(dataset=dataset_new_antigen, batch_size=training_args.train_batch_size)
            optimizer_grad = torch.optim.Adam(model_grad.parameters(), eps=training_args.eps, lr=training_args.lr)
            grad = []
            for data, _ in loader:
                optimizer_grad.zero_grad()
                out = model_grad(data).flatten()
                preds0 = torch.zeros(len(out)).to(device)
                loss0 = criterion(out.flatten(), preds0, reduction='sum')
                loss0.backward()
                batch_grad_t0 = model_grad.embetter.grad.detach().cpu().numpy()
                
                optimizer_grad.zero_grad()
                out = model_grad(data).flatten()
                preds1 = torch.ones(len(out)).to(device)
                loss1 = criterion(out.flatten(), preds1, reduction='sum')
                loss1.backward()
                batch_grad_t1 = model_grad.embetter.grad.detach().cpu().numpy()
                for grad_n in range(model_grad.embetter.grad.shape[0]):
                    grad.append(distance.euclidean( sum(batch_grad_t0[grad_n].T), sum(batch_grad_t1[grad_n].T)))
                
            if metrics == 'mean':
                gradients.append(sum(grad)/len(grad))
            elif metrics == 'max':
                gradients.append(max([abs(i) for i in grad]))
            else: 
                raise ValueError(f"Unknown metrics option: {metrics}")
        
        new_antigen = antigen_add_list.pop(np.argmax(gradients))
        antigen_base_list.append(new_antigen)

        torch.manual_seed(random_state)
        nets, df_train_ags_iter = train_committee(dataset, [net], antigen_base_list, training_args,
                                            device, random_state, k+1, -1)
        net = nets[0]

        torch.manual_seed(random_state)
        nets_grad, df_train_ags_grad = train_committee(dataset, [net_grad], antigen_base_list, training_args,
                                            device, random_state, k+1, -1)
        net_grad = nets_grad[0]


        df_train_ags = pd.concat([df_train_ags, df_train_ags_iter], ignore_index=True)
    return df_train_ags


def hamming_opt(dataset, antigen_base_list, antigen_add_list):
    """
    Counts max hamming-based distance between an antigene and a set of antigens.
    """
    base_list = list(dataset.AgSeq[dataset.AgSeq.isin(antigen_base_list)].unique())
    add_list = list(dataset.AgSeq[dataset.AgSeq.isin(antigen_add_list)].unique())
    dist_max = 0
    ag_name = ""
    for new_ag in add_list:
        dist=0
        for old_ag in base_list:
            dist+= sum(c1 != c2 for c1, c2 in zip(old_ag, new_ag))
        if dist>dist_max:
            dist_max = dist
            ag_name = new_ag
    ag_full_name = dataset.AgSeq[dataset.AgSeq==ag_name].unique()[0]
    return(antigen_add_list.index(ag_full_name))

def hamming_opt_min_dist(dataset, antigen_base_list, antigen_add_list):
    """
    Counts min hamming-based distance between an antigene and a set of antigens.
    """   
    base_list = list(dataset.AgSeq[dataset.AgSeq.isin(antigen_base_list)].unique())
    add_list = list(dataset.AgSeq[dataset.AgSeq.isin(antigen_add_list)].unique())
    dist_min = 10000000
    ag_name = ""
    for new_ag in add_list:
        dist=0
        for old_ag in base_list:
            dist+= sum(c1 != c2 for c1, c2 in zip(old_ag, new_ag))
        if dist<dist_min:
            dist_min = dist
            ag_name = new_ag
    ag_full_name = dataset.AgSeq[dataset.AgSeq==ag_name].unique()[0]
    return(antigen_add_list.index(ag_full_name))

# Alignments
def aligns_ag(dataset, antigen_base_list, antigen_add_list):
    """
    Counts ax alignment-based distance between an antigene and a set of antigens.
    """
    base_list = list(dataset.AgSeq[dataset.AgSeq.isin(antigen_base_list)].unique())
    add_list = list(dataset.AgSeq[dataset.AgSeq.isin(antigen_add_list)].unique())
    aligns_dist = []
    for new_ag in add_list:
        score = 0
        for old_ag in base_list:
            aligner = Align.PairwiseAligner()
            alignments = aligner.align(old_ag, new_ag)
            score += alignments.score
        aligns_dist.append(score) 
    return(np.argmax(aligns_dist))

def distance_based_iter(dataset: pd.DataFrame, iterations: int, base_antigens_count, training_args: AbAgConvArgs, device: str, random_state: int, option = 'aligns') -> tp.List[float]: 
    """
    Runs a set of distance-based methods for several iterations to select antigens for the model.
    """
    df_antigens = dataset[dataset.total_split=='train'][['AgSeq']].drop_duplicates().reset_index(drop=True)
    antigen_list = list(df_antigens.sample(frac=1.0, random_state=random_state).AgSeq)
    antigen_base_list = antigen_list[:base_antigens_count]
    antigen_add_list = antigen_list[base_antigens_count:]

    torch.manual_seed(random_state)
    net = AbAgConvNet().to(device)
    nets, df_train_ags = train_committee(dataset, [net], antigen_base_list, training_args,
                                            device, random_state, 0, -1)
    net = nets[0]
    for k in tqdm(range(iterations)):
        if option == 'aligns':
            new_antigen = antigen_add_list.pop(aligns_ag(df_antigens, antigen_base_list, antigen_add_list))
        elif option == 'hamming_max':
            new_antigen = antigen_add_list.pop(hamming_opt(df_antigens, antigen_base_list, antigen_add_list))
        elif option == 'hamming_min':
            new_antigen = antigen_add_list.pop(hamming_opt_min_dist(df_antigens, antigen_base_list, antigen_add_list))
        else:
            raise ValueError(f"Unknown option: {method}")
        antigen_base_list.append(new_antigen)
        torch.manual_seed(random_state)
        nets, df_train_ags_iter = train_committee(dataset, [net], antigen_base_list, training_args,
                                            device, random_state, k+1, -1)
        net = nets[0]
        df_train_ags = pd.concat([df_train_ags, df_train_ags_iter], ignore_index=True)
        
    return df_train_ags