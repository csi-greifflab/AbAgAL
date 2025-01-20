from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
import pandas as pd
import typing as tp
import Levenshtein
from fastDamerauLevenshtein import damerauLevenshtein
from scipy.spatial import distance
from Bio import Align

import AbAgAL
import importlib

importlib.reload(AbAgAL)
from AbAgAL import *


class AbAgConvArgs:
    def __init__(self):
        self.train_batch_size = 64
        self.val_batch_size = 1024
        self.test_batch_size = 1024
        self.epochs = 5
        self.eps = 1e-07
        self.lr = 1e-02
        self.gamma = 0.7
        self.no_cuda = False
        self.no_mps = False
        self.dry_run = False
        self.seed = 0
        self.log_interval = 10
        self.save_model = False


def train(args, model, device, train_loader, validation_loader, optimizer, criterion, epochs, verbose=False):
    patience = 0
    for epoch in range(epochs):
#     for epoch in range(1):
        if patience == 3:
            break
        model.train(True)
        running_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.flatten(), target.float())
            loss.backward()
            running_loss += loss.item()
            optimizer.step()
            if verbose:
                if batch_idx % args.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                                   len(train_loader.dataset),
                                                                                   100. * batch_idx / len(train_loader),
                                                                                   loss.item()))
        avg_loss = running_loss / (batch_idx + 1)
        running_vloss = 0.0
        model.eval()
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels = vdata
                voutputs = model(vinputs)
                vloss = criterion(voutputs.flatten(), vlabels.float())
                running_vloss += vloss
        avg_vloss = running_vloss / (i + 1)
        print('Epoch: {}, LOSS train {} valid {}'.format(epoch + 1, avg_loss, avg_vloss))
        if epoch == 0:
            best_vloss = avg_vloss
        elif avg_vloss < best_vloss:
            best_vloss = avg_vloss
            patience = 0
        else:
            patience += 1


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    output_tensor = torch.ones(test_loader.dataset.df.shape[0]).to(device) * -1
    target_tensor = torch.ones(test_loader.dataset.df.shape[0]).to(device) * -1
    i = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            output_tensor[i:i + len(target)] = output.view(-1)
            target_tensor[i:i + len(target)] = target
            i += len(target)
            test_loss = F.binary_cross_entropy_with_logits(output.flatten(), target.float(), reduction='sum').item()
    model.train()
    test_loss /= len(test_loader.dataset)
    fpr, tpr, thresholds = roc_curve(target_tensor.to('cpu').numpy(), torch.sigmoid(output_tensor).to('cpu').numpy())
    roc_auc = auc(fpr, tpr)
    print('\nTest set: Average loss: {:.4f}, ROC AUC: {:.3f}\n'.format(test_loss, roc_auc))
    return roc_auc


def model_train(dataset: pd.DataFrame, model: AbAgConvNet, antigen_base_list: tp.List[str],
                     training_args, device, random_state):
    
    df1, df2 = train_test_split(dataset[dataset.AgSeq.isin(antigen_base_list)], test_size=0.2,
                                random_state=random_state)
    print(antigen_base_list)
    dataset_train = AbAgDataset(df=df1, device=device)
    train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=training_args.train_batch_size)
    dataset_val = AbAgDataset(df=df2, device=device)
    validation_loader = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=training_args.val_batch_size)
    model_optimizer = torch.optim.Adam(model.parameters(), eps=training_args.eps, lr=training_args.lr)
    criterion = F.binary_cross_entropy_with_logits
   
    train(args=training_args, model=model, device=device, train_loader=train_loader,
              validation_loader=validation_loader, optimizer=model_optimizer, criterion=criterion,
              epochs=100, verbose=False)
    
    y_val_true = dataset_val.y.to('cpu').numpy()
        
    with torch.no_grad(): 
        model = model.eval()
        y_val_output = model(dataset_val.x).flatten()
        model = model.train()
        
    y_val_pred = torch.sigmoid(torch.Tensor.cpu(y_val_output)).detach().numpy()
    roc_auc = roc_auc_score(y_val_true, y_val_pred)
    print('Model, ROC AUC: {:.3f}\n'.format(roc_auc))

    return model, roc_auc

############################################################################################################################################
# Random
def random(dataset: pd.DataFrame, iterations: int, base_antigens_count,
                training_args: AbAgConvArgs, device: str, random_state: int) -> tp.List[float]: 

    df_antigens = dataset[['AgSeq']].drop_duplicates().reset_index(drop=True)
    antigen_list = list(df_antigens.sample(frac=1.0, random_state=random_state).AgSeq)[:100]
    antigen_base_list = antigen_list[:base_antigens_count]
    
    antigen_add_list = antigen_list[base_antigens_count:]
    training_args = AbAgConvArgs()
    torch.manual_seed(random_state)
    net = AbAgConvNet().to(device)
    net, roc_auc = model_train(dataset, net, antigen_base_list, training_args,
                                            device, random_state)
    roc_aucs = [roc_auc]
    df_train_ags = pd.DataFrame(columns = ['AgSeq', 'iter', 'binding_ratio', 'roc_auc'])
    for ag in antigen_base_list:
        df_train_ags.loc[len(df_train_ags)] = [ag, 0, dataset[dataset.AgSeq==ag].BindClass.mean(), roc_auc]

    for k in range(iterations):
        new_antigen = antigen_add_list.pop(np.random.randint(len(antigen_add_list)))
        antigen_base_list.append(new_antigen)
        
        torch.manual_seed(random_state)
        net, roc_auc_iter = model_train(dataset, net, antigen_base_list, training_args,
                                            device, random_state)
        roc_aucs += [roc_auc_iter]
        df_train_ags.loc[len(df_train_ags)] = [new_antigen, k+1, dataset[dataset.AgSeq==new_antigen].BindClass.mean(), roc_auc_iter]
    return roc_aucs, df_train_ags

# Gradient
def gradient(dataset: pd.DataFrame, iterations: int, base_antigens_count,
                training_args: AbAgConvArgs, device: str, random_state: int, threshold = 0.5) -> tp.List[float]: 

    df_antigens = dataset[['AgSeq']].drop_duplicates().reset_index(drop=True)
    antigen_list = list(df_antigens.sample(frac=1.0, random_state=random_state).AgSeq)[:100]
    antigen_base_list = antigen_list[:base_antigens_count]
    
    antigen_add_list = antigen_list[base_antigens_count:]
    training_args = AbAgConvArgs()
    torch.manual_seed(random_state)
    net = AbAgConvNet().to(device)
    net, roc_auc = model_train(dataset, net, antigen_base_list, training_args,
                                            device, random_state)
    roc_aucs = [roc_auc]
    df_train_ags = pd.DataFrame(columns = ['AgSeq', 'iter', 'binding_ratio', 'roc_auc'])
    for ag in antigen_base_list:
        df_train_ags.loc[len(df_train_ags)] = [ag, 0, dataset[dataset.AgSeq==ag].BindClass.mean(), roc_auc]

    criterion = F.binary_cross_entropy_with_logits
    
    for k in range(iterations):
        gradients = []
        for n in range(len(antigen_add_list)):
            model = net.eval()
            df3 = dataset[dataset.AgSeq == antigen_add_list[n]]
            
            if df3.shape[0] > 100:
                df3 = df3.sample(n=100, random_state=random_state)
                
            dataset_new_antigen = AbAgDataset(df=df3, device=device)
            loader = torch.utils.data.DataLoader(dataset=dataset_new_antigen, batch_size=training_args.train_batch_size)
            optimizer = torch.optim.Adam(model.parameters(), eps=training_args.eps, lr=training_args.lr)
            grad = 0
            
            for data, _ in loader:
                optimizer.zero_grad()
                out = model(data).flatten()
                # preds = torch.sigmoid(torch.Tensor.cpu(out))
                preds = torch.sigmoid(out)
                target = (preds > threshold).float().detach()
                loss = criterion(out.flatten(), target, reduction='sum')
                loss.backward()
                batch_grad_t = model.fc2.weight.grad.detach().cpu().numpy()
                batch_grad = abs(batch_grad_t)
                grad += sum(sum(batch_grad))
                
            gradients.append(grad)
        
        new_antigen = antigen_add_list.pop(np.argmax(gradients))
        antigen_base_list.append(new_antigen)
        
        torch.manual_seed(random_state)
        net, roc_auc_iter = model_train(dataset, net, antigen_base_list, training_args,
                                            device, random_state)
        roc_aucs += [roc_auc_iter]
        df_train_ags.loc[len(df_train_ags)] = [new_antigen, k+1, dataset[dataset.AgSeq==new_antigen].BindClass.mean(), roc_auc_iter]
    return roc_aucs, df_train_ags

############################################################################################################################################
# Alignments
def aligns_ag(dataset, antigen_base_list, antigen_add_list):
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


def aligns(dataset: pd.DataFrame, iterations: int, base_antigens_count,
                training_args: AbAgConvArgs, device: str, random_state: int) -> tp.List[float]: 

    df_antigens = dataset[['AgSeq']].drop_duplicates().reset_index(drop=True)
    antigen_list = list(df_antigens.sample(frac=1.0, random_state=random_state).AgSeq)[:100]
    antigen_base_list = antigen_list[:base_antigens_count]
    
    antigen_add_list = antigen_list[base_antigens_count:]
    training_args = AbAgConvArgs()
    torch.manual_seed(random_state)
    net = AbAgConvNet().to(device)
    net, roc_auc = model_train(dataset, net, antigen_base_list, training_args,
                                            device, random_state)
    roc_aucs = [roc_auc]
    df_train_ags = pd.DataFrame(columns = ['AgSeq', 'iter', 'binding_ratio', 'roc_auc'])
    for ag in antigen_base_list:
        df_train_ags.loc[len(df_train_ags)] = [ag, 0, dataset[dataset.AgSeq==ag].BindClass.mean(), roc_auc]
   
    for k in range(iterations):
        new_antigen = antigen_add_list.pop(aligns_ag(dataset, antigen_base_list, antigen_add_list))
        antigen_base_list.append(new_antigen)
        torch.manual_seed(random_state)
        net, roc_auc_iter = model_train(dataset, net, antigen_base_list, training_args,
                                            device, random_state)
        roc_aucs += [roc_auc_iter]
        df_train_ags.loc[len(df_train_ags)] = [new_antigen, k+1, dataset[dataset.AgSeq==new_antigen].BindClass.mean(), roc_auc_iter]
        
    return roc_aucs, df_train_ags

############################################################################################################################################
# Hamming distance
def hamming_opt(dataset, antigen_base_list, antigen_add_list):
    
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

def hamming(dataset: pd.DataFrame, iterations: int, base_antigens_count,
                training_args: AbAgConvArgs, device: str, random_state: int) -> tp.List[float]: 

    df_antigens = dataset[['AgSeq']].drop_duplicates().reset_index(drop=True)
    antigen_list = list(df_antigens.sample(frac=1.0, random_state=random_state).AgSeq)[:100]
    antigen_base_list = antigen_list[:base_antigens_count]
    
    antigen_add_list = antigen_list[base_antigens_count:]
    training_args = AbAgConvArgs()
    torch.manual_seed(random_state)
    net = AbAgConvNet()
    net, roc_auc = model_train(dataset, net, antigen_base_list, training_args,
                                            device, random_state)
    roc_aucs = [roc_auc]
    df_train_ags = pd.DataFrame(columns = ['AgSeq', 'iter', 'binding_ratio', 'roc_auc'])
    for ag in antigen_base_list:
        df_train_ags.loc[len(df_train_ags)] = [ag, 0, dataset[dataset.AgSeq==ag].BindClass.mean(), roc_auc]

    for k in range(iterations):
        new_antigen = antigen_add_list.pop(hamming_opt(dataset, antigen_base_list, antigen_add_list))
        antigen_base_list.append(new_antigen)
        torch.manual_seed(random_state)
        net, roc_auc_iter = model_train(dataset, net, antigen_base_list, training_args,
                                            device, random_state)
        roc_aucs += [roc_auc_iter]
        df_train_ags.loc[len(df_train_ags)] = [new_antigen, k+1, dataset[dataset.AgSeq==new_antigen].BindClass.mean(), roc_auc_iter]
        
    return roc_aucs, df_train_ags

def hamming_min(dataset: pd.DataFrame, iterations: int, base_antigens_count,
                training_args: AbAgConvArgs, device: str, random_state: int) -> tp.List[float]: 

    df_antigens = dataset[['AgSeq']].drop_duplicates().reset_index(drop=True)
    antigen_list = list(df_antigens.sample(frac=1.0, random_state=random_state).AgSeq)[:100]
    antigen_base_list = antigen_list[:base_antigens_count]
    
    antigen_add_list = antigen_list[base_antigens_count:]
    training_args = AbAgConvArgs()
    torch.manual_seed(random_state)
    net = AbAgConvNet()
    net, roc_auc = model_train(dataset, net, antigen_base_list, training_args,
                                            device, random_state)
    roc_aucs = [roc_auc]
    df_train_ags = pd.DataFrame(columns = ['AgSeq', 'iter', 'binding_ratio', 'roc_auc'])
    for ag in antigen_base_list:
        df_train_ags.loc[len(df_train_ags)] = [ag, 0, dataset[dataset.AgSeq==ag].BindClass.mean(), roc_auc]

    for k in range(iterations):
        new_antigen = antigen_add_list.pop(hamming_opt_min_dist(dataset, antigen_base_list, antigen_add_list))
        antigen_base_list.append(new_antigen)
        torch.manual_seed(random_state)
        net, roc_auc_iter = model_train(dataset, net, antigen_base_list, training_args,
                                            device, random_state)
        roc_aucs += [roc_auc_iter]
        df_train_ags.loc[len(df_train_ags)] = [new_antigen, k+1, dataset[dataset.AgSeq==new_antigen].BindClass.mean(), roc_auc_iter]
        
    return roc_aucs, df_train_ags

############################################################################################################################################
# Gradient 0-1 
def gradient1(dataset: pd.DataFrame, iterations: int, base_antigens_count,
                training_args: AbAgConvArgs, device: str, random_state: int) -> tp.List[float]: 

    df_antigens = dataset[['AgSeq']].drop_duplicates().reset_index(drop=True)
    antigen_list = list(df_antigens.sample(frac=1.0, random_state=random_state).AgSeq)[:100]
    antigen_base_list = antigen_list[:base_antigens_count]
    
    antigen_add_list = antigen_list[base_antigens_count:]
    training_args = AbAgConvArgs()
    torch.manual_seed(random_state)
    net = AbAgConvNet().to(device)
    net, roc_auc = model_train(dataset, net, antigen_base_list, training_args,
                                            device, random_state)
    roc_aucs = [roc_auc]
    df_train_ags = pd.DataFrame(columns = ['AgSeq', 'iter', 'binding_ratio', 'roc_auc'])
    for ag in antigen_base_list:
        df_train_ags.loc[len(df_train_ags)] = [ag, 0, dataset[dataset.AgSeq==ag].BindClass.mean(), roc_auc]

    criterion = F.binary_cross_entropy_with_logits
    
    for k in range(iterations):
        gradients = []
        for n in range(len(antigen_add_list)):
            model = net.eval()
            df3 = dataset[dataset.AgSeq == antigen_add_list[n]]
            
            if df3.shape[0] > 100:
                df3 = df3.sample(n=100, random_state=random_state)
                
            dataset_new_antigen = AbAgDataset(df=df3, device=device)
            loader = torch.utils.data.DataLoader(dataset=dataset_new_antigen, batch_size=training_args.train_batch_size)
            optimizer = torch.optim.Adam(model.parameters(), eps=training_args.eps, lr=training_args.lr)
            grad = 0
            
            for data, _ in loader:
                optimizer.zero_grad()
                out = model(data).flatten()
                preds0 = torch.zeros(len(out))
                loss0 = criterion(out.flatten(), preds0, reduction='sum')
                loss0.backward()
                batch_grad_t0 = model.fc2.weight.grad.detach().cpu().numpy()
                
                optimizer.zero_grad()
                out = model(data).flatten()
                preds1 = torch.zeros(len(out))
                loss1 = criterion(out.flatten(), preds1, reduction='sum')
                loss1.backward()
                batch_grad_t1 = model.fc2.weight.grad.detach().cpu().numpy()
                grad += distance.euclidean(batch_grad_t0[0], batch_grad_t1[0])
                
            gradients.append(grad)
        
        new_antigen = antigen_add_list.pop(np.argmax(gradients))
        antigen_base_list.append(new_antigen)
        
        torch.manual_seed(random_state)
        net, roc_auc_iter = model_train(dataset, net, antigen_base_list, training_args,
                                            device, random_state)
        roc_aucs += [roc_auc_iter]
        df_train_ags.loc[len(df_train_ags)] = [new_antigen, k+1, dataset[dataset.AgSeq==new_antigen].BindClass.mean(), roc_auc_iter]
    return roc_aucs, df_train_ags

############################################################################################################################################
# Gradient on confounding labels
def gradient2(dataset: pd.DataFrame, iterations: int, base_antigens_count,
                training_args: AbAgConvArgs, device: str, random_state: int) -> tp.List[float]: 

    df_antigens = dataset[['AgSeq']].drop_duplicates().reset_index(drop=True)
    antigen_list = list(df_antigens.sample(frac=1.0, random_state=random_state).AgSeq)[:100]
    antigen_base_list = antigen_list[:base_antigens_count]
    
    antigen_add_list = antigen_list[base_antigens_count:]
    training_args = AbAgConvArgs()
    torch.manual_seed(random_state)
    net = AbAgConvNet().to(device)
    net, roc_auc = model_train(dataset, net, antigen_base_list, training_args,
                                            device, random_state)
    roc_aucs = [roc_auc]
    df_train_ags = pd.DataFrame(columns = ['AgSeq', 'iter', 'binding_ratio', 'roc_auc'])
    for ag in antigen_base_list:
        df_train_ags.loc[len(df_train_ags)] = [ag, 0, dataset[dataset.AgSeq==ag].BindClass.mean(), roc_auc]

    criterion = F.binary_cross_entropy_with_logits
    
    for k in range(iterations):
        gradients = []
        for n in range(len(antigen_add_list)):
            model = net.eval()
            df3 = dataset[dataset.AgSeq == antigen_add_list[n]]
            
            if df3.shape[0] > 100:
                df3 = df3.sample(n=100, random_state=random_state)
                
            dataset_new_antigen = AbAgDataset(df=df3, device=device)
            loader = torch.utils.data.DataLoader(dataset=dataset_new_antigen, batch_size=training_args.train_batch_size)
            optimizer = torch.optim.Adam(model.parameters(), eps=training_args.eps, lr=training_args.lr)
            grad = 0
            
            for data, _ in loader:
                optimizer.zero_grad()
                out = model(data).flatten()
                preds = torch.ones(len(out))*(-1)
                loss = criterion(out.flatten(), preds, reduction='sum')
                loss.backward()
                batch_grad_t = model.fc2.weight.grad.detach().cpu().numpy()
                grad += sum(sum(abs(batch_grad_t)))
                
            gradients.append(grad)
        
        new_antigen = antigen_add_list.pop(np.argmax(gradients))
        antigen_base_list.append(new_antigen)

        torch.manual_seed(random_state)
        net, roc_auc_iter = model_train(dataset, net, antigen_base_list, training_args,
                                            device, random_state)
        roc_aucs += [roc_auc_iter]
        df_train_ags.loc[len(df_train_ags)] = [new_antigen, k+1, dataset[dataset.AgSeq==new_antigen].BindClass.mean(), roc_auc_iter]
    return roc_aucs, df_train_ags

############################################################################################################################################
# Gradient with respect to the input
def gradient3(dataset: pd.DataFrame, iterations: int, base_antigens_count,
                training_args: AbAgConvArgs, device: str, random_state: int) -> tp.List[float]: 

    df_antigens = dataset[['AgSeq']].drop_duplicates().reset_index(drop=True)
    antigen_list = list(df_antigens.sample(frac=1.0, random_state=random_state).AgSeq)[:100]
    antigen_base_list = antigen_list[:base_antigens_count]
    
    antigen_add_list = antigen_list[base_antigens_count:]
    training_args = AbAgConvArgs()
    
    torch.manual_seed(random_state)
    net = AbAgConvNet().to(device)
    net, roc_auc = model_train(dataset, net, antigen_base_list, training_args,
                                            device, random_state)
    torch.manual_seed(random_state)
    net_grad = AbAgConvNet_grad().to(device)
    net_grad, roc_auc_grad = model_train(dataset, net_grad, antigen_base_list, training_args,
                                            device, random_state)
    roc_aucs = [roc_auc]
    df_train_ags = pd.DataFrame(columns = ['AgSeq', 'iter', 'binding_ratio', 'roc_auc'])
    for ag in antigen_base_list:
        df_train_ags.loc[len(df_train_ags)] = [ag, 0, dataset[dataset.AgSeq==ag].BindClass.mean(), roc_auc]

    criterion = F.binary_cross_entropy_with_logits
    
    for k in range(iterations):
        gradients = []
        for n in range(len(antigen_add_list)):
            model_grad = net_grad.eval()
            df3 = dataset[dataset.AgSeq == antigen_add_list[n]]
            
            if df3.shape[0] > 100:
                df3 = df3.sample(n=100, random_state=random_state)
                
            dataset_new_antigen = AbAgDataset(df=df3, device=device)
            loader = torch.utils.data.DataLoader(dataset=dataset_new_antigen, batch_size=training_args.train_batch_size)
            optimizer_grad = torch.optim.Adam(model_grad.parameters(), eps=training_args.eps, lr=training_args.lr)
            grad = 0
            for data, _ in loader:
                optimizer_grad.zero_grad()
                out = model_grad(data).flatten()
                preds0 = torch.zeros(len(out))
                loss0 = criterion(out.flatten(), preds0, reduction='sum')
                loss0.backward()
                batch_grad_t0 = model_grad.embetter.grad.detach().cpu().numpy()
                
                optimizer_grad.zero_grad()
                out = model_grad(data).flatten()
                preds1 = torch.zeros(len(out))
                loss1 = criterion(out.flatten(), preds1, reduction='sum')
                loss1.backward()
                batch_grad_t1 = model_grad.embetter.grad.detach().cpu().numpy()
                for grad_n in range(model_grad.embetter.grad.shape[0]):
                    grad += distance.euclidean( sum(batch_grad_t0[grad_n].T), sum(batch_grad_t1[grad_n].T))
                
            gradients.append(grad)
        
        new_antigen = antigen_add_list.pop(np.argmax(gradients))
        antigen_base_list.append(new_antigen)

        torch.manual_seed(random_state)
        net, roc_auc_iter = model_train(dataset, net, antigen_base_list, training_args,
                                            device, random_state)
        roc_aucs += [roc_auc_iter]
        df_train_ags.loc[len(df_train_ags)] = [new_antigen, k+1, dataset[dataset.AgSeq==new_antigen].BindClass.mean(), roc_auc_iter]
    return roc_aucs, df_train_ags

############################################################################################################################################ Gradient on model
def gradient4(dataset: pd.DataFrame, iterations: int, base_antigens_count,
                training_args: AbAgConvArgs, device: str, random_state: int) -> tp.List[float]: 

    df_antigens = dataset[['AgSeq']].drop_duplicates().reset_index(drop=True)
    antigen_list = list(df_antigens.sample(frac=1.0, random_state=random_state).AgSeq)[:100]
    antigen_base_list = antigen_list[:base_antigens_count]
    
    antigen_add_list = antigen_list[base_antigens_count:]
    training_args = AbAgConvArgs()
    torch.manual_seed(random_state)
    net = AbAgConvNet().to(device)
    net, roc_auc = model_train(dataset, net, antigen_base_list, training_args,
                                            device, random_state)
    roc_aucs = [roc_auc]
    df_train_ags = pd.DataFrame(columns = ['AgSeq', 'iter', 'binding_ratio', 'roc_auc'])
    for ag in antigen_base_list:
        df_train_ags.loc[len(df_train_ags)] = [ag, 0, dataset[dataset.AgSeq==ag].BindClass.mean(), roc_auc]

    criterion = F.binary_cross_entropy_with_logits
    
    for k in range(iterations):
        gradients = []
        for n in range(len(antigen_add_list)):
            model = net.eval()
            df3 = dataset[dataset.AgSeq == antigen_add_list[n]]
            
            if df3.shape[0] > 100:
                df3 = df3.sample(n=100, random_state=random_state)
                
            dataset_new_antigen = AbAgDataset(df=df3, device=device)
            loader = torch.utils.data.DataLoader(dataset=dataset_new_antigen, batch_size=training_args.train_batch_size)
            optimizer = torch.optim.Adam(model.parameters(), eps=training_args.eps, lr=training_args.lr)
            grad = 0
            
            for data, _ in loader:
                for id, item in enumerate(data):
                    optimizer.zero_grad()
                    out = model(item)
                    out.backward()
                    batch_grad_t = model.fc2.weight.grad.detach().cpu().numpy()
                    grad += sum(sum(abs(batch_grad_t)))
                
            gradients.append(grad)
        
        new_antigen = antigen_add_list.pop(np.argmax(gradients))
        antigen_base_list.append(new_antigen)

        torch.manual_seed(random_state)
        net, roc_auc_iter = model_train(dataset, net, antigen_base_list, training_args,
                                            device, random_state)
        roc_aucs += [roc_auc_iter]
        df_train_ags.loc[len(df_train_ags)] = [new_antigen, k+1, dataset[dataset.AgSeq==new_antigen].BindClass.mean(), roc_auc_iter]
    return roc_aucs, df_train_ags

############################################################################################################################################ Gradient on model
def gradient4(dataset: pd.DataFrame, iterations: int, base_antigens_count,
                training_args: AbAgConvArgs, device: str, random_state: int) -> tp.List[float]: 

    df_antigens = dataset[['AgSeq']].drop_duplicates().reset_index(drop=True)
    antigen_list = list(df_antigens.sample(frac=1.0, random_state=random_state).AgSeq)[:100]
    antigen_base_list = antigen_list[:base_antigens_count]
    
    antigen_add_list = antigen_list[base_antigens_count:]
    training_args = AbAgConvArgs()
    torch.manual_seed(random_state)
    net = AbAgConvNet().to(device)
    net, roc_auc = model_train(dataset, net, antigen_base_list, training_args,
                                            device, random_state)
    roc_aucs = [roc_auc]
    df_train_ags = pd.DataFrame(columns = ['AgSeq', 'iter', 'binding_ratio', 'roc_auc'])
    for ag in antigen_base_list:
        df_train_ags.loc[len(df_train_ags)] = [ag, 0, dataset[dataset.AgSeq==ag].BindClass.mean(), roc_auc]

    #criterion = F.binary_cross_entropy_with_logits
    
    for k in range(iterations):
        gradients = []
        for n in range(len(antigen_add_list)):
            model = net.eval()
            df3 = dataset[dataset.AgSeq == antigen_add_list[n]]
            
            if df3.shape[0] > 100:
                df3 = df3.sample(n=100, random_state=random_state)
                
            dataset_new_antigen = AbAgDataset(df=df3, device=device)
            loader = torch.utils.data.DataLoader(dataset=dataset_new_antigen, batch_size=training_args.train_batch_size)
            optimizer = torch.optim.Adam(model.parameters(), eps=training_args.eps, lr=training_args.lr)
            grad = 0
            
            for data, _ in loader:
                for id, item in enumerate(data):
                    optimizer.zero_grad()
                    out = model(item)
                    #out = sum(out)*(-1)
                    out.backward()
                    batch_grad_t = model.fc2.weight.grad.detach().cpu().numpy()
                    grad += sum(sum(abs(batch_grad_t)))
                
            gradients.append(grad)
        
        new_antigen = antigen_add_list.pop(np.argmax(gradients))
        antigen_base_list.append(new_antigen)

        torch.manual_seed(random_state)
        net, roc_auc_iter = model_train(dataset, net, antigen_base_list, training_args,
                                            device, random_state)
        roc_aucs += [roc_auc_iter]
        df_train_ags.loc[len(df_train_ags)] = [new_antigen, k+1, dataset[dataset.AgSeq==new_antigen].BindClass.mean(), roc_auc_iter]
    return roc_aucs, df_train_ags
