from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
import pandas as pd
import typing as tp
import AbAgAL
import importlib
importlib.reload(AbAgAL)
from AbAgAL import *
from tqdm import tqdm


def train(args, model, device, train_loader, validation_loader, optimizer, criterion, epochs, verbose=False):
    """
    Trains the model for a given number of epochs, using early stopping based on validation loss.
    """
    patience = 0
    for epoch in range(epochs):
        if patience == args.patience:
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

        if epoch == 0:
            best_vloss = avg_vloss
        elif avg_vloss < best_vloss:
            best_vloss = avg_vloss
            patience = 0
        else:
            patience += 1


def test(args, model, device, test_loader):
    """
    Tests the model on a test set and calculates the ROC AUC score.
    """
    model.eval()
    all_labels = []
    all_probabilities = []
    with torch.no_grad():
        for i, vdata in enumerate(test_loader):
            vinputs, vlabels = vdata
            predictions = model(vinputs).flatten()
            pred_probabilities = torch.sigmoid(predictions).detach().cpu().numpy()
            all_labels.extend(vlabels.cpu().numpy())
            all_probabilities.extend(pred_probabilities)
    try:
        roc_auc = roc_auc_score(all_labels, all_probabilities)
    except ValueError:
        roc_auc = np.NaN
    return roc_auc


def train_committee(dataset: pd.DataFrame, committee: tp.List[AbAgConvNet], antigen_base_list: tp.List[str],
                     training_args, device, random_state, k, r):
    """
    Trains a set of models (committee) on the dataset, using cross-validation and recording ROC AUC scores.
    """
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
        loaders[split_type] = torch.utils.data.DataLoader(dataset=datasets[split_type], batch_size=batch_size)
    committee_optimizers = [torch.optim.Adam(model.parameters(), eps=training_args.eps, lr=training_args.lr) for model in committee]
    criterion = F.binary_cross_entropy_with_logits
    roc_aucs = {split_type: [] for split_type in ['val', 'test', 'testAB', 'testAG']}
    for i, model in enumerate(committee):
        train(args=training_args, model=model, device=device, train_loader=loaders['train'],
              validation_loader=loaders['val'], optimizer=committee_optimizers[i], criterion=criterion,
              epochs=training_args.epochs, verbose=False)
        for split_type in ['val', 'test', 'testAB', 'testAG']:
            true_labels = datasets[split_type].y.to('cpu').numpy()
            
            roc_auc = test(args=training_args, model = model, device=device, test_loader=loaders[split_type])

            roc_aucs[split_type].append(roc_auc)
    df_train_ags = pd.DataFrame(columns=['AgSeq', 'iter', 'binding_ratio', 'roc_auc_val', 'roc_aucs_test', 
                                         'roc_aucs_testAB', 'roc_aucs_testAG'])
    for ag in antigen_base_list[r:]:
        df_train_ags.loc[len(df_train_ags)] = [ag, k, dataset[dataset.AgSeq == ag].BindClass.mean(), 
                                               sum(roc_aucs['val']) / len(roc_aucs['val']), 
                                               sum(roc_aucs['test']) / len(roc_aucs['test']), 
                                               sum(roc_aucs['testAB']) / len(roc_aucs['testAB']), 
                                               sum(roc_aucs['testAG']) / len(roc_aucs['testAG'])]
    return committee, df_train_ags


def choose_next_antigen(dataset, committee, antigen_base_list, antigen_add_list, iterations, training_args, device, random_state, dis_quantile):
    """
    Chooses the next antigen to add to the base list based on model disagreement (committee variance).
    """
    disagreements = []
    for n in range(len(antigen_add_list)):
        df3 = dataset[dataset.AgSeq == antigen_add_list[n]]
        dataset_new_antigen = AbAgDataset(df=df3, device=device)
        y_new_antigen_outputs = []
        for i in range(len(committee)):
            with torch.no_grad():
                model = committee[i].to(device).eval()
                y_new_antigen_output = torch.sigmoid(model(dataset_new_antigen.x)).flatten()
                y_new_antigen_outputs.append(y_new_antigen_output)
                model = committee[i].to(device).train()
        disagreement = torch.Tensor.cpu(torch.var(torch.stack(y_new_antigen_outputs), dim=0))
        disagreement_quant = pd.DataFrame(disagreement, columns=['disagreement'])
        disagreement_quant = disagreement_quant[
            disagreement_quant.disagreement >= np.quantile(disagreement_quant.disagreement, dis_quantile)]
        disagreements.append(disagreement_quant.mean().item())
    new_antigen = antigen_add_list.pop(np.argmax(disagreements))
    antigen_base_list.append(new_antigen)
    return antigen_base_list, antigen_add_list


def query_by_committee_iter(dataset, committee_size, iterations, base_antigens_count, training_args, device, random_state, dis_quantile):
    """
    Runs a query-by-committee approach for several iterations to select antigens for the model.
    """
    df_antigens = dataset[dataset.total_split=='train'][['AgSeq']].drop_duplicates().reset_index(drop=True)
    antigen_list = list(df_antigens.sample(frac=1.0, random_state=random_state).AgSeq)
    antigen_base_list = antigen_list[:base_antigens_count]
    antigen_add_list = antigen_list[base_antigens_count:]

    committee = [AbAgConvNet().to(device) for _ in range(committee_size)]
    committee, df_train_ags = train_committee(
        dataset, committee, antigen_base_list, training_args, device, random_state, 0, 0)
    for k in tqdm(range(iterations)):
        memory_usage = torch.cuda.memory_allocated() / 1024**3

        antigen_base_list, antigen_add_list = choose_next_antigen(
            dataset, committee, antigen_base_list, antigen_add_list, iterations, training_args, device, random_state, dis_quantile)
        committee, df_train_ags_iter = train_committee(
            dataset, committee, antigen_base_list, training_args, device, random_state, k+1, -1)
        df_train_ags = pd.concat([df_train_ags, df_train_ags_iter], ignore_index=True)
    return df_train_ags


def random_iter(dataset, committee_size, iterations, base_antigens_count, training_args, device, random_state):
    """
    Randomly selects antigens for training iterations, without considering disagreement.
    """
    df_antigens = dataset[dataset.total_split=='train'][['AgSeq']].drop_duplicates().reset_index(drop=True)
    antigen_list = list(df_antigens.sample(frac=1.0, random_state=random_state).AgSeq)
    antigen_base_list = antigen_list[:base_antigens_count]
    antigen_add_list = antigen_list[base_antigens_count:]

    committee = [AbAgConvNet().to(device) for _ in range(committee_size)]
    committee, df_train_ags = train_committee(
        dataset, committee, antigen_base_list, training_args, device, random_state, 0, 0)
    np.random.seed(seed=random_state)
    for k in tqdm(range(iterations)):
        memory_usage = torch.cuda.memory_allocated() / 1024**3

        new_antigen = antigen_add_list.pop(np.random.randint(len(antigen_add_list)))
        antigen_base_list.append(new_antigen)
        committee, df_train_ags_iter = train_committee(dataset, committee, antigen_base_list, 
                                                         training_args, device, random_state, k+1, -1)
        df_train_ags = pd.concat([df_train_ags, df_train_ags_iter], ignore_index=True)
    return df_train_ags
