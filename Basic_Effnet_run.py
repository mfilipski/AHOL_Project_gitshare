

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%% All the import statements %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


import random
import torch
from torch import nn
import torch.nn as nn
import torch.optim as optim


from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models
from torchsummary import summary

# !pip install efficientnet_pytorch -qq
from efficientnet_pytorch import EfficientNet

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn import model_selection as sk_model_selection
from sklearn.metrics import roc_curve, roc_auc_score

import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from pylab import text
from nnAudio.features import CQT1992v2

import os
import re
import gc
import wandb
import time
from tqdm import tqdm
import pandas as pd
import numpy as np
import seaborn as sns

import glob


# Set
def set_seed(seed=23):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


set_seed()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device available now:', device)

print('done with imports')


class color:
    S = '\033[1m' + '\033[93m'
    E = '\033[0m'



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%% Get the data in and processed %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
train = pd.read_csv('Data/training_labels.csv')
test = pd.read_csv('Data/sample_submission.csv')

def get_train_file_path(image_id):
    return "Data/train/{}/{}/{}/{}.npy".format(
        image_id[0], image_id[1], image_id[2], image_id)

def get_test_file_path(image_id):
    return "Data/test/{}/{}/{}/{}.npy".format(
        image_id[0], image_id[1], image_id[2], image_id)

train['path'] = train['id'].apply(get_train_file_path)
test['path'] = test['id'].apply(get_test_file_path)

# print(train.head())
# print(test.head())


# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# ~~~~~FUNCTIONS~~~~~
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def plot_loss_graph(train_losses, valid_losses, epoch, fold):
    '''Lineplot of the training/validation losses.'''

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 2.5))
    fig.suptitle(f"Fold {fold} | Epoch {epoch}", fontsize=12, y=1.05)
    axes = [ax1, ax2]
    data = [train_losses, valid_losses]
    sns.lineplot(y=train_losses, x=range(len(train_losses)),
                 lw=2.3, ls=":", color=my_colors[3], ax=ax1)
    sns.lineplot(y=valid_losses, x=range(len(valid_losses)),
                 lw=2.3, ls="-", color=my_colors[5], ax=ax2)
    for ax, t, d in zip(axes, ["Train", "Valid"], data):
        ax.set_title(f"{t} Evolution", size=12, weight='bold')
        ax.set_xlabel("Iteration", weight='bold', size=9)
        ax.set_ylabel("Loss", weight='bold', size=9)
        ax.tick_params(labelsize=9)
    plt.show()


def get_auc_score(valid_preds, valid_targets, gpu=True):
    '''Compute ROC AUC score.'''
    if gpu:
        predictions = torch.cat(valid_preds).cpu().detach().numpy().tolist()
    else:
        predictions = torch.cat(valid_preds).detach().numpy().tolist()
    actuals = [int(x) for x in valid_targets]

    roc_auc = roc_auc_score(actuals, predictions)
    return roc_auc



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%% This is the dataloader %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Note: This data loader incorporates a Qtransform.  Probably what slows it down.

class G2Dataset(Dataset):

    def __init__(self, path, features, target=None, test=False, prints=False):
        '''Initiate the arguments & import the numpy file/s.'''
        self.path = path
        self.features = features
        self.target = target
        self.test = test
        self.prints = prints
        self.TRANSFORM = CQT1992v2(sr=2048, fmin=20,
                               fmax=1024, hop_length=32,
                               verbose=False)

    def __len__(self):
        return len(self.path)


    def __transform__(self, np_file):
        '''Transforms the np_file into spectrogram.'''
        # spectrogram = []
        # TRANSFORM = CQT1992v2(sr=2048, fmin=20,
        #                       fmax=1024, hop_length=32,
        #                       verbose=False)
        #
        # # Create an image with 3 channels - for the 3 sites
        # for i in range(3):
        #     waves = np_file[i] / np.max(np_file[i])
        #     waves = torch.from_numpy(waves).float()
        #     channel = TRANSFORM(waves).squeeze().numpy()
        #     spectrogram.append(channel)
        #     # print('spectrogram', spectrogram)
        #     # print(np.shape(spectrogram))

        spectrogram = torch.zeros(3,69,129)
        for i in range(3):
            waves = np_file[i] / np.max(np_file[i])
            waves = torch.from_numpy(waves).float()
            channel = self.TRANSFORM(waves).squeeze()
            spectrogram[i] = channel
            # print('spectrogram', spectrogram)
            # print(np.shape(spectrogram))
            # print(spectrogram.shape)


        # spectrogram = torch.tensor(spectrogram).float()

        if self.prints:
            plt.figure(figsize=(5, 5))
            plot = spectrogram.detach().cpu().numpy()
            plot = np.transpose(plot, (1, 2, 0))
            plt.imshow(plot)
            plt.axis("off")
            plt.show();

        return spectrogram


    def __getitem__(self, i):
        # Load the numpy file
        np_file = np.load(self.path[i])
        # Create the spectrograms
        spectrograms = self.__transform__(np_file)
        # Select the features
        metadata = np.array(self.features.iloc[i].values, dtype=np.float32)

        # Return the images & target if available
        if self.test == False:
            y = torch.tensor(self.target[i], dtype=torch.float)
            return {"spectrogram": spectrograms,
                    "metadata": metadata,
                    "targets": y}
        else:
            return {"spectrogram": spectrograms,
                    "metadata": metadata}
# --------------------------------------------------------------------------------------

# Check the dataloader (just a check)
# *=========================
# Sample
# path = train["path"][:4].values
# features = train.iloc[:4, 3:]
# target = train["target"][:4].values
#
# print('target', target)
#
# # Initiate the Dataset
# dataset = G2Dataset(path=path, target=target, features=features,
#                     test=False, prints=True)
#
# print('dataset', dataset)
#
# # Initiate the Dataloader
# dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
# ""
# # Output of the Dataloader
# for k, data in enumerate(dataloader):
#     spectrograms, features, targets = data.values()
#     print(color.S + f"Batch: {k}" + color.E, "\n" +
#           color.S + "Spectrograms:" + color.E, spectrograms.shape, "\n" +
#           color.S + f"Features:" + color.E, features.shape, "\n" +
#           color.S + "Target:" + color.E, targets, "\n" +
#           "="*50)
# --------------------------------------------------------------------------------------


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%% This is the Model %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


class G2EffNet(nn.Module):

    def __init__(self, no_features, no_neurons=250):
        super().__init__()

        # NN for the spectrogram - out layer = 2560
        self.spectrogram = EfficientNet.from_pretrained('efficientnet-b7')
        self.classification = nn.Sequential(nn.Linear(2560, 1))


    def forward(self, spectrogram, features, prints=False):

        if prints: print(color.S + 'Spectrogram In:' + color.E, spectrogram.shape, '\n' +
                         color.S + 'Features In:' + color.E, features.shape, '\n' +
                         '=' * 40)

        # Spectrogram
        spectrogram = self.spectrogram.extract_features(spectrogram)
        if prints: print(color.S + 'Spectrogram Out:' + color.E, spectrogram.shape)

        spectrogram = F.avg_pool2d(spectrogram, spectrogram.size()[2:]).reshape(-1, 2560)
        if prints: print(color.S + 'Spectrogram Reshaped:' + color.E, spectrogram.shape)

        out = self.classification(spectrogram)

        return torch.sigmoid(out)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# This is a quick run of the model
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ====================================================================================
model_example = G2EffNet(no_features=0, no_neurons=250)

summary(model_example)

# We'll use previous datasets & dataloader

# Sample
path = train["path"][:4].values
features = train.iloc[:4, 3:]
target = train["target"][:4].values

print('target', target)

# Initiate the Dataset
dataset = G2Dataset(path=path, target=target, features=features,
                    test=False, prints=False)

print('dataset', dataset)

# Initiate the Dataloader
dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

# example for 1 batch
for k, data in enumerate(dataloader):
    spectrograms, features, targets = data.values()
    break

# Outputs
out = model_example(spectrograms, features, prints=False)

# Criterion
criterion_example = nn.BCEWithLogitsLoss()
# Unsqueeze(1) from shape=[3] => shape=[3, 1]
loss = criterion_example(out, targets.unsqueeze(1))
print(color.S + 'LOSS:' + color.E, loss.item())


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# This trains the model
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
CONFIG = {'competition': 'g2net', '_wandb_kernel': 'aot'}

def train_effnet(name, epochs, splits, batch_size, no_neurons, lr, weight_decay, sample):

    # === W&B Experiment ===
    s = time.time()
    params = dict(model=name, epochs=epochs, split=splits,
                  batch=batch_size, neurons=no_neurons,
                  lr=lr, weight_decay=weight_decay, sample=sample)
    CONFIG.update(params)
    # run = wandb.init(project='g2net', name=f"effnet_{name}", config=CONFIG, anonymous="allow")


    # === CV Split ===
    df = train.sample(sample, random_state=23)
    cv = StratifiedKFold(n_splits=splits)
    cv_splits = cv.split(X=df, y=df['target'].values)



    for fold, (train_i, valid_i) in enumerate(cv_splits):

        print("~"*25)
        print("~"*8, color.S+f"FOLD {fold}"+color.E, "~"*8)
        print("~"*25)

        train_df = df.iloc[train_i, :]
        # To go quicker through validation
        valid_df = df.iloc[valid_i, :].sample(int(sample*(splits/10)*0.6),
                                              random_state=23)

        # Datasets & Dataloader
        train_dataset = G2Dataset(path=train_df["path"].values, target=train_df["target"].values,
                                  features=train_df.iloc[:, 3:], test=False)
        valid_dataset = G2Dataset(path=valid_df["path"].values, target=valid_df["target"].values,
                                  features=valid_df.iloc[:, 3:], test=False)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

        # Model/ Optimizer/ Criterion/ Scheduler
        model = G2EffNet(no_features=15, no_neurons=no_neurons).to(device)
        optimizer = Adam(model.parameters(), lr=lr,
                         weight_decay=weight_decay, amsgrad=False)
        criterion = nn.BCEWithLogitsLoss()
        # scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='max', verbose=True,
        #                               patience=VAR.patience, factor=VAR.factor)
        scaler = GradScaler()

        # ~~~~~~~~~~~~
        # ~~~ LOOP ~~~
        # ~~~~~~~~~~~~
        BEST_SCORE = 0.0

        for epoch in range(epochs):
            print("="*8, color.S+f"Epoch {epoch}"+color.E, "="*8)

            # === TRAIN ===
            model.train()
            train_losses = []
            for k, data in enumerate(train_loader):
                spectrograms, features, targets = data.values()
                spectrograms, features, targets = spectrograms.to(device), features.to(device), targets.to(device)

                with autocast():
                    out = model(spectrograms, features)
                    loss = criterion(out, targets.unsqueeze(1))
                    train_losses.append(loss.cpu().detach().numpy().tolist())

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            mean_train_loss = np.mean(train_losses)
            print(color.S+"Mean Train Loss:"+color.E, mean_train_loss)
            # wandb.log({"mean_train_loss": np.float(mean_train_loss)}, step=epoch)


            # === EVAL ===
            model.eval()
            valid_losses, valid_preds, valid_targets = [], [], []
            with torch.no_grad():
                for k, data in enumerate(valid_loader):
                    spectrograms, features, targets = data.values()
                    valid_targets.extend(targets.detach().numpy().tolist())
                    spectrograms, features, targets = spectrograms.to(device), features.to(device), targets.to(device)

                    out = model(spectrograms, features)

                    valid_preds.extend(out)
                    loss = criterion(out, targets.unsqueeze(1))
                    valid_losses.append(loss.cpu().detach().numpy().tolist())

            mean_valid_loss = np.mean(valid_losses)
            print(color.S+"Mean Valid Loss:"+color.E, mean_valid_loss)
            # wandb.log({"mean_valid_loss": np.float(mean_valid_loss)}, step=epoch)
#             plot_loss_graph(train_losses, valid_losses, epoch, fold)


            # === UPDATES ===
            roc_auc = get_auc_score(valid_preds, valid_targets, gpu=torch.cuda.is_available())
            print(color.S+"ROC AUC:"+color.E, roc_auc)
            print(color.S + f"Time so far: {round((time.time() - s) / 60, 2)} minutes" + color.E)
            # wandb.log({"roc_auc": np.float(roc_auc)}, step=epoch)

            if roc_auc > BEST_SCORE:
                print('roc_auc', roc_auc, 'better than best score', BEST_SCORE)
                print("! Saving model in fold {} | epoch {} ...".format(fold, epoch), "\n")
                # torch.save(model.state_dict(), f"runs/Baseline_fold_{fold}_auc_{round(roc_auc, 5)}.pt")
#                 model.save(os.path.join(wandb.run.dir, "model.pt"))

                BEST_SCORE = roc_auc


        del model, optimizer, criterion, spectrograms, features, targets
        torch.cuda.empty_cache()
        gc.collect()

#     wandb.finish()
    print(color.S+f"FINAL Time to run: {round((time.time() - s)/60, 2)} minutes"+color.E)


class VAR:
    name = "full"
    splits = 3
    epochs = 2
    # Note = Can't support more than 20 on my cuda
    batch_size = 24
    #     batch_size = 8
    no_neurons = 250
    lr = 0.0001
    weight_decay = 0.000001
    patience = 1
    factor = 0.01
    sample = 100



train_effnet(name=VAR.name, epochs=VAR.epochs, splits=VAR.splits,
             batch_size=VAR.batch_size, no_neurons=VAR.no_neurons, lr=VAR.lr,
             weight_decay=VAR.weight_decay, sample=VAR.sample)