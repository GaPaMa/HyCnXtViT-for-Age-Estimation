import os
import torch
import pytorch_lightning as pl
import pandas as pd
from yacs.config import CfgNode as CN
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from data import DATASET, load_transforms
from model.model import MODEL
import torchmetrics.functional as metrics
from datetime import datetime
# Get just the date
today_date = datetime.now().strftime("%d_%m_%Y")
import time


### CONFIGS ###

config='ConvNeXT'
### MODEL ###
CFG = CN()

CFG.Trainer = CN()
# CFG.Trainer.checkpoint_file = False
CFG.Trainer.num_epochs = 500
CFG.Trainer.batch_size = 256 #64
# CFG.Trainer.num_workers = os.cpu_count() - 6
CFG.Trainer.num_workers = 8
CFG.Trainer.accelerator = 'gpu'
CFG.Trainer.devices = [2 ,3] # not use 0 and 1
CFG.Trainer.loss_func = 'MAE' # 'MAE', 'MSE', 'Adaptive', 'Huber' --> Adaptive
CFG.Trainer.num_class = 1
CFG.Trainer.sigma = 2 # used for Adaptive loss
CFG.Trainer.lr = 1e-4 #OneCycleLR_1 1e-4 5e-5
CFG.Trainer.lr_patience = 3 #5
CFG.Trainer.lr_min = 1e-6 #1e-8
# CFG.Trainer.scheduler = False
# CFG.Trainer.scheduler = "OneCycleLR"
CFG.Trainer.scheduler = "WarmupCosineSchedule"
CFG.Trainer.scheduler_interval = "step"
CFG.Trainer.scheduler_monitor = "Loss/Val"
CFG.Trainer.scheduler_factor = 0.1 
CFG.Trainer.scheduler_warmup_steps = 0
CFG.Trainer.scheduler_total_steps = 0

CFG.Model = CN()
CFG.Model.name = "ConvNeXT"
CFG.Model.TPS = True
CFG.Model.num_fiducial = 20


CFG.ConvNeXT = CN()
CFG.ConvNeXT.pretrained = "/lscratch/gmaroun/Projects/ConvNeXT_Vision_Transformer/weights/convnext_tiny_22k_224.pth"
# CFG.ConvNeXT.pretrained = False
CFG.ConvNeXT.in_chans=3
CFG.ConvNeXT.num_classes= CFG.Trainer.num_class
CFG.ConvNeXT.depths=[3, 3, 9, 3] #[3, 3, 27, 3]
CFG.ConvNeXT.dims=[96, 192, 384, 768]
CFG.ConvNeXT.drop_path_rate=0. 
CFG.ConvNeXT.layer_scale_init_value=1e-6
CFG.ConvNeXT.head_init_scale=1.
CFG.ConvNeXT.fc = 256 # None, 32, 64, 128, 192, 256



datasets = ["AFAD", "MORPH2", "CACD", "imdb-clean-1024"] 


# for tm in trained_model:
for dataset in datasets:

    # CFG.Trainer.checkpoint_file = False
    CFG.Trainer.checkpoint_file = './checkpoints/checkpoints_ConvNeXT/MORPH2_ConvNeXT_FC2_256_epoch_499_mae_2.290440150948744_2ndRound.ckpt'

    ### DATA ###
    path = "<path>/Datasets"

    tf_train, tf_test = load_transforms()
    train = DATASET(path, dataset, "Train", tf_train)
    valid = DATASET(path, dataset, "Val", tf_test)
    test = DATASET(path, dataset, "Test", tf_test)

    CFG.Trainer.scheduler_total_steps = round(CFG.Trainer.num_epochs * (len(train) / CFG.Trainer.batch_size))
    CFG.Trainer.scheduler_warmup_steps = round(CFG.Trainer.scheduler_factor * CFG.Trainer.scheduler_total_steps)

    train_loader = DataLoader(dataset=train, batch_size=CFG.Trainer.batch_size, shuffle=True, num_workers=CFG.Trainer.num_workers, pin_memory=True)
    valid_loader = DataLoader(dataset=valid, batch_size=CFG.Trainer.batch_size, shuffle=False, num_workers=CFG.Trainer.num_workers, pin_memory=True)
    test_loader = DataLoader(dataset=test, batch_size=CFG.Trainer.batch_size, shuffle=False, num_workers=CFG.Trainer.num_workers, pin_memory=True)

    ### Result saving 
    if CFG.ConvNeXT.fc == None:
        weights_name = f"{dataset}_ConvNeXT_FC1_{CFG.Trainer.loss_func}_{CFG.Trainer.num_class}"
    else:
        weights_name = f"{dataset}_ConvNeXT_FC2_batch{CFG.Trainer.batch_size}_{CFG.Trainer.loss_func}"

    save_file = f"./results_ConvNeXT/{today_date}/{weights_name}.pkl"
    if not os.path.exists(f"./results_ConvNeXT/{today_date}"):
        os.makedirs(f"./results_ConvNeXT/{today_date}")

    ### CREATE FOLDER ###
    directory = f'./checkpoints/checkpoints_ConvNeXT_{today_date}'
    if not os.path.exists(directory):
        os.makedirs(directory)

    ### Model
    model = MODEL(CFG)

    ### TRAIN/EVAL ###
    loss = '_epoch_{epoch:02d}_mae_{MAE/Val}'
    if CFG.Trainer.loss_func == 'CrossEntropy':
        loss= '_epoch_{epoch:02d}_top1_{TOP1/Val}_top5_{TOP5/Val}'

    trainer = pl.Trainer(accelerator='gpu', 
                         devices=CFG.Trainer.devices, 
                        strategy='ddp',
                        callbacks=[pl.callbacks.ModelCheckpoint(
                                    save_top_k = 1, 
                                    dirpath = directory,
                                    filename = weights_name  + loss,
                                    auto_insert_metric_name=False)],
                        logger=pl.loggers.TensorBoardLogger(f'/data2/gmaroun/Projects/ConvNeXT_Vision_Transformer/logs/{dataset}/{config}_{today_date}',
                                                            name=weights_name, 
                                                            default_hp_metric=False),
                        max_epochs=CFG.Trainer.num_epochs,
                        benchmark=True,
                        precision=32,
                        )

    if CFG.Trainer.checkpoint_file:
        trainer.fit(model, train_loader, valid_loader, ckpt_path=CFG.Trainer.checkpoint_file)
    else:
        trainer.fit(model, train_loader, valid_loader)

    trainer.test(model, dataloaders=test_loader)
    stats = measure_computational_stats(model, test_loader)
    print(stats)
    data = pd.DataFrame({"gt": model.labels_gt, "p": model.labels_p})
    data.to_pickle(save_file)
