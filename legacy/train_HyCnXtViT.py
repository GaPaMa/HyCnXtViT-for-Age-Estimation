import os
import torch
import pytorch_lightning as pl
import pandas as pd
from yacs.config import CfgNode as CN
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from data import DATASET, load_transforms
from model.model import MODEL
import numpy as np
import random
from tqdm import tqdm
# from lightning.pytorch.strategies import DDPStrategy
from datetime import datetime
# Get just the date
today_date = datetime.now().strftime("%d_%m_%Y")
import time
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from torchcam.methods import GradCAM
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

### MODEL ###

config = 'HyCnXtViT'
CFG = CN()
CFG.Trainer = CN()
CFG.Trainer.checkpoint_file = False

CFG.Trainer.num_epochs = 500
CFG.Trainer.batch_size = 64 #64  #imdb-clean-1024
CFG.Trainer.num_workers = os.cpu_count() - 6
# CFG.Trainer.num_workers = 8
CFG.Trainer.accelerator = 'gpu'
CFG.Trainer.devices = [1, 4] 
CFG.Trainer.loss_func = 'Adaptive' # 'MAE', 'MSE', 'Adaptive', 'Huber', WeightedMSE --> Adaptive, KLDivLoss (no use yet)
CFG.Trainer.num_class = 1
CFG.Trainer.sigma = 2 # used for Adaptive loss
CFG.Trainer.lr = 1e-4 #OneCycleLR_1 1e-4
CFG.Trainer.lr_patience = 5
CFG.Trainer.lr_min = 1e-8
# CFG.Trainer.scheduler = "CosineAnnealingLR" # ReduceLROnPlateau, WarmupCosineSchedule, OneCycleLR, CosineAnnealingLR, StepLR, ExponentialLR, CyclicLR
CFG.Trainer.scheduler_interval = "step"
CFG.Trainer.scheduler_monitor = "Loss/Val"
CFG.Trainer.scheduler_factor = 0.1
CFG.Trainer.scheduler_warmup_steps = 0
CFG.Trainer.scheduler_total_steps = 0


CFG.Model = CN()
CFG.Model.name = "ConvNeXT_Transformer"
CFG.Model.TPS = False
CFG.Model.num_fiducial = 20

CFG.ConvNeXT = CN()
CFG.ConvNeXT.pretrained = False
# CFG.ConvNeXT.weights = "" 
CFG.ConvNeXT.in_chans=3
CFG.ConvNeXT.num_classes = CFG.Trainer.num_class
CFG.ConvNeXT.depths=[3, 3, 27, 3] #imdb-clean-1024
# CFG.ConvNeXT.depths=[3, 3, 9, 3]
CFG.ConvNeXT.dims=[96, 192, 384, 768]
CFG.ConvNeXT.drop_path_rate=0.1
CFG.ConvNeXT.layer_scale_init_value=1e-6
CFG.ConvNeXT.head_init_scale=1.


CFG.Transformer = CN()

CFG.Transformer.backbone = "vit_tiny_patch16_224" #"vit_tiny_patch16_224" vit_tiny_patch16_384
CFG.Transformer.transfer_learning = True
CFG.Transformer.pretrained = True
CFG.Transformer.num_class = CFG.Trainer.num_class 


datasets = ["MORPH2", "AFAD", "CACD", "imdb-clean-1024"]


for dataset in datasets:
    if dataset == "CACD":
        CFG.ConvNeXT.weights = "<path>/checkpoints_ConvNeXT/CACD_ConvNeXT_FC2_128_epoch_99_mae_4.1533247541818.ckpt" 
        CFG.Transformer.weights = "<path>/CACD_Transformer_FC2_vit_tiny_patch16_224_256_batch256_ImgNet_Cosine0.001_epoch_499_mae_4.713215787607605.ckpt"  #CACD
        CFG.Transformer.fc = 256 # None, 32, 64, 128, 192, 256 
        CFG.ConvNeXT.fc = 128 # None, 32, 64, 128, 192, 256
    elif dataset == "MORPH2":
            CFG.ConvNeXT.weights = "<path>/imdb-clean-1024_ConvNeXT_FC2_256_batch128_epoch_99_mae_4.235945536028426.ckpt" 
            CFG.Transformer.weights = "<path>/imdb-clean-face-data_Transformer_FC2_256_batch64_Huber_1classes_epoch_99_mae_5.137645244598389.ckpt"  
            CFG.Transformer.fc = 256 # None, 32, 64, 128, 192, 256 
            CFG.ConvNeXT.fc = 256 # None, 32, 64, 128, 192, 256
       

    ### DATA ###
    path = "<path>/Datasets"

    tf_train, tf_test = load_transforms()
    train = DATASET(path, dataset, "Train", tf_train)
    valid = DATASET(path, dataset, "Val", tf_test)
    test = DATASET(path, dataset, "Test", tf_test)
    print(f"Train size: {len(train)} - Valid size: {len(valid)} - Test size: {len(test)}")

    CFG.Trainer.scheduler_total_steps = round(CFG.Trainer.num_epochs * (len(train) / CFG.Trainer.batch_size))
    CFG.Trainer.scheduler_warmup_steps = round(CFG.Trainer.scheduler_factor * CFG.Trainer.scheduler_total_steps)


    train_loader = DataLoader(dataset=train, batch_size=CFG.Trainer.batch_size, shuffle=True, num_workers=CFG.Trainer.num_workers, pin_memory=True)
    valid_loader = DataLoader(dataset=valid, batch_size=CFG.Trainer.batch_size, shuffle=False, num_workers=CFG.Trainer.num_workers, pin_memory=True)
    test_loader = DataLoader(dataset=test, batch_size=CFG.Trainer.batch_size, shuffle=False, num_workers=CFG.Trainer.num_workers, pin_memory=True)

    CFG.Trainer.scheduler = "WarmupCosineSchedule"

    # Initialize the model
    model = MODEL(CFG)

    ### Result saving 
    if CFG.Transformer.fc == None:
            weights_name = f"{dataset}_CNNXT_VIT_FC1_{CFG.Trainer.loss_func}_{config}"
        else:
            weights_name = f"{dataset}_CNNXT_VIT_FC2_batch{CFG.Trainer.batch_size}_{CFG.Trainer.loss_func}_{config}"

    save_file = f"./results_ConvNeXT_Transformer/{today_date}/{weights_name}.pkl"
    if not os.path.exists(f"./results_ConvNeXT_Transformer/{today_date}"):
        os.makedirs(f"./results_ConvNeXT_Transformer/{today_date}")

    ### CREATE FOLDER ###
    directory = f'./checkpoints/checkpoints_ConvNeXT_Transformer_{today_date}'
    if not os.path.exists(directory):
        os.makedirs(directory)

    ### TRAIN/EVAL ###
    loss = '_epoch_{epoch:02d}_mae_{MAE/Val}'
    if CFG.Trainer.loss_func == 'CrossEntropy':
        loss= '_epoch_{epoch:02d}_top1_{TOP1/Val}_top5_{TOP5/Val}'

    trainer = pl.Trainer(accelerator='gpu', 
                        devices= CFG.Trainer.devices, 
                        strategy='ddp',
                        callbacks=[pl.callbacks.ModelCheckpoint(
                                    save_top_k = 1, 
                                    dirpath = directory,
                                    filename = weights_name  + loss,
                                    auto_insert_metric_name=False),
                            ],
                        logger=pl.loggers.TensorBoardLogger(f'/data2/gmaroun/Projects/ConvNeXT_Vision_Transformer/logs/{dataset}/{config}_{today_date}',
                                                            name=weights_name , 
                                                            default_hp_metric=False),
                        max_epochs=CFG.Trainer.num_epochs,
                        benchmark=True,
                        precision=32,
                        )

    if CFG.Trainer.checkpoint_file:
        ckpt = torch.load(CFG.Trainer.checkpoint_file, map_location='cpu')
        # Load only model weights (not optimizer/scheduler)
        state_dict = ckpt["state_dict"]
        # Remove mismatched head keys (optional: cleaner)
        filtered_dict = {k.replace("model.", ""): v for k, v in state_dict.items()
                        if "vit.head" not in k}
        # Load with strict=False to skip mismatches
        model.model.load_state_dict(filtered_dict, strict=False)
    else:
        trainer.fit(model, train_loader, valid_loader)
    # val_data = pd.DataFrame({"gt": model.val_labels_gt, "p": model.val_labels_p})
    # val_data.to_pickle(f"./results_ConvNeXT_Transformer/{today_date}/VAL_{weights_name}.pkl")
    trainer.test(model, dataloaders=test_loader)
    data = pd.DataFrame({"gt": model.labels_gt, "p": model.labels_p})
    data.to_pickle(save_file)
    model.eval().to('cuda')
