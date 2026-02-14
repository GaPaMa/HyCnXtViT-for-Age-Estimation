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

### CONFIGS ###
datasets = ["AFAD", "MORPH2", "CACD"] 

trained_model = [
"AFAD_ConvNeXT_FC2_256_epoch_99_mae_5.285212279458425",
"AFAD_ConvNeXT_FC2_256_epoch_499_mae_5.472355589302445_2ndRound",
# "CACD_ConvNeXT_FC2_128_epoch_99_mae_4.1533247541818",
# "CACD_ConvNeXT_FC2_128_epoch_499_mae_4.252772177708951_2ndRound",
# "CACD_ConvNeXT_FC2_256_epoch_99_mae_4.190253332861508",
# "CACD_ConvNeXT_FC2_256_epoch_499_mae_4.237253556403623_2ndRound",
# "ConvNeXT_FC2_64_epoch_99_mae_4.431495231910499.ckpt",
# "ConvNeXT_FC2_256_epoch_99_mae_4.458243568685161",
# "MORPH2_ConvNeXT_FC2_256_epoch_99_mae_2.364719239643642",
# "MORPH2_ConvNeXT_FC2_256_epoch_499_mae_2.290440150948744_2ndRound",
]
data = pd.DataFrame(columns=['Model', 'Dataset', 'MAE/Test'])

# checkpoint_file = False
# num_epochs = 100
# batch_size = 128
# num_workers = os.cpu_count() - 6
# accelerator = 'gpu'
# devices = [1] # not use 0 and 1
# TPS = True
# num_fiducial = 20
# loss_func = 'MAE' # 'MAE', 'MSE', 'Adaptive', 'Huber'
# sigma = 2 # used for Adaptive loss
# lr = 1e-4
# lr_patience = 5
# lr_min = 1e-7

### MODEL ###
CFG = CN()
CFG.model = "ConvNeXT"

CFG.Model = CN()
CFG.Model.name = "ConvNeXT"
CFG.Model.TPS = True
CFG.Model.num_fiducial = 20


CFG.ConvNeXT = CN()
# CFG.ConvNeXT.pretrained = "/lscratch/gmaroun/Projects/ConvNeXT_Vision_Transformer/weights/convnext_tiny_22k_224.pth"
CFG.ConvNeXT.pretrained = False
CFG.ConvNeXT.in_chans=3
# CFG.ConvNeXT.num_classes=21841
CFG.ConvNeXT.num_classes=1
CFG.ConvNeXT.depths=[3, 3, 9, 3]
CFG.ConvNeXT.dims=[96, 192, 384, 768]
CFG.ConvNeXT.drop_path_rate=0. 
CFG.ConvNeXT.layer_scale_init_value=1e-6
CFG.ConvNeXT.head_init_scale=1.
CFG.ConvNeXT.fc = 256 # None, 32, 64, 128, 192, 256

CFG.Trainer = CN()
CFG.Trainer.checkpoint_file = False
CFG.Trainer.num_epochs = 1
CFG.Trainer.batch_size = 64 #64
CFG.Trainer.num_workers = os.cpu_count() - 6
CFG.Trainer.accelerator = 'gpu'
CFG.Trainer.devices = [0] # not use 0 and 1
CFG.Trainer.loss_func = 'MAE' # 'MAE', 'MSE', 'Adaptive', 'Huber' --> Adaptive, KLDivLoss (no use yet)
CFG.Trainer.sigma = 2 # used for Adaptive loss
CFG.Trainer.lr = 5e-5 #OneCycleLR_1 1e-4
CFG.Trainer.lr_patience = 3 #5
CFG.Trainer.lr_min = 1e-6 #1e-8
CFG.Trainer.scheduler = False
CFG.Trainer.scheduler_interval = "step"
CFG.Trainer.scheduler_monitor = "Loss/Val"
CFG.Trainer.scheduler_factor = 0.1 # 0.1 => 0.5 0.001 # 0 < factor < 1 (if its > 1 it means the)
CFG.Trainer.scheduler_warmup_steps = 0
CFG.Trainer.scheduler_total_steps = 0

datasets = ["AFAD"] 

for tm in trained_model:
    for dataset in datasets:

        # checkpoint_file = f"./checkpoints_convneXt/{tm}.ckpt"
        CFG.Trainer.checkpoint_file = False
        CFG.Trainer.checkpoint_file = "/home/gmaroun/Projects/ConvNeXT_Vision_Transformer/checkpoints_ConvNeXT_10_11_24/AFAD_ConvNeXT_FC2_256_v2_test_epoch_00_mae_10.557867567741111.ckpt"

        ### DATA ###
        path = "/lscratch/gmaroun/Datasets"
        tf_train, tf_test = load_transforms()
        train = DATASET(path, dataset, "Train", tf_train)
        valid = DATASET(path, dataset, "Val", tf_test)
        test = DATASET(path, dataset, "Test", tf_test)

        train_loader = DataLoader(dataset=train, batch_size=CFG.Trainer.batch_size, shuffle=True, num_workers=CFG.Trainer.num_workers, pin_memory=True)
        valid_loader = DataLoader(dataset=valid, batch_size=CFG.Trainer.batch_size, shuffle=False, num_workers=CFG.Trainer.num_workers, pin_memory=True)
        test_loader = DataLoader(dataset=test, batch_size=CFG.Trainer.batch_size, shuffle=False, num_workers=CFG.Trainer.num_workers, pin_memory=True)

        ### Result saving 
        if CFG.ConvNeXT.fc == None:
            weights_name = f"{dataset}_ConvNeXT_FC1_v2"
        else:
            weights_name = f"{dataset}_ConvNeXT_FC2_{CFG.ConvNeXT.fc}_v2"

        # save_file = f"./results/{weights_name}_MAE.pkl"
        save_file = f"/home/gmaroun/Projects/ConvNeXT_Vision_Transformer/results_ConvNeXT/{weights_name}_MAE.pkl"
        directory = '/home/gmaroun/Projects/ConvNeXT_Vision_Transformer/checkpoints_ConvNeXT_10_11_24'

        ### Model
        # model = MODEL(CFG, loss_func=CFG.Trainer.loss_func, TPS=CFG.Model.TPS, num_fiducial=CFG.Model.num_fiducial, lr = CFG.Trainer.lr, lr_patience = CFG.Trainer.lr_patience, lr_min = CFG.Trainer.lr_min, sigma= CFG.Trainer.sigma)
        model = MODEL(CFG)

        ### TRAIN/EVAL ###
        trainer = pl.Trainer(accelerator='gpu', devices=CFG.Trainer.devices, auto_select_gpus=False,
                            callbacks=[pl.callbacks.ModelCheckpoint(save_top_k = 1, 
                                                                    dirpath = directory,
                                        filename = weights_name  + '_epoch_{epoch:02d}_mae_{MAE/Val}',
                                        auto_insert_metric_name=False)],
                            logger=pl.loggers.TensorBoardLogger('/home/gmaroun/Projects/ConvNeXT_Vision_Transformer/logs', 
                                                                name=weights_name, 
                                                                default_hp_metric=False),
                            max_epochs=CFG.Trainer.num_epochs,
                            benchmark=True,
                            precision=32,
                            )

        if CFG.Trainer.checkpoint_file:
            # checkpoints = torch.load(CFG.Trainer.checkpoint_file)['state_dict']
            # model.load_state_dict(checkpoints)
            # trainer.test(model, dataloaders=test_loader)
            trainer.fit(model, train_loader, valid_loader, ckpt_path=CFG.Trainer.checkpoint_file)
            # trainer.fit(model, train_loader, valid_loader)
        else:
            # print("else")
            trainer.fit(model, train_loader, valid_loader)
        # print("fit")
        # trainer.fit(model, train_loader, valid_loader)
        trainer.test(model, dataloaders=test_loader)
        data = pd.DataFrame({"gt": model.labels_gt, "p": model.labels_p})
        # mae_result = trainer.callback_metrics['MAE/Test']
        # data = data.append({"Model": tm, 
        #                     "Dataset": dataset,
        #                     "MAE/Test": mae_result}, ignore_index=True)
        data.to_pickle(save_file)
        data = pd.DataFrame(data)
        print(data)
        print(weights_name)
# data.to_pickle("./results/all_MAE.pkl")

# print(f"MAE/Test: {mae_result}")