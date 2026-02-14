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

# def test_batch_size(batch_size, CFG, dataset_name):
#     try:
#         CFG.Trainer.batch_size = batch_size
#         path = "/lscratch/gmaroun/Datasets"
#         tf_train, tf_test = load_transforms()
#         train = DATASET(path, dataset_name, "Train", tf_train)
#         train_loader = DataLoader(dataset=train, batch_size=CFG.Trainer.batch_size, shuffle=True, num_workers=CFG.Trainer.num_workers, pin_memory=True)
        
#         model = MODEL(CFG)
#         trainer = pl.Trainer(accelerator='gpu', devices=CFG.Trainer.devices, max_epochs=1)
#         trainer.fit(model, train_loader)
#         return True
#     except RuntimeError as e:
#         if 'out of memory' in str(e):
#             torch.cuda.empty_cache()
#             return False
#         else:
#             raise e
# torch.cuda.empty_cache()
### CONFIGS ###
#checkpoint_file = "/home/gmaroun/Projects/ConvNeXT_Vision_Transformer/checkpoints_Transformer/Transformer_FC2_256_epoch_50_mae_8.325320756485693.ckpt"
# checkpoint_file = "checkpoints_Transformer/MORPH2_Transformer_FC2_256_batch256_ImgNet_2nd_epoch_49_mae_4.482994710152716.ckpt"

# def set_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# Set a seed for reproducibility
# set_seed(42)
# def find_max_batch_size(CFG, dataset_name, min_batch_size=112, max_batch_size=1024):
#     while min_batch_size < max_batch_size:
#         mid_batch_size = (min_batch_size + max_batch_size + 1) // 2
#         if test_batch_size(mid_batch_size, CFG, dataset_name):
#             min_batch_size = mid_batch_size
#         else:
#             max_batch_size = mid_batch_size - 1
#     return min_batch_size
### MODEL ###
config = '6'
CFG = CN()
CFG.Trainer = CN()
CFG.Trainer.checkpoint_file = False
# CFG.Trainer.checkpoint_file = "/home/gmaroun/Projects/ConvNeXT_Vision_Transformer/checkpoints_ConvNeXT_Transformer_15_10_23/MORPH2_CNNXT_VIT_preT_MORPH2_FC2_batch128_lossfunc_Adaptive_epoch_499_mae_2.266825668598924.ckpt"
# CFG.Trainer.checkpoint_file = "Projects/ConvNeXT_Vision_Transformer/checkpoints_ConvNeXT_Transformer_07_09_24/CACD_CNNXT_VIT_preT_MORPH2_FC2_batch64_lossfunc_Adaptive_epoch_19_mae_4.422359941865465.ckpt"
CFG.Trainer.num_epochs = 100
CFG.Trainer.batch_size = 64 #64
CFG.Trainer.num_workers = os.cpu_count() - 6
CFG.Trainer.accelerator = 'gpu'
CFG.Trainer.devices = [0] # not use 0 and 1
CFG.Trainer.loss_func = 'Adaptive' # 'MAE', 'MSE', 'Adaptive', 'Huber' --> Adaptive, KLDivLoss (no use yet)
CFG.Trainer.sigma = 2 # used for Adaptive loss
CFG.Trainer.lr = 5e-5 #OneCycleLR_1 1e-4
CFG.Trainer.lr_patience = 3 #5
CFG.Trainer.lr_min = 1e-6 #1e-8
# CFG.Trainer.scheduler = "CosineAnnealingLR" # ReduceLROnPlateau, WarmupCosineSchedule, OneCycleLR, CosineAnnealingLR, StepLR, ExponentialLR, CyclicLR
CFG.Trainer.scheduler_interval = "step"
CFG.Trainer.scheduler_monitor = "Loss/Val"
CFG.Trainer.scheduler_factor = 0.1 # 0.1 => 0.5 0.001 # 0 < factor < 1 (if its > 1 it means the)
CFG.Trainer.scheduler_warmup_steps = 0
CFG.Trainer.scheduler_total_steps = 0


CFG.Model = CN()
CFG.Model.name = "ConvNeXT_Transformer"
CFG.Model.TPS = False
CFG.Model.num_fiducial = 20

CFG.ConvNeXT = CN()
CFG.ConvNeXT.pretrained = False
# CFG.ConvNeXT.backbone = "convnext_base" 
# CFG.ConvNeXT.weights = "/home/gmaroun/Projects/ConvNeXT_Vision_Transformer/checkpoints_ConvNeXT/MORPH2_ConvNeXT_FC2_256_epoch_499_mae_2.290440150948744_2ndRound.ckpt" #MORPH2
# CFG.ConvNeXT.weights = "/home/gmaroun/Projects/ConvNeXT_Vision_Transformer/checkpoints_ConvNeXT/CACD_ConvNeXT_FC2_128_epoch_99_mae_4.1533247541818.ckpt" #CACD
# CFG.ConvNeXT.weights = "/home/gmaroun/Projects/ConvNeXT_Vision_Transformer/checkpoints_ConvNeXT/AFAD_ConvNeXT_FC2_256_epoch_99_mae_5.285212279458425.ckpt" #AFAD
CFG.ConvNeXT.in_chans=3
CFG.ConvNeXT.num_classes = 1
CFG.ConvNeXT.depths=[3, 3, 9, 3]
CFG.ConvNeXT.dims=[96, 192, 384, 768]
CFG.ConvNeXT.drop_path_rate=0. 
CFG.ConvNeXT.layer_scale_init_value=1e-6
CFG.ConvNeXT.head_init_scale=1.
# CFG.ConvNeXT.fc = 128 # None, 32, 64, 128, 192, 256
# CFG.ConvNeXT.fc = 256 # None, 32, 64, 128, 192, 256

CFG.Transformer = CN()
# CFG.Transformer.pretrained = False
# CFG.Transformer.weights = "/home/gmaroun/Projects/ConvNeXT_Vision_Transformer/checkpoints_Transformer_ImgNet_Cosine_19_8_23/MORPH2_Transformer_FC2_128_batch256_ImgNet_Cosine0.001_epoch_499_mae_2.4702752187618136.ckpt" #MORPH2
# CFG.Transformer.weights = "/home/gmaroun/Projects/ConvNeXT_Vision_Transformer/checkpoints_Transformer_ImgNet_Cosine_14_7_23/CACD_Transformer_FC2_vit_tiny_patch16_224_256_batch256_ImgNet_Cosine0.001_epoch_499_mae_4.713215787607605.ckpt"  #CACD
# CFG.Transformer.weights = "/home/gmaroun/Projects/ConvNeXT_Vision_Transformer/checkpoints_Transformer_ImgNet_Cosine_19_8_23/AFAD_Transformer_FC2_256_batch256_ImgNet_Cosine0.001_epoch_499_mae_6.207571459113868.ckpt" #AFAD
CFG.Transformer.backbone = "vit_tiny_patch16_224" #"vit_tiny_patch16_224"
CFG.Transformer.transfer_learning = True
CFG.Transformer.pretrained = True
CFG.Transformer.num_class = 1 
# CFG.Transformer.fc = 128 # None, 32, 64, 128, 192, 256 
# CFG.Transformer.fc = 256 # None, 32, 64, 128, 192, 256 



# batches = [128, 256]
datasets = ["AFAD", "CACD"]
# dataset = datasets[0]
# CFG.Trainer.batch_size = 64

# for batch_size in batches:
#     CFG.Trainer.batch_size = batch_size
for dataset in datasets:
# i want to repeat the work twice on 2 different weights
    if dataset == "CACD":
        CFG.ConvNeXT.weights = "/home/gmaroun/Projects/ConvNeXT_Vision_Transformer/checkpoints_ConvNeXT/CACD_ConvNeXT_FC2_128_epoch_99_mae_4.1533247541818.ckpt" #CACD
        CFG.Transformer.weights = "/home/gmaroun/Projects/ConvNeXT_Vision_Transformer/checkpoints_Transformer_ImgNet_Cosine_14_7_23/CACD_Transformer_FC2_vit_tiny_patch16_224_256_batch256_ImgNet_Cosine0.001_epoch_499_mae_4.713215787607605.ckpt"  #CACD
        CFG.Transformer.fc = 256 # None, 32, 64, 128, 192, 256 
        CFG.ConvNeXT.fc = 128 # None, 32, 64, 128, 192, 256
    elif dataset == "MORPH2":
        CFG.ConvNeXT.weights = "/home/gmaroun/Projects/ConvNeXT_Vision_Transformer/checkpoints_ConvNeXT/MORPH2_ConvNeXT_FC2_256_epoch_499_mae_2.290440150948744_2ndRound.ckpt" #MORPH2
        CFG.Transformer.weights = "/home/gmaroun/Projects/ConvNeXT_Vision_Transformer/checkpoints_Transformer_ImgNet_Cosine_19_8_23/MORPH2_Transformer_FC2_128_batch256_ImgNet_Cosine0.001_epoch_499_mae_2.4702752187618136.ckpt"  #MORPH2
        CFG.Transformer.fc = 128 # None, 32, 64, 128, 192, 256 
        CFG.ConvNeXT.fc = 256 # None, 32, 64, 128, 192, 256
    else:
        CFG.ConvNeXT.weights = "/home/gmaroun/Projects/ConvNeXT_Vision_Transformer/checkpoints_ConvNeXT/AFAD_ConvNeXT_FC2_256_epoch_99_mae_5.285212279458425.ckpt" #AFAD
        CFG.Transformer.weights = "/home/gmaroun/Projects/ConvNeXT_Vision_Transformer/checkpoints_Transformer_ImgNet_Cosine_19_8_23/AFAD_Transformer_FC2_256_batch256_ImgNet_Cosine0.001_epoch_499_mae_6.207571459113868.ckpt" #AFAD
        CFG.ConvNeXT.fc = 256 # None, 32, 64, 128, 192, 256
        CFG.Transformer.fc = 256 # None, 32, 64, 128, 192, 256 

    ### DATA ###
    # datasets = ["AFAD", "CACD", "MORPH2"] 
    path = "/lscratch/gmaroun/Datasets"
    # dataset = datasets[1]

    tf_train, tf_test = load_transforms()
    train = DATASET(path, dataset, "Train", tf_train)
    valid = DATASET(path, dataset, "Val", tf_test)
    test = DATASET(path, dataset, "Test", tf_test)

    # if CFG.Trainer.scheduler != "OneCycleLR":
    CFG.Trainer.scheduler_total_steps = round(CFG.Trainer.num_epochs * (len(train) / CFG.Trainer.batch_size))
    CFG.Trainer.scheduler_warmup_steps = round(CFG.Trainer.scheduler_factor * CFG.Trainer.scheduler_total_steps)
    # else:
    #     CFG.Trainer.scheduler_total_steps = round(CFG.Trainer.num_epochs * (len(train) / CFG.Trainer.batch_size))
    #     CFG.Trainer.scheduler_warmup_steps = round(len(train) / CFG.Trainer.batch_size)

    train_loader = DataLoader(dataset=train, batch_size=CFG.Trainer.batch_size, shuffle=True, num_workers=CFG.Trainer.num_workers, pin_memory=True)
    valid_loader = DataLoader(dataset=valid, batch_size=CFG.Trainer.batch_size, shuffle=False, num_workers=CFG.Trainer.num_workers, pin_memory=True)
    test_loader = DataLoader(dataset=test, batch_size=CFG.Trainer.batch_size, shuffle=False, num_workers=CFG.Trainer.num_workers, pin_memory=True)

    # schedulers = ["StepLR", "ExponentialLR"]    # ReduceLROnPlateau, WarmupCosineSchedule, OneCycleLR, CosineAnnealingLR, StepLR, ExponentialLR, CyclicLR
    # schedulers = ["CyclicLR"]    
    # for sched in schedulers:

    # CFG.Trainer.scheduler = sched
    CFG.Trainer.scheduler = "WarmupCosineSchedule"

    # dataset_name = "AFAD"
    # max_batch_size = find_max_batch_size(CFG, dataset_name)
    # print(f"The maximum batch size that fits in memory is: {max_batch_size}")

    # Initialize the model
    model = MODEL(CFG)

    ### Result saving 
    if CFG.Transformer.fc == None:
        if CFG.Model.TPS == True:
            weights_name = f"{dataset}_CNNXT_VIT_FC1_TPS_batch{CFG.Trainer.batch_size}_lossfunc_{CFG.Trainer.loss_func}_config{config}"
        else:
            weights_name = f"{dataset}_CNNXT_VIT_FC1_batch{CFG.Trainer.batch_size}_lossfunc_{CFG.Trainer.loss_func}_config{config}"
    else:
        if CFG.Model.TPS == True:
            weights_name = f"{dataset}_CNNXT_VIT_FC2_TPS_batch{CFG.Trainer.batch_size}_lossfunc_{CFG.Trainer.loss_func}_config{config}"
        else:
            weights_name = f"{dataset}_CNNXT_VIT_FC2_batch{CFG.Trainer.batch_size}_lossfunc_{CFG.Trainer.loss_func}_config{config}"

    save_file = f"/home/gmaroun/Projects/ConvNeXT_Vision_Transformer/results_ConvNeXT_Transformer/{weights_name}.pkl"

    ### CREATE FOLDER ###
    directory = '/home/gmaroun/Projects/ConvNeXT_Vision_Transformer/checkpoints_ConvNeXT_Transformer_27_10_24'
    if not os.path.exists(directory):
        os.makedirs(directory)


    # early_stopping_callback = pl.callbacks.EarlyStopping(
    #     monitor = 'Loss/Val',
    #     patience = 10,
    #     verbose = True,
    #     mode = 'min'
    # )

    ### TRAIN/EVAL ###
    trainer = pl.Trainer(accelerator='gpu', 
                        devices= CFG.Trainer.devices, 
                        auto_select_gpus=False,
                        callbacks=[pl.callbacks.ModelCheckpoint(
                                    save_top_k = 1, 
                                    dirpath = directory,
                                    filename = weights_name  + '_epoch_{epoch:02d}_mae_{MAE/Val}',
                                    auto_insert_metric_name=False),
                            # early_stopping_callback #config_ReduceLROnPlateau_1
                            ],
                        logger=pl.loggers.TensorBoardLogger('/home/gmaroun/Projects/ConvNeXT_Vision_Transformer/logs', 
                                                            name=weights_name , 
                                                            default_hp_metric=False),
                        max_epochs=CFG.Trainer.num_epochs,
                        benchmark=True,
                        precision=32,
                        # accumulate_grad_batches=4  # Accumulate gradients over 4 batches
                        )

    if CFG.Trainer.checkpoint_file:
        trainer.fit(model, train_loader, valid_loader, ckpt_path=CFG.Trainer.checkpoint_file)
    else:
        trainer.fit(model, train_loader, valid_loader)
    trainer.test(model, dataloaders=test_loader)
    data = pd.DataFrame({"gt": model.labels_gt, "p": model.labels_p})
    data.to_pickle(save_file)
    print(data)
    print(weights_name)

"""
conda activate conda3.9
python /home/gmaroun/Projects/ConvNeXT_Vision_Transformer/train_transformer_convnext.py

for step, batch in enumerate(train_loader, 1):
    # Perform the necessary training operations for each step
    
    # Monitoring the step
    if step % 10 == 0:  # Print the step every 10 steps (adjust as needed)
        print(f"Step: {step}/{len(train_loader)}")

checkpoints = torch.load(checkpoint_file)['state_dict']
model.load_state_dict(checkpoints)
trainer.test(model, dataloaders=test_loader)
mae_result = trainer.callback_metrics['MAE/Val']



## CONFIGS No 2 ###
CFG.Trainer.lr = 5e-5 
CFG.Trainer.lr_patience = 3 
CFG.Trainer.lr_min = 1e-6 
CFG.Trainer.scheduler_factor = 0.1 

## CONFIGS No 3 ###
# Initialize the trainer with a limited number of epochs for the learning rate finder
trainer = pl.Trainer(accelerator='gpu', 
                     devices=CFG.Trainer.devices, 
                     auto_select_gpus=False,
                     max_epochs=CFG.Trainer.num_epochs,
                     benchmark=True,
                     precision=32,
                     )
# Use the learning rate finder
lr_finder = trainer.tuner.lr_find(model, train_loader, valid_loader)
# # Plot the learning rate finder results
# fig = lr_finder.plot(suggest=True)
# fig.show()
# Pick the suggested learning rate
new_lr = lr_finder.suggestion()
# Update the configuration with the new learning rate
CFG.Trainer.lr = new_lr

# Reinitialize the model
model = MODEL(CFG)


## CONFIGS No 4 ###
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        warmup_scheduler = WarmupCosineSchedule(optimizer, self.scheduler_warmup_steps, self.scheduler_total_steps, cycles=0.5, last_epoch=-1)
        # Define the second scheduler, e.g., ReduceLROnPlateau or OneCycleLR
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=self.lr_patience, min_lr=self.lr_min, verbose=True)
        # Define a Lambda function for conditionally switching from WarmupCosineSchedule to ReduceLROnPlateau
        def lr_lambda(epoch):
            if epoch < self.scheduler_warmup_steps:
                return warmup_scheduler.get_last_lr()
            else:
                return plateau_scheduler.get_last_lr()
        # Combine both schedulers into a LambdaLR scheduler
        combined_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": combined_scheduler, "monitor": self.scheduler_monitor, "interval": self.scheduler_interval}}


## CONFIGS No 5 (+2) ###

        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=self.scheduler_factor, 
            patience=self.lr_patience, 
            min_lr=self.lr_min, 
            verbose=True
        )
        
        return {
            'optimizer': optimizer, 
            'lr_scheduler': {
                'scheduler': scheduler, 
                'monitor': self.scheduler_monitor, 
                'interval': 'epoch', 
                'frequency': 1
            }
        }



## CONFIGS No 6 ###
weights of CACD best transformers and convNeXt models
weights of AFAD best transformers and convNeXt models
since 28/10/24 I started using warmupcosinescheduler (it was not being taken into account before)

## CONFIGS No 7 ###
 # config 1 + 6 ##
CFG.Trainer.sigma = 2 
CFG.Trainer.lr = 1e-4 
CFG.Trainer.lr_patience = 5 
CFG.Trainer.lr_min = 1e-8 
CFG.Trainer.scheduler_factor = 0.1 
"""