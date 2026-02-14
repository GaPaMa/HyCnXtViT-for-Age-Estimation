import os
import torch
import pytorch_lightning as pl
import pandas as pd
from yacs.config import CfgNode as CN
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from data import DATASET, load_transforms
from model.model import MODEL

### CONFIGS ###

config='Transformers'
### MODEL ###
CFG = CN()
CFG.Trainer = CN()
# CFG.Trainer.checkpoint_file = False
CFG.Trainer.checkpoint_file = './checkpoints/checkpoints_Transformer_ImgNet_Cosine_19_8_23/MORPH2_Transformer_FC2_128_batch256_ImgNet_Cosine0.001_epoch_499_mae_2.4702752187618136.ckpt'
CFG.Trainer.num_epochs = 500
CFG.Trainer.batch_size = 256
CFG.Trainer.num_workers = os.cpu_count() - 6
# CFG.Trainer.num_workers = 8
CFG.Trainer.accelerator = 'gpu'
CFG.Trainer.devices = [1, 4] # not use 0 and 1

CFG.Trainer.loss_func = 'MAE' # 'MAE', 'MSE', 'Adaptive', 'Huber'
CFG.Trainer.num_class = 1
CFG.Trainer.sigma = 2 # used for Adaptive loss
CFG.Trainer.lr = 1e-5
CFG.Trainer.lr_patience = 3
CFG.Trainer.lr_min = 1e-6
CFG.Trainer.scheduler = "WarmupCosineSchedule" # ReduceLROnPlateau, WarmupCosineSchedule
CFG.Trainer.scheduler_interval = "step"
CFG.Trainer.scheduler_monitor = "Loss/Val"
CFG.Trainer.scheduler_factor = 0.1 
CFG.Trainer.scheduler_warmup_steps = 0
CFG.Trainer.scheduler_total_steps = 0


CFG.Model = CN()
CFG.Model.name = "Transformer"
CFG.Model.TPS = False
CFG.Model.num_fiducial = 20

CFG.Transformer = CN()
CFG.Transformer.transfer_learning = True
CFG.Transformer.num_class = CFG.Trainer.num_class
CFG.Transformer.backbone = "vit_tiny_patch16_224" 
CFG.Transformer.pretrained = True
CFG.Transformer.fc = 128 # None, 32, 64, 128, 192, 256 
CFG.Transformer.image_size = 224
CFG.Transformer.patch_size = 16
CFG.Transformer.dim = 768
CFG.Transformer.depth = 8
CFG.Transformer.heads = 12
CFG.Transformer.mlp_dim = 2048
CFG.Transformer.pool = 'cls'
CFG.Transformer.channels = 3
CFG.Transformer.dim_head = 64
CFG.Transformer.dropout = 0.1
CFG.Transformer.emb_dropout = 0.1

### DATA ###
datasets = ["AFAD", "MORPH2", "CACD", "imdb-clean-face-data"] 
path = ".../Datasets"

dataset = datasets[1]

tf_train, tf_test = load_transforms()
train = DATASET(path, dataset, "Train", tf_train)
valid = DATASET(path, dataset, "Val", tf_test)
test = DATASET(path, dataset, "Test", tf_test)

if CFG.Trainer.scheduler == "WarmupCosineSchedule":
    CFG.Trainer.scheduler_total_steps = round(CFG.Trainer.num_epochs * (len(train) / CFG.Trainer.batch_size))
    CFG.Trainer.scheduler_warmup_steps = round(CFG.Trainer.scheduler_factor * CFG.Trainer.scheduler_total_steps)

train_loader = DataLoader(dataset=train, batch_size=CFG.Trainer.batch_size, shuffle=True, num_workers=CFG.Trainer.num_workers, pin_memory=True)
valid_loader = DataLoader(dataset=valid, batch_size=CFG.Trainer.batch_size, shuffle=False, num_workers=CFG.Trainer.num_workers, pin_memory=True)
test_loader = DataLoader(dataset=test, batch_size=CFG.Trainer.batch_size, shuffle=False, num_workers=CFG.Trainer.num_workers, pin_memory=True)

model = MODEL(CFG)


### Result saving 
if CFG.Transformer.fc == None:
    weights_name = f"{dataset}_Transformer_FC1_{CFG.Trainer.loss_func}"
else:
    weights_name = f"{dataset}_Transformer_FC2_batch{CFG.Trainer.batch_size}_{CFG.Trainer.loss_func}"

save_file = f"./results_Transformer/{today_date}/{weights_name}.pkl"
if not os.path.exists(f"./results_Transformer/{today_date}"):
    os.makedirs(f"./results_Transformer/{today_date}")

### CREATE FOLDER ###
directory = f'./checkpoints/checkpoints_Transformer_{today_date}'
if not os.path.exists(directory):
    os.makedirs(directory)

### TRAIN/EVAL ###
loss = '_epoch_{epoch:02d}_mae_{MAE/Val}'
if CFG.Trainer.loss_func == 'CrossEntropy':
    loss= '_epoch_{epoch:02d}_top1_{TOP1/Val}_top5_{TOP5/Val}'
    
trainer = pl.Trainer(accelerator='gpu', 
                    devices= CFG.Trainer.devices, 
                    strategy='ddp',
                    callbacks=[pl.callbacks.ModelCheckpoint(save_top_k = 1, 
                                                            dirpath = directory,
                                                            filename = weights_name  + loss,
                                                            auto_insert_metric_name=False)],
                    logger=pl.loggers.TensorBoardLogger(f'/data2/gmaroun/Projects/ConvNeXT_Vision_Transformer/logs/{dataset}/{config}_{today_date}', 
                                                        name=weights_name , 
                                                        default_hp_metric=False),
                    max_epochs=CFG.Trainer.num_epochs,
                    benchmark=True,
                    precision=32,
                    gradient_clip_val = 1.0
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
trainer.test(model, dataloaders=test_loader)

data = pd.DataFrame({"gt": model.labels_gt, "p": model.labels_p})
data.to_pickle(save_file)
