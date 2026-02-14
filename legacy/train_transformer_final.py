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
#checkpoint_file = "/home/gmaroun/Projects/ConvNeXT_Vision_Transformer/checkpoints_Transformer/Transformer_FC2_256_epoch_50_mae_8.325320756485693.ckpt"
# checkpoint_file = "checkpoints_Transformer/MORPH2_Transformer_FC2_256_batch256_ImgNet_2nd_epoch_49_mae_4.482994710152716.ckpt"

### MODEL ###
CFG = CN()
CFG.Trainer = CN()
CFG.Trainer.checkpoint_file = False
CFG.Trainer.num_epochs = 500
CFG.Trainer.batch_size = 256
CFG.Trainer.num_workers = os.cpu_count() - 6
CFG.Trainer.accelerator = 'gpu'
CFG.Trainer.devices = [2] # not use 0 and 1
CFG.Trainer.loss_func = 'MAE' # 'MAE', 'MSE', 'Adaptive', 'Huber'
CFG.Trainer.sigma = 2 # used for Adaptive loss
CFG.Trainer.lr = 1e-2
CFG.Trainer.lr_patience = 5
CFG.Trainer.lr_min = 1e-8
CFG.Trainer.scheduler = "WarmupCosineSchedule" # ReduceLROnPlateau, WarmupCosineSchedule
CFG.Trainer.scheduler_interval = "step"
CFG.Trainer.scheduler_monitor = "Loss/Val"
CFG.Trainer.scheduler_factor = 0.001 # 0 < factor < 1 (if its > 1 it means the)
CFG.Trainer.scheduler_warmup_steps = 0
CFG.Trainer.scheduler_total_steps = 0


CFG.Model = CN()
CFG.Model.name = "Transformer"
CFG.Model.TPS = False
CFG.Model.num_fiducial = 20

CFG.Transformer = CN()
CFG.Transformer.transfer_learning = True
CFG.Transformer.num_class = 1 
CFG.Transformer.pretrained = "checkpoints_Transformer_training_ImgNet_Cosine/ImageNet_Transformer_FC2_256_batch256_ImgNet_Cosine0.001_epoch_499_top1_0.39869657158851624_top5_0.6500916481018066.ckpt"
CFG.Transformer.fc = 256 # None, 32, 64, 128, 192, 256 
CFG.Transformer.image_size = 224
CFG.Transformer.patch_size = 32
CFG.Transformer.dim = 512
CFG.Transformer.depth = 6
CFG.Transformer.heads = 8
CFG.Transformer.mlp_dim = 2048
CFG.Transformer.pool = 'cls'
CFG.Transformer.channels = 3
CFG.Transformer.dim_head = 64
CFG.Transformer.dropout = 0.1
CFG.Transformer.emb_dropout = 0.1

### DATA ###
datasets = ["AFAD", "MORPH2", "CACD"] 
path = "/lscratch/gmaroun/Datasets"
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
    if CFG.Model.TPS == True:
        weights_name = f"{dataset}_Transformer_FC1_TPS_batch{CFG.Trainer.batch_size}_ImgNet_Cosine{CFG.Trainer.scheduler_factor}"
    else:
        weights_name = f"{dataset}_Transformer_FC1_batch{CFG.Trainer.batch_size}_ImgNet_Cosine{CFG.Trainer.scheduler_factor}"
else:
    if CFG.Model.TPS == True:
        weights_name = f"{dataset}_Transformer_FC2_TPS_{CFG.Transformer.fc}_batch{CFG.Trainer.batch_size}_ImgNet_Cosine{CFG.Trainer.scheduler_factor}"
    else:
        weights_name = f"{dataset}_Transformer_FC2_{CFG.Transformer.fc}_batch{CFG.Trainer.batch_size}_ImgNet_Cosine{CFG.Trainer.scheduler_factor}"

save_file = f"./results_Transformer_ImgNet_Cosine/{weights_name}.pkl"
### TRAIN/EVAL ###
trainer = pl.Trainer(accelerator='gpu', devices= CFG.Trainer.devices, auto_select_gpus=False,
                    callbacks=[pl.callbacks.ModelCheckpoint(save_top_k = 1, dirpath = 'checkpoints_Transformer_ImgNet_Cosine',
                                 filename = weights_name  + '_epoch_{epoch:02d}_mae_{MAE/Val}',
                                 auto_insert_metric_name=False)],
                    logger=pl.loggers.TensorBoardLogger('logs', name=weights_name , default_hp_metric=False),
                    max_epochs=CFG.Trainer.num_epochs,
                    benchmark=True,
                    precision=32,
                    )

if CFG.Trainer.checkpoint_file:
    trainer.fit(model, train_loader, valid_loader, ckpt_path=CFG.Trainer.checkpoint_file)
else:
    trainer.fit(model, train_loader, valid_loader)
trainer.test(model, dataloaders=test_loader)
data = pd.DataFrame({"gt": model.labels_gt, "p": model.labels_p})
data.to_pickle(save_file)
print(data)

for step, batch in enumerate(train_loader, 1):
    # Perform the necessary training operations for each step
    
    # Monitoring the step
    if step % 10 == 0:  # Print the step every 10 steps (adjust as needed)
        print(f"Step: {step}/{len(train_loader)}")

# checkpoints = torch.load(checkpoint_file)['state_dict']
# model.load_state_dict(checkpoints)
# trainer.test(model, dataloaders=test_loader)
# mae_result = trainer.callback_metrics['MAE/Val']
# print(mae_result)
