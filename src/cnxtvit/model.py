import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics.functional as metrics
import timm
from einops import repeat
from torch.optim.lr_scheduler import LambdaLR
from .convnext import ConvNeXt
from .vit import Transformer, ViT


class MODEL(pl.LightningModule):
    def __init__(
            self,
            cfg
    ):
        super().__init__()

        if cfg.Model.name == "ConvNeXT_Transformer":
            self.model = ConvNeXT_Transformer(cfg)
        elif cfg.Model.name == "ConvNeXT":
            self.model = ConvNeXT_Model(cfg)
        elif cfg.Model.name == "Transformer":
            self.model = Transformer(cfg.Transformer)
        else:
            self.model = ConvNeXT_Transformer(cfg)


        self.regression = True

        if cfg.Trainer.loss_func == 'MAE':
            self.loss_func = nn.L1Loss()
        elif cfg.Trainer.loss_func == 'MSE':
            self.loss_func = nn.MSELoss()
        elif cfg.Trainer.loss_func == 'Adaptive':
            self.loss_func = self.adaptive_loss_function
        elif cfg.Trainer.loss_func == 'Huber':
            self.loss_func = nn.HuberLoss(reduction='mean', delta=1.0)
        elif cfg.Trainer.loss_func == 'KLDivLoss':
            self.loss_func = nn.KLDivLoss(reduction="batchmean")
        elif cfg.Trainer.loss_func == 'CrossEntropy':
            self.regression = False
            self.num_class = cfg.num_class
            self.loss_func = nn.CrossEntropyLoss()
        else:
            self.loss_func = nn.L1Loss()

        self.sigma = cfg.Trainer.sigma
        self.lr = cfg.Trainer.lr
        self.lr_min = cfg.Trainer.lr_min
        self.lr_patience = cfg.Trainer.lr_patience

        self.scheduler = cfg.Trainer.scheduler
        self.scheduler_factor = cfg.Trainer.scheduler_factor
        self.scheduler_total_steps = cfg.Trainer.scheduler_total_steps
        self.scheduler_warmup_steps = cfg.Trainer.scheduler_warmup_steps
        self.scheduler_interval = cfg.Trainer.scheduler_interval
        self.scheduler_monitor = cfg.Trainer.scheduler_monitor

        self.labels_p = []
        self.labels_gt = []

    def forward(self, images):
        return self.model(images)

    def adaptive_loss_function(self, labels, predicted):
        # https://www.sciencedirect.com/science/article/abs/pii/S0957417419306608
        loss = torch.mean((1 + self.sigma) * torch.pow(labels - predicted, 2) / (torch.abs(labels - predicted) + self.sigma))
        return loss

    def training_step(self, batch, batch_idx):
        images, labels = batch
        if self.regression:
            labels = labels.reshape(1, -1).t()
            output = self.forward(images)
            loss = self.loss_func(output, labels)
            mae = metrics.mean_absolute_error(output, labels)
            return {'loss': loss, 'mae': mae}
        else:
            output = self.forward(images)
            loss = self.loss_func(output, labels)
            top1 = metrics.accuracy(output, labels, task="multiclass", num_classes=self.num_class, top_k=1)
            top5 = metrics.accuracy(output, labels, task="multiclass", num_classes=self.num_class, top_k=5)
            return {'loss': loss, 'top1': top1, 'top5': top5}

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        if self.regression:
            labels = labels.reshape(1, -1).t()
            output = self.forward(images)
            loss = self.loss_func(output, labels)
            mae = metrics.mean_absolute_error(output, labels)
            return {'loss': loss, 'mae': mae}
        else:
            output = self.forward(images)
            loss = self.loss_func(output, labels)
            top1 = metrics.accuracy(output, labels, task="multiclass", num_classes=self.num_class, top_k=1)
            top5 = metrics.accuracy(output, labels, task="multiclass", num_classes=self.num_class, top_k=5)
            return {'loss': loss, 'top1': top1, 'top5': top5}

    def test_step(self, batch, batch_idx):
        images, labels = batch
        labels = labels.reshape(1, -1).t()
        output = self.forward(images)
        loss = self.loss_func(output, labels)
        mae = metrics.mean_absolute_error(output, labels)

        self.labels_p = self.labels_p + output.squeeze().tolist()
        self.labels_gt = self.labels_gt + labels.squeeze().tolist()
        return {"loss": loss, "mae": mae}


    def training_epoch_end(self, outs):
        if self.regression:
            loss = torch.stack([x['loss'] for x in outs]).mean()
            mae = torch.stack([x['mae'] for x in outs]).mean()
            self.log('Loss/Train', loss)
            self.log('MAE/Train', mae)
        else:
            loss = torch.stack([x['loss'] for x in outs]).mean()
            top1 = torch.stack([x['top1'] for x in outs]).mean()
            top5 = torch.stack([x['top5'] for x in outs]).mean()
            self.log('Loss/Train', loss)
            self.log('TOP1/Train', top1)
            self.log('TOP5/Train', top5)

    def validation_epoch_end(self, outs):
        if self.regression:
            loss = torch.stack([x['loss'] for x in outs]).mean()
            mae = torch.stack([x['mae'] for x in outs]).mean()
            self.log('Loss/Val', loss, prog_bar=True)
            self.log('MAE/Val', mae, prog_bar=True)
        else:
            loss = torch.stack([x['loss'] for x in outs]).mean()
            top1 = torch.stack([x['top1'] for x in outs]).mean()
            top5 = torch.stack([x['top5'] for x in outs]).mean()
            self.log('Loss/Val', loss, prog_bar=True)
            self.log('TOP1/Val', top1, prog_bar=True)
            self.log('TOP5/Val', top5, prog_bar=True)

    def test_epoch_end(self, outs):
        if self.regression:
            loss = torch.stack([x['loss'] for x in outs]).mean()
            mae = torch.stack([x['mae'] for x in outs]).mean()
            self.log('Loss/Test', loss, prog_bar=True)
            self.log('MAE/Test', mae, prog_bar=True)
        else:
            loss = torch.stack([x['loss'] for x in outs]).mean()
            top1 = torch.stack([x['top1'] for x in outs]).mean()
            top5 = torch.stack([x['top5'] for x in outs]).mean()
            self.log('Loss/Test', loss, prog_bar=True)
            self.log('TOP1/Test', top1, prog_bar=True)
            self.log('TOP5/Test', top5, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        interval = 'epoch'
        if self.scheduler == "OneCycleLR":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.lr,
                steps_per_epoch=self.scheduler_warmup_steps,
                epochs=self.scheduler_total_steps,
                anneal_strategy='cos',
                cycle_momentum=False
            )
        elif self.scheduler == "WarmupCosineSchedule":
            interval = 'step'
            scheduler = WarmupCosineSchedule(
                optimizer, 
                self.scheduler_warmup_steps, 
                self.scheduler_total_steps, 
                cycles=0.5, 
                last_epoch=-1)
        else:
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
                'interval': interval,
                'frequency': 1
            }
        }
    
class ConvNeXT_Model(nn.Module):
    def __init__(self, CFG):
        super().__init__()

        convNeXt = ConvNeXt(in_chans=CFG.ConvNeXT.in_chans, num_classes=CFG.ConvNeXT.num_classes, 
                depths=CFG.ConvNeXT.depths, dims=CFG.ConvNeXT.dims, drop_path_rate=CFG.ConvNeXT.drop_path_rate, 
                layer_scale_init_value=CFG.ConvNeXT.layer_scale_init_value, head_init_scale=CFG.ConvNeXT.head_init_scale)

        if CFG.ConvNeXT.pretrained:
            weights = torch.load(CFG.ConvNeXT.pretrained, map_location="cpu")['model']
            convNeXt.load_state_dict(weights)

        self.downsample_layers = convNeXt.downsample_layers
        self.stages = convNeXt.stages
        self.norm = convNeXt.norm

        self.output_size = self.forward_features(torch.Tensor(2, 3, 224, 224)).shape[1:]


        if CFG.ConvNeXT.fc:
            self.fc = nn.Sequential(nn.Linear(self.output_size[-1], CFG.ConvNeXT.fc),
                                    nn.Linear(CFG.ConvNeXT.fc, CFG.ConvNeXT.num_classes))
        else:
            self.fc = nn.Linear(self.output_size[-1], CFG.ConvNeXT.num_classes)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))

    def forward(self, x):
        x = self.forward_features(x)
        return self.fc(x)

class Transformer(nn.Module):
    def __init__(self, CFG):
        super().__init__()

        self.vit = timm.create_model(CFG.backbone, pretrained=CFG.pretrained)

        self.output_size = [self.vit.head.in_features]


        if CFG.fc:
            self.vit.head = nn.Sequential(nn.LayerNorm(self.output_size[-1]),
                                    nn.Linear(self.output_size[-1], CFG.fc),
                                    nn.Linear(CFG.fc, CFG.num_class))
        else:
            self.vit.head = nn.Sequential(nn.LayerNorm(self.output_size[-1]), nn.Linear(self.output_size[-1], CFG.num_class))


        if CFG.transfer_learning:
            if CFG.fc:
                self.vit.head = nn.Sequential(nn.LayerNorm(self.output_size[-1]),
                                    nn.Linear(self.output_size[-1], CFG.fc),
                                    nn.Linear(CFG.fc, CFG.num_class))
            else:
                self.vit.head = nn.Sequential(nn.LayerNorm(self.output_size[-1]), nn.Linear(self.output_size[-1], CFG.num_class))


    def forward(self, x):
        return self.vit(x)


class Transformer_Model(nn.Module):
    def __init__(self, CFG):
        super().__init__()

        self.vit = ViT(image_size=CFG.image_size, patch_size=CFG.patch_size, num_classes=CFG.num_class, 
                    dim=CFG.dim, depth=CFG.depth, heads=CFG.heads, mlp_dim=CFG.mlp_dim, 
                    pool=CFG.pool, channels=CFG.channels, dim_head=CFG.dim_head, 
                    dropout=CFG.dropout, emb_dropout=CFG.emb_dropout)

        self.output_size = self.vit(torch.Tensor(2, 3, 224, 224)).shape[1:]


        if CFG.fc:
            self.fc = nn.Sequential(nn.LayerNorm(self.output_size[-1]),
                                    nn.Linear(self.output_size[-1], CFG.fc),
                                    nn.Linear(CFG.fc, CFG.num_class))
        else:
            self.fc = nn.Sequential(nn.LayerNorm(self.output_size[-1]), nn.Linear(self.output_size[-1], CFG.num_class))

        if CFG.pretrained:
            weights = torch.load(CFG.pretrained, map_location="cpu")['state_dict']
            self.vit.load_state_dict(weights, strict=False)

        if CFG.transfer_learning:
            if CFG.fc:
                self.fc = nn.Sequential(nn.LayerNorm(self.output_size[-1]),
                                    nn.Linear(self.output_size[-1], CFG.fc),
                                    nn.Linear(CFG.fc, CFG.num_class))
            else:
                self.fc = nn.Sequential(nn.LayerNorm(self.output_size[-1]), nn.Linear(self.output_size[-1], CFG.num_class))


    def forward(self, x):
        x = self.vit(x)
        return self.fc(x)

    def init(self):
        for m in self.vit.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class ConvNeXT_Transformer(nn.Module):
    def __init__(self, CFG):
        super().__init__()
        self.ConvNeXt = ConvNeXT_Model(CFG)

        if CFG.ConvNeXT.weights:
            weights = torch.load(CFG.ConvNeXT.weights, map_location="cpu")
            if 'state_dict' in weights.keys():
                weights = weights['state_dict']

            fixed_weights = {}
            for key in weights.keys():
                fixed_key = key.replace('model.', '')
                fixed_weights[fixed_key] = weights[key]
            self.ConvNeXt.load_state_dict(fixed_weights)

        self.Transformer = Transformer(CFG.Transformer)
        

        if CFG.Transformer.weights:
            weights = torch.load(CFG.Transformer.weights, map_location="cpu")
            if 'state_dict' in weights.keys():
                weights = weights['state_dict']

            fixed_weights = {}
            for key in weights.keys():
                fixed_key = key.replace('model.', '')
                fixed_weights[fixed_key] = weights[key]
            self.Transformer.load_state_dict(fixed_weights)

        self.Transformer.vit.patch_embed = nn.Identity()

    def forward(self, x):
        x = self.forward_convnext(x)
        x = self.forward_transformer(x)

        return x

    def forward_convnext(self, x):
        for i in range(4):
            x = self.ConvNeXt.downsample_layers[i](x)
            x = self.ConvNeXt.stages[i](x)
        x = torch.reshape(x, (x.shape[0], 192, 196)).permute(0, 2, 1)
        return x

    def forward_transformer(self, x):
        x = self.Transformer(x)

        return x

class WarmupCosineSchedule(LambdaLR):
    """Linear warmup and then cosine decay.
    Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
    Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps
    following a cosine curve.
    If `cycles` (default=0.5) is different from default, learning rate follows
    cosine function after warmup.
    """

    def __init__(self, optimizer, warmup_steps, t_total, cycles=0.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))

