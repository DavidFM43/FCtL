#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function

import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
from dataset.deep_globe import DeepGlobe, classToRGB, is_image_file
from utils.loss import FocalLoss
from utils.lr_scheduler import LR_Scheduler
# from tensorboardX import SummaryWriter
from helper import create_model_load_weights, get_optimizer, Trainer, Evaluator, collate, collate_test
from option import Options
import wandb
from kaggle_secrets import UserSecretsClient
wandb_key = UserSecretsClient().get_secret("wandb_key")

import warnings
warnings.filterwarnings("ignore")

args = Options().parse()
dataset = args.dataset
if dataset == 1:
    args.n_class = 7 
    args.data_path = "data/"
    args.model_path = "saved_models/"
    args.log_path = "uns/"
else:
    pass
n_class = args.n_class 
print("n_class:",n_class)

torch.backends.cudnn.deterministic = True
data_path = args.data_path #data
model_path = args.model_path #saved_models
log_path = args.log_path #log
if not os.path.isdir(model_path): os.mkdir(model_path)
if not os.path.isdir(log_path): os.mkdir(log_path)
print("data_path:",data_path , "model_path:",model_path, "log_path",log_path)

task_name = "train_mode_0"
print("task_name:",task_name)

mode = 0 
train = True 
val = True 
print("mode:",mode, "train:",train, "val:",val)

###################################
print("preparing datasets and dataloaders......")
batch_size = args.batch_size 
num_worker = os.cpu_count() # args.num_worker
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ids_train = [image_name for image_name in os.listdir(os.path.join(data_path, "train", "Sat")) if is_image_file(image_name)]
ids_test = [image_name for image_name in os.listdir(os.path.join(data_path, "test", "Sat")) if is_image_file(image_name)]
ids_val = [image_name for image_name in os.listdir(os.path.join(data_path, "crossvali", "Sat")) if is_image_file(image_name)]

dataset_train = DeepGlobe(dataset, os.path.join(data_path, "train"), ids_train, label=True, transform=True)
dataloader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch_size, num_workers=num_worker, collate_fn=collate, shuffle=True, pin_memory=True)
dataset_test = DeepGlobe(dataset, os.path.join(data_path, "test"), ids_test, label=False)
dataloader_test = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=batch_size, num_workers=num_worker, collate_fn=collate_test, shuffle=False, pin_memory=True)
dataset_val = DeepGlobe(dataset, os.path.join(data_path, "crossvali"), ids_val, label=True)
dataloader_val = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=batch_size, num_workers=num_worker, collate_fn=collate, shuffle=False, pin_memory=True)
print('train_len:',len(dataset_train)) 
print('test_len:',len(dataset_test)) 
print('val_len:',len(dataset_val))  

##### sizes are (w, h) ##############################
size_p = (args.size_p, args.size_p) 
size_g = (args.size_g, args.size_g) 
context = args.context # context
sub_batch_size = args.sub_batch_size 
###################################
print("creating models......")

pre_path = os.path.join(model_path, "")
c_path = os.path.join(model_path, "")
glo_path = os.path.join(model_path, "")
print("pre_path:", pre_path, "c_path:", c_path, "glo_path:", glo_path)
model, c_fixed, global_fixed = create_model_load_weights(n_class, pre_path, glo_path, c_path, mode)
###################################
num_epochs = args.num_epochs 
lens = args.lens
start = args.start
learning_rate = args.lr 

optimizer = get_optimizer(model, learning_rate)
scheduler = LR_Scheduler('poly', learning_rate, num_epochs, len(dataloader_train)) 
##################################

criterion1 = FocalLoss(gamma=3)
criterion = lambda x,y: criterion1(x, y)

if val:
#     writer = SummaryWriter(log_dir=log_path + task_name) 
#     f_log = open(log_path + task_name + ".log", 'w') 
    wandb.login(key=wandb_key)
    wandb.init(
        tags=["FCtL"],
        entity="landcover-classification",
        notes="",
        project="ml-experiments",
    )

trainer = Trainer(criterion, optimizer, n_class, size_p, size_g, sub_batch_size, mode, dataset, context)
evaluator = Evaluator(n_class, size_p, size_g, sub_batch_size, mode, train, dataset, context)

# prediction_path = "/home/azureuser/cloudfiles/code/Users/davidfelipemr/FCtL/prediction/"
best_pred = 0.0
print("start training......")
from pprint import pprint
for epoch in range(start, start + lens):
    if not train:
        break
    trainer.set_train(model)
    optimizer.zero_grad()
    tbar = tqdm(dataloader_train); train_loss = 0
    for i_batch, sample_batched in enumerate(tbar):
        scheduler(optimizer, i_batch, epoch, best_pred)  # update lr
        loss = trainer.train(sample_batched, model, c_fixed, global_fixed)
        train_loss += loss.item()
        # score_train, iou = trainer.get_scores()
        # wandb
        wandb.log({"train/loss": loss.item()})
        
    score_train, iou = trainer.get_scores()
    trainer.reset_metrics()
    # print("Iou Metrics mine")
    # pprint(iou)
    # print("Iou Metrics them")
    # pprint(score_train)
    # wandb
    wandb.log({"epoch": epoch, "train/mean_loss": train_loss / (i_batch + 1), **iou})
    
    cnt = 5 
    # if epoch >= 34:
    #     cnt = 1
    if (epoch+1) % cnt == 0:
        torch.save(model.state_dict(), model_path + task_name + ".epoch" + str(epoch) + ".pth")

    if (epoch) % 1 == 0:
        with torch.no_grad():
            print("evaluation...")
            model.eval()
            tbar = tqdm(dataloader_val)
            for i_batch, sample_batched in enumerate(tbar):
                predictions = evaluator.eval_test(sample_batched, model, c_fixed, global_fixed)
                # score_val = evaluator.get_scores()
                # use [1:] since class0 is not considered in deep_globe metric
                # tbar.set_description('mIoU: %.3f' % (np.mean(np.nan_to_num(score_val["iou"])[:-1])))
                # images = sample_batched['image']
                # labels = sample_batched['label'] # PIL images

#                 if i_batch * batch_size + len(images) > (epoch % len(dataloader_val)) and i_batch * batch_size <= (epoch % len(dataloader_val)):
#                     writer.add_image('image', transforms.ToTensor()(images[(epoch % len(dataloader_val)) - i_batch * batch_size]), epoch)
#                     writer.add_image('mask', classToRGB(dataset, np.array(labels[(epoch % len(dataloader_val)) - i_batch * batch_size])) , epoch)
#                     writer.add_image('prediction', classToRGB(dataset, predictions[(epoch % len(dataloader_val)) - i_batch * batch_size]), epoch)

#             #torch.save(model.state_dict(), model_path + task_name + ".epoch" + str(epoch) + ".pth")

            score_val, iou = evaluator.get_scores()
            evaluator.reset_metrics()
            # print("Iou Metrics mine")
            # pprint(iou)
            # print("Iou Metrics them")
            # pprint(score_val)
            # wandb
            wandb.log({"epoch": epoch, **iou})

#             if np.mean(np.nan_to_num(score_val["iou"][:-1])) > best_pred: best_pred = np.mean(np.nan_to_num(score_val["iou"][:-1]))
#             log = ""
#             log = log + 'epoch [{}/{}] IoU: train = {:.4f}, val = {:.4f}'.format(epoch+1, num_epochs, np.mean(np.nan_to_num(score_train["iou"][:-1])), np.mean(np.nan_to_num(score_val["iou"][:-1]))) + "\n"
#             log = log + "train: " + str(score_train["iou"]) + "\n"
#             log = log + "val:" + str(score_val["iou"]) + "\n"
#             log += "================================\n"
#             print(log)

#             f_log.write(log)
#             f_log.flush()
#             writer.add_scalars('IoU', {'train iou': np.mean(np.nan_to_num(score_train["iou"][:-1])), 'validation iou': np.mean(np.nan_to_num(score_val["iou"][:-1]))}, epoch)
# if val: f_log.close()

# if not train:
#     with torch.no_grad():
#         print("testing...")
#         model.eval()
#         tbar = tqdm(dataloader_test)
#         for i_batch, sample_batched in enumerate(tbar):
#             predictions = evaluator.eval_test(sample_batched, model, c_fixed, global_fixed)

#             images = sample_batched['image']
#             if not os.path.isdir(prediction_path):
#                 os.mkdir(prediction_path)
#             for i in range(len(images)):
#                 transforms.functional.to_pil_image(classToRGB(dataset, predictions[i])).save(
#                     prediction_path + sample_batched["id"][i] + "_mask.png"
#                 )