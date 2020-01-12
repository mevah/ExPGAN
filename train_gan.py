  
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import argparse
import numpy as np
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torch.autograd import Variable
import math
import os
from tensorboardX import SummaryWriter
import time
import datetime

from gan_model import CS_Dataset
from gan_model import LeftDiscriminator, RightDiscriminator, ExPGenerator


parser = argparse.ArgumentParser()
parser.add_argument("--num_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--mini_D_num_epochs", type=int, default=1, help="number of epochs of training D")
parser.add_argument("--mini_G_num_epochs", type=int, default=1, help="number of epochs of training G")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--lr_gen", type=float, default=0.0002, help="adam: learning rate for the generator")
parser.add_argument("--lr_disc", type=float, default=0.002, help="adam: learning rate for the discriminator")
parser.add_argument("--lambda_seg", type=float, default=1.0, help="loss scale term for segmentation loss")
parser.add_argument("--lambda_disc", type=float, default=1.0, help="loss scale term for discriminator")
parser.add_argument("--lambda_recon", type=float, default=1.0, help="loss scale term for reconstruction")
parser.add_argument("--b1_gen", type=float, default=0.5, help="adam: decay of first order momentum of gradient, for the generator")
parser.add_argument("--b2_gen", type=float, default=0.999, help="adam: decay of first order momentum of gradient, for the generator")
parser.add_argument("--b1_disc", type=float, default=0.5, help="adam: decay of first order momentum of gradient, for the discriminator")
parser.add_argument("--b2_disc", type=float, default=0.999, help="adam: decay of first order momentum of gradient, for the discriminator")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
parser.add_argument("--dataset_folder", type=str, default='/cluster/scratch/takmaza/DL', help="dataset folder, directory which includes left8bit and gtFine folders")
parser.add_argument("--model_save", type=str, default='/cluster/scratch/takmaza/DL/', help='specify the directory to save models')
parser.add_argument("--log_frequency", type=int, default=20, help="log frequency in terms of steps")
parser.add_argument("--logfile_name", type=str, default='logs.txt')
parser.add_argument("--model_load", type=str, default="")

opt = parser.parse_args()       


def get_model_name(opt):
    writer_log_dir = os.path.join(opt.model_save, 'models')
    curr_time = datetime.datetime.now()
    writer_log_dir = os.path.join(writer_log_dir, str(curr_time.year) + '-' + str('%02d' %curr_time.month) + '-' + str('%02d' %curr_time.day) + '-' + str('%02d' %curr_time.hour) +str('%02d' %curr_time.minute)+ str('%02d' %curr_time.second)) #'_' + str(args.lr) 
    if not os.path.exists(writer_log_dir):
        os.makedirs(writer_log_dir)
    return writer_log_dir

model_save_dir = get_model_name(opt)

def save_opts(args):
    """Save options to disk
    """
    models_dir = model_save_dir #os.path.join(model_save_dir, "model")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    to_save = args.__dict__.copy()

    with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
        json.dump(to_save, f, indent=2)
        
def logging(str, log_file=opt.logfile_name, to_stdout=True):
    with open(os.path.join(model_save_dir, log_file), 'a') as f:
        f.write(str + '\n')
    if to_stdout:
        print(str)
        
def to_categorical_np(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

def to_categorical_torch(y, num_columns):
    """Returns one-hot encoded Variable"""
    y_cat = np.zeros((y.shape[0], num_columns))
    y_cat[range(y.shape[0]), y] = 1.0

    return Variable(FloatTensor(y_cat))

def write_losses(loss, loss_seg,loss_D_left,loss_D_right,loss_recon_left,loss_recon_right):
    losses = {}
    losses["losses/loss_seg"] = loss_seg
    losses["losses/loss_D_left"] = loss_D_left
    losses["losses/loss_D_right"] = loss_D_right
    losses["losses/loss_recon_left"] = loss_recon_left
    losses["losses/loss_recon_right"] = loss_recon_right
    losses["total_loss"] = loss
    return losses

def call_logger(batch_idx, opt, total_step, mode="train"):
    early_phase = batch_idx % opt.log_frequency == 0 and total_step < 2000
    late_phase = total_step % 2000 == 0
    return (early_phase or late_phase)
       
def save_model(left_D, right_D, generator_G):
    logging("Saving models to {} ".format(model_save_dir))
    torch.save({
        "left_disc": left_D.state_dict(),
        "right_disc": right_D.state_dict(),
        "generator": generator_G.state_dict()
        },
        os.path.join(model_save_dir, "model.pt"))
    
def load_model(exp_img_shape):
    logging('Loading models from {} '.format(opt.model_load))
    #loaded = torch.load(os.path.join(model_save_dir, "model.pt"))
    loaded = torch.load(opt.model_load)
    
    left_D = LeftDiscriminator(exp_img_shape)
    right_D = RightDiscriminator(exp_img_shape)
    generator_G = ExPGenerator()
    
    left_D.load_state_dict(loaded.get('left_disc'))
    right_D.load_state_dict(loaded.get('right_disc'))
    generator_G.load_state_dict(loaded.get('generator'))
    return left_D, right_D, generator_G


def img_denorm(img):
    mean = np.asarray([0.28689554, 0.32513303, 0.28389177])
    std = np.asarray([0.18696375, 0.19017339, 0.18720214])
        
    denormalize = transforms.Normalize((-1 * mean / std), (1.0 / std))

    res = img.squeeze(0)
    res = denormalize(res)

    #Image needs to be clipped since the denormalize function will map some
    #values below 0 and above 1
    res = torch.clamp(res, 0, 1)
    return(res)

        
def log_tbx(writers, mode, batch, outputs, losses, total_step):
    """
    Write an event to the tensorboard events file
    """
    writer = writers[mode]
    for l, v in losses.items():
        writer.add_scalar("{}".format(l), v, total_step)

    for j in range(min(4, opt.batch_size)):  
        writer.add_image(
            "input/{}".format(j),
            img_denorm(batch[("img",0)][j]).data, total_step)

        writer.add_image(
            "generated/{}".format(j),
            img_denorm(outputs[("generated")][j]), total_step)

        writer.add_image(
            "generated_normalized/{}".format(j),
            outputs[("generated")][j], total_step)


### MAIN

logging(str(opt))
exp_img_shape = (3, 256, 384)
cuda = True if torch.cuda.is_available() else False
print('cuda status: ', cuda)
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


train_dataset = CS_Dataset(opt.dataset_folder, split='train', mode='fine', target_type='semantic', transform=None, target_transform=None, transforms=None)
val_dataset = CS_Dataset(opt.dataset_folder, split='val', mode='fine', target_type='semantic', transform=None, target_transform=None, transforms=None)

train_loader = DataLoader(
    train_dataset, opt.batch_size, True,
    num_workers=1, pin_memory=True, drop_last=True)

val_loader = DataLoader(
    val_dataset, opt.batch_size, True,
    num_workers=1, pin_memory=True, drop_last=True)


if opt.model_load == "":
    left_D = LeftDiscriminator(exp_img_shape)
    right_D = RightDiscriminator(exp_img_shape)
    generator_G = ExPGenerator()
else:
    left_D, right_D, generator_G = load_model(exp_img_shape)

if cuda:
    left_D.cuda()
    right_D.cuda()
    generator_G.cuda()
    
    
optimizer_G = torch.optim.Adam(generator_G.parameters(), lr=opt.lr_gen, betas=(opt.b1_gen, opt.b2_gen))
optimizer_D_left = torch.optim.Adam(left_D.parameters(), lr=opt.lr_disc, betas=(opt.b1_disc, opt.b2_disc))
optimizer_D_right = torch.optim.Adam(right_D.parameters(), lr=opt.lr_disc, betas=(opt.b1_disc, opt.b2_disc))

adversarial_loss = torch.nn.BCELoss()

#### BECAME OLD NOW
def add_noise(ins, is_training=True, mean=0, stddev=0.01):
    if is_training:
        noise = Variable(ins.data.new(ins.size()).normal_(mean, stddev))
        return ins + noise
    return ins

def mask_segm(full_segm, cropped_segm):
    return full_segm

####Define weight matrix for segmentation loss
mask_row = np.zeros((1,512))
def sigmoid(start=0,end=128, c1=0.1,c2=0):
    x = np.arange(start,end)
    return (1 / (1 + np.exp(-1 * c1 * (x-c2))))    
mask_row[0,0:128] = sigmoid(0,128,c2=64)
mask_row[0,128:256] = 1-sigmoid(128,256,c2=192)
mask_row[0,256:384] = sigmoid(256,384,c2=320)
mask_row[0,384:512] =  1-sigmoid(384,512,c2=448)
mask_tensor = torch.from_numpy(np.squeeze(mask_row)).float()
mask_tensor_rec_left = torch.from_numpy(np.squeeze(mask_row[:,0:128])).float()
mask_tensor_rec_right = torch.from_numpy(np.squeeze(mask_row[:,384:512])).float()


total_step = 0
mini_D_num_epochs = opt.mini_D_num_epochs
mini_G_num_epochs = opt.mini_G_num_epochs

writers = {}
for mode in ["train", "val"]:
    writers[mode] = SummaryWriter(os.path.join(model_save_dir, mode))
    
best_val_loss = None

#Make networks multigpu available
if torch.cuda.device_count() > 1:
    print("There are", torch.cuda.device_count(), "gpus available.")
    left_D = nn.DataParallel(left_D)
    right_D= nn.DataParallel(right_D)
    generator_G = nn.DataParallel(generator_G)

print('Switched to training mode')
for epoch in range(opt.num_epochs):
    print('#'*30)
    print('Epoch:', epoch)
    #Switch models to training mode
    left_D.train()
    right_D.train()
    generator_G.train()
    for batch_idx, batch in enumerate(train_loader):
        weight_segmentation = mask_tensor.repeat(batch['cropped'].size(0), 256, 1)
        weight_rec_left = mask_tensor_rec_left.repeat(batch['cropped'].size(0), 3, 256, 1)
        weight_rec_right = mask_tensor_rec_right.repeat(batch['cropped'].size(0), 3, 256, 1)

        if cuda: 
            weight_segmentation = weight_segmentation.cuda()
            weight_rec_left = weight_rec_left.cuda()
            weight_rec_right = weight_rec_right.cuda()

        #if batch_idx % 200 == 0:
        #    print('batch_idx: ',  batch_idx)
        #print('#'*20)
        for epoch_D in range(mini_D_num_epochs):
            #print('epoch_D: ', epoch_D)
            source_img = batch["cropped"]
            source_segm = batch["cropped_segm"]
            true_im = add_noise(batch["img",0]) #Before cropping original image  noise needs to be added 
            source_img = source_img.cuda()
            source_segm = source_segm.cuda()
            true_im = true_im.cuda()
            gen_fake_left, gen_fake_right, gen_fake_seg = generator_G(source_img, source_segm)

            fake_left = torch.cat((gen_fake_left.detach(), source_img), dim=3)
            fake_right= torch.cat((source_img, gen_fake_right.detach()), dim=3)
            true_left = true_im[:, : , :, 0:exp_img_shape[-1]]
            true_right = true_im[:, :, : , -exp_img_shape[-1]:]

            true_lbl = Variable(Tensor(source_img.size(0), 1).fill_(1), requires_grad=False)
            fake_lbl = Variable(Tensor(source_img.size(0), 1).fill_(0), requires_grad=False)
            true_lbl = true_lbl.cuda()
            fake_lbl = fake_lbl.cuda()

            lbls = torch.squeeze(torch.cat((true_lbl, fake_lbl), dim=0))
            left_imgs = torch.cat((true_left, fake_left), dim=0)
            right_imgs = torch.cat((true_right, fake_right), dim=0)

            lbl_est_left = left_D(left_imgs)
            loss_D_left = adversarial_loss(torch.squeeze(lbl_est_left), torch.squeeze(lbls))

            optimizer_D_left.zero_grad()
            loss_D_left.backward()
            optimizer_D_left.step()

            lbl_est_right = right_D(right_imgs)
            loss_D_right =adversarial_loss(torch.squeeze(lbl_est_right), torch.squeeze(lbls))

            optimizer_D_right.zero_grad()
            loss_D_right.backward()
            optimizer_D_right.step()

        for epoch_G in range(mini_G_num_epochs):
            #print('epoch_G: ', epoch_G)
            source_img = batch["cropped"]
            source_segm = batch["cropped_segm"]
            true_im = batch["img",0]
            true_segm = batch["segm",0]
            source_img = source_img.cuda()
            source_segm = source_segm.cuda()
            true_im = true_im.cuda()
            true_segm = true_segm.cuda()
            #print(true_segm.shape)
            masked_target = mask_segm(true_segm, source_segm).to(torch.int64)
            orj_left = true_im[:,:,:, 0:128]
            orj_right = true_im[:,:,:, -128:]

            gen_fake_left, gen_fake_right, gen_fake_seg = generator_G(source_img, source_segm)

            fake_left = torch.cat((gen_fake_left, source_img), dim=3)
            fake_right= torch.cat((source_img, gen_fake_right), dim=3)
            true_lbl = Variable(Tensor(source_img.size(0), 1).fill_(1.0), requires_grad=False)
            true_lbl = true_lbl.cuda()
            #GET DISCRIMINATOR RESULTS
            lbl_est_left = left_D(fake_left)
            lbl_est_right = right_D(fake_right)
            
            optimizer_G.zero_grad()
            #print(torch.squeeze(gen_fake_seg).shape)
            #print(torch.squeeze(masked_target).shape)
            
            #print('SHAPE:')
            #print('gen_fake_seg', torch.squeeze(gen_fake_seg).shape)
            #print('masked_target_before_max', torch.squeeze(masked_target).shape)
            _, masked_target = masked_target.max(dim=1)
            #print('masked_target_after_max', torch.squeeze(masked_target).shape)
            #nn.CrossEntropyLoss()(out, Variable(targets))
            #print(torch.squeeze(gen_fake_seg).shape)
            #print(torch.squeeze(masked_target).shape)

            loss_seg = nn.CrossEntropyLoss(reduction='none')(torch.squeeze(gen_fake_seg), torch.squeeze(masked_target))
            loss_seg = weight_segmentation*loss_seg # Ensure that weights are scaled appropriately
            loss_seg = torch.mean(loss_seg) # Sums the loss per image

            #loss_seg = F.cross_entropy(torch.squeeze(gen_fake_seg), torch.squeeze(masked_target))

            loss_D_left = adversarial_loss(torch.squeeze(lbl_est_left), torch.squeeze(true_lbl))
            loss_D_right = adversarial_loss(torch.squeeze(lbl_est_right), torch.squeeze(true_lbl))

            #print(loss_seg.requires_grad)
                
            loss_recon_left = nn.MSELoss(reduction='none')(gen_fake_left, orj_left)
            #print(loss_recon_left.shape)
            #print(weight_rec_left.shape)
            loss_recon_left = weight_rec_left*loss_recon_left # Ensure that weights are scaled appropriately
            loss_recon_left = torch.mean(loss_recon_left) # Sums the loss per image
            loss_recon_right = nn.MSELoss(reduction='none')(gen_fake_right, orj_right)
            loss_recon_right = weight_rec_right*loss_recon_right # Ensure that weights are scaled appropriately
            loss_recon_right = torch.mean(loss_recon_right) # Sums the loss per image

            #loss_recon_left = F.mse_loss(gen_fake_left, orj_left)
            #loss_recon_right = F.mse_loss(gen_fake_right, orj_right)

            loss = opt.lambda_seg * loss_seg - opt.lambda_disc * loss_D_left - opt.lambda_disc * loss_D_right + opt.lambda_recon*loss_recon_left + opt.lambda_recon*loss_recon_right
            losses = write_losses(loss, loss_seg,loss_D_left,loss_D_right,loss_recon_left,loss_recon_right)
            loss.backward()
            optimizer_G.step()
        total_step += 1
        
        if call_logger(batch_idx, opt, total_step):
            outputs = {}
            outputs['generated']= torch.cat((fake_left,gen_fake_right), dim=3)
            log_tbx(writers, "train", batch, outputs, losses, total_step)

    #IMPLEMENT VALIDATION
    #Switch models to evaluation mode
    print('Switched to eval mode')
    left_D.eval()
    right_D.eval()
    generator_G.eval()
    val_loss = 0
    for batch_idx, batch in enumerate(val_loader):
        weight_segmentation = mask_tensor.repeat(batch['cropped'].size(0), 256, 1)
        weight_rec_left = mask_tensor_rec_left.repeat(batch['cropped'].size(0), 3, 256, 1)
        weight_rec_right = mask_tensor_rec_right.repeat(batch['cropped'].size(0), 3, 256, 1)

        if cuda: 
            weight_segmentation = weight_segmentation.cuda()
            weight_rec_left = weight_rec_left.cuda()
            weight_rec_right = weight_rec_right.cuda()

        #print('epoch_G: ', epoch_G)
        source_img = batch["cropped"]
        source_segm = batch["cropped_segm"]
        true_im = batch["img",0]
        true_segm = batch["segm",0]
        source_img = source_img.cuda()
        source_segm = source_segm.cuda()
        true_im = true_im.cuda()
        true_segm = true_segm.cuda()
        #print(true_segm.shape)
        masked_target = mask_segm(true_segm, source_segm).to(torch.int64)
        orj_left = true_im[:,:,:, 0:128]
        orj_right = true_im[:,:,:, -128:]

        gen_fake_left, gen_fake_right, gen_fake_seg = generator_G(source_img, source_segm)

        fake_left = torch.cat((gen_fake_left, source_img), dim=3)
        fake_right= torch.cat((source_img, gen_fake_right), dim=3)
        true_lbl = Variable(Tensor(source_img.size(0), 1).fill_(1.0), requires_grad=False)
        true_lbl = true_lbl.cuda()
        #GET DISCRIMINATOR RESULTS
        lbl_est_left = left_D(fake_left)
        lbl_est_right = right_D(fake_right)

        optimizer_G.zero_grad()
        #print(torch.squeeze(gen_fake_seg).shape)
        #print(torch.squeeze(masked_target).shape)

        _, masked_target = masked_target.max(dim=1)
        #nn.CrossEntropyLoss()(out, Variable(targets))
        loss_seg = nn.CrossEntropyLoss(reduction='none')(torch.squeeze(gen_fake_seg), torch.squeeze(masked_target))
        loss_seg = weight_segmentation*loss_seg # Ensure that weights are scaled appropriately
        loss_seg = torch.mean(loss_seg) # Sums the loss per image

        #loss_seg = F.cross_entropy(torch.squeeze(gen_fake_seg), torch.squeeze(masked_target))

        loss_D_left = adversarial_loss(torch.squeeze(lbl_est_left), torch.squeeze(true_lbl))
        loss_D_right = adversarial_loss(torch.squeeze(lbl_est_right), torch.squeeze(true_lbl))

        #print(loss_seg.requires_grad)

        loss_recon_left = nn.MSELoss(reduction='none')(gen_fake_left, orj_left)
        loss_recon_left = weight_rec_left*loss_recon_left # Ensure that weights are scaled appropriately
        loss_recon_left = torch.mean(loss_recon_left) # Sums the loss per image
        loss_recon_right = nn.MSELoss(reduction='none')(gen_fake_right, orj_right)
        loss_recon_right = weight_rec_right*loss_recon_right # Ensure that weights are scaled appropriately
        loss_recon_right = torch.mean(loss_recon_right) # Sums the loss per image

        #loss_recon_left = F.mse_loss(gen_fake_left, orj_left)
        #loss_recon_right = F.mse_loss(gen_fake_right, orj_right)

        loss = opt.lambda_seg * loss_seg - opt.lambda_disc * loss_D_left - opt.lambda_disc * loss_D_right + opt.lambda_recon*loss_recon_left + opt.lambda_recon*loss_recon_right
        
        batch_loss = loss.item()
        val_loss+=batch_loss

    logging("Epoch {}: validation_loss={}, best_validation_loss={} ".format(epoch, val_loss ,best_val_loss))
    
    # last batch
    losses = write_losses(loss, loss_seg,loss_D_left,loss_D_right,loss_recon_left,loss_recon_right)
    outputs = {}
    outputs['generated']= torch.cat((fake_left,gen_fake_right), dim=3)
    log_tbx(writers, "val", batch, outputs, losses, total_step)

    if best_val_loss is None or val_loss < best_val_loss: 
        best_val_loss = val_loss
        #save_model(left_D, right_D, generator_G)

    if epoch % 5 == 0:
        save_model(left_D, right_D, generator_G)

# log


