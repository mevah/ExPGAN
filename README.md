# ExPGAN
## ExPGAN: Segmentation Guided Image Outpainting<br>


### Data Preparation Instructions
Training and evaluation of our model is performed using the Cityscapes dataset.<br>
Download the Cityscapes images and segmentations from the following links: <br>
#### Ground-truth segmentations:
gtFine_trainvaltest.zip (241MB)<br>
https://www.cityscapes-dataset.com/file-handling/?packageID=1<br>
#### Images:
leftImg8bit_trainvaltest.zip (11GB)<br>
https://www.cityscapes-dataset.com/file-handling/?packageID=3<br>
<br>
After downloading these two folders, unzip them and put them into the same directory, with the following folder structure:<br>

/path/to/dataset   <br>
&nbsp;&nbsp;&nbsp;&nbsp;|__ left8bit      <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|__ train      <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|__ val        <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|__ test       <br>
&nbsp;&nbsp;&nbsp;&nbsp;|__ gtFine        <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|__ train      <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|__ val        <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|__ test       <br>

Create 2 new sub
Please move the *frankfurt* subdirectories inside *left8bit/val* and *gtFine/val* directories to *left8bit/ourtest* and *gtFine/ourtest* directories, as we do not use the images from the *frankfurt* folder during validation, instead we use them only for the evaluation of our method.

## Training
For two different types of experiments, two different training scripts, ``train_gan.py`` and ``train_gan_3phase.py`` are used. 

We have performed our experiments with the following:<br>
``
Python 3.7.1
CUDA 10.0
torch=1.3.1
torchvision=0.2.0
``<br>
Training approximately lasts for 25 hours.

### train_gan_3phase.py
Start training using the following command: <br>
``
train_gan_3phase.py --dataset_folder=/path/to/dataset/ --model_save=/path/to/log/directory
``

On Leonhard, you can run the following command: <br>
``
bsub -n 8 -W 24:00 -o training_log -R "rusage[mem=30G, ngpus_excl_p=1]" python train_gan_3phase.py   
--dataset_folder=/path/to/dataset/ --model_save=/path/to/log/directory
``

### train_gan.py
Start training using the following command: <br>
``
train_gan_3phase.py --dataset_folder=/path/to/dataset/ --model_save=/path/to/log/directory
``

On Leonhard, you can run the following command: <br>
``
bsub -n 8 -W 24:00 -o training_log -R "rusage[mem=30G, ngpus_excl_p=1]" python train_gan_3phase.py          
--dataset_folder=/path/to/dataset/ --model_save=/path/to/log/directory
``

In order to repeat the ablation study regarding the segmentation loss, you can add the following argument to set the segmentation loss contribution to zero: <br>``--lambda_seg=0``.

#### Loading Pre-Trained ExpGAN Model for Continuing the Training
If you would like to continue training from one of the ExpGAN models we have previously trained, download the model, and add the argument <br>
``
--model_load=/path/to/pretrained/model/folder
``




## Test
In order to generate images and evaluate the trained model, you can run the ``test.py`` script.
Download the segmentation model we have previously trained from this link: <br>
https://drive.google.com/drive/folders/1nDIsfuWHrAA4V6fnJ0PNW_bnJ1ao8SwE?usp=sharing <br>
Then, move this segmentation model with the name ``fcn8s_cityscapes_best_model.pkl`` to an arbitrary directory ``/path/to/pretrained/segmentation/model``.

Download the pre-trained ExpGAN model from this link:
https://drive.google.com/drive/folders/1nDIsfuWHrAA4V6fnJ0PNW_bnJ1ao8SwE?usp=sharing
and after unzipping this file ``2020-01-16-162729.zip`` move it to an arbitrary directory ``/path/to/model/2020-01-16-162729``.

You can than run the ``test.py`` script with the following arguments, along with specifying the output directory ``/path/to/output`` where you would like to save the generated (outpainted) images.

### test.py
``
python test.py --seg_model_path=/path/to/pretrained/segmentation/model 
--gen_model_path=/path/to/model/2020-01-16-162729 --img_path=/path/to/dataset --out_path=/path/to/output
``