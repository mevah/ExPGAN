import os
import torch
import argparse
import numpy as np
import scipy.misc as misc
import glob
import torchvision.models as models
from fcn_model import fcn8s
import cityscapes_seg_loader as sloader
from gan_model import ExPGenerator
import torch.nn.functional as F
def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 

#set up image and label paths
    print("Read Input Images from : {}".format(args.img_path))
    img_path= args.img_path + "leftImg8bit/ourtest"
    label_path= args.img_path + "gtFine/ourtest"
    img_path_list= os.listdir(img_path)
    n_classes= 7
    loader= sloader.cityscapesLoader(root=args.img_path, is_transform=True, img_norm=True, test_mode=True)
    #iterate over all images in test folder
    for city in img_path_list:
        file_list= os.listdir(os.path.join(img_path,city))
        for file in file_list:
            img = misc.imread(os.path.join(img_path,city,file))
            file_root= file.split("leftImg8bit.png")[0]
            label_name = file_root + "gtFine_labelIds.png"
            filename= os.path.join(label_path, city, label_name)
            label= misc.imread(filename)

    #prepare image for inference
            orig_size = img.shape[:-1]
            img_full = misc.imresize(img, (256, 512))
            img= img_full[:,128:384, :]
            img = img[:, :, ::-1]
            img = img.astype(np.float64)
            #img -= loader.mean
            img = img.astype(float) / 255.0
            # NHWC -> NCHW
            img = img.transpose(2, 0, 1)
            img = np.expand_dims(img, 0)
            img = torch.from_numpy(img).float()
    # Setup Segmentation Model
            print("Loading segmentation model from...", args.seg_model_path)
            model_dict = {"arch": "fcn8s"}            
            model = fcn8s(n_classes=n_classes)
            vgg16 = models.vgg16(pretrained=True)
            model.init_vgg16_params(vgg16)     
            state = (torch.load(args.seg_model_path)["model_state"])
            model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
            model.load_state_dict(state)
            model.eval()
            model.to(device)
            print("Segmentation model loaded.")
            images = img.to(device)
    #make predictions
            outputs = model(images)
            pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)           
            decoded = loader.decode_segmap(pred) #returns rgb version of the label numbers
            print("Classes found: ", np.unique(pred))
            if not os.path.exists(args.out_path + "/segoutputs"):
                os.mkdir(os.path.join(os.getcwd(),args.out_path) + "/segoutputs")
            misc.imsave(args.out_path + "/segoutputs/" + file_root + "_pred.png", decoded)
            print("Segmentation mask saved at: {}".format(args.out_path + "/segoutputs"))
    #downsample and crop images and segmentations for outpainting
            label_resized=misc.imresize(label, (256,512))                
            img_crop= img
            label_crop=label_resized[:,128:384]
         #   print(pred.shape    )
         #   seg_resized= (pred[ ::4, ::4])
            seg_crop= pred

    #Load generator to generate images
            generator_G= ExPGenerator()
            state = (torch.load(args.gen_model_path))
            generator_G.load_state_dict(state.get('generator'))
            generator_G.eval()
            generator_G.to(device)

    #perform generation
            img_resized = torch.from_numpy(img_full).cuda()
            #seg_resized = torch.from_numpy(seg_resized).cuda()
            img_crop = img_crop.cuda()
            seg_crop = torch.from_numpy(seg_crop).cuda()
            seg_crop= torch.nn.functional.one_hot(seg_crop).permute(2,0,1).float()
            seg_crop= seg_crop.unsqueeze(0)

            orj_left = img_resized[:,:, 0:128]
            orj_right = img_resized[:,:, -128:]
        
            gen_fake_left, gen_fake_right, gen_fake_seg = generator_G(img_crop, seg_crop)
            generated= torch.cat((gen_fake_left, img_crop, gen_fake_right), dim=3)
            gen_im= generated.detach().cpu().numpy()
            gen_im= np.squeeze(gen_im, axis=0)
            gen_im= gen_im.transpose(1,2,0)
    #save generated images 
            if not os.path.exists(args.out_path + "/genoutputs"):
                os.mkdir(os.path.join(os.getcwd(),args.out_path) + "/genoutputs")
            misc.imsave(args.out_path + "/genoutputs/" + file_root + "_gen.png", gen_im)
            print("Outpainted image saved at: {}".format(args.out_path + "/genoutputs"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Params")
    parser.add_argument(
        "--seg_model_path",
        nargs="?",
        type=str,
        default="runs/.../.../fcn8s_cityscapes_best_model.pkl",
        help="Path to the saved seg model",
    )
    parser.add_argument(
        "--gen_model_path",
        nargs="?",
        type=str,
        default="runs/...",
        help="Path to the saved seg model",
    )

    parser.add_argument(
        "--img_path", nargs="?", type=str, default=None, help="Path of the input data"
    )
    parser.add_argument(
        "--out_path", nargs="?", type=str, default=None, help="Path of the output images and segmentation maps"
    )
    args = parser.parse_args()
    test(args)
