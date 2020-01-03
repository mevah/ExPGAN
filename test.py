import os
import torch
import argparse
import numpy as np
import scipy.misc as misc
import glob
import torchvision.models as models
from fcn_model import fcn8s
import cityscapes_seg_loader as sloader
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
            img = misc.imresize(img, (512, 1024))
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
            
            decoded = loader.decode_segmap(pred)

            print("Classes found: ", np.unique(pred))
            if not os.path.exists(args.out_path + "/segoutputs"):
                os.mkdir(os.path.join(os.getcwd(),args.out_path) + "/segoutputs")
            misc.imsave(args.out_path + "/segoutputs/" + file_root + "_pred.png", decoded)
            print("Segmentation Mask Saved at: {}".format(args.out_path + "/segoutputs"))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Params")
    parser.add_argument(
        "--seg_model_path",
        nargs="?",
        type=str,
        default="fcn8s_cityscapes_best_model.pkl",
        help="Path to the saved seg model",
    )
    parser.add_argument(
        "--gen_model_path",
        nargs="?",
        type=str,
        default="fcn8s_pascal_1_26.pkl",
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
