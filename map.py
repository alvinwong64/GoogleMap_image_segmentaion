from unet import UNet
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset
import os
import torch
import torchvision
from PIL import Image
import numpy as np
import os
import imgviz
import matplotlib.pyplot as plt
from numpy import asarray

if __name__ == "__main__":
    src = r"C:/Users/GuanSheng.Wong/Desktop/Unet/predicted/test_image/"
    outfolder = r"C:/Users/GuanSheng.Wong/Desktop/Unet/predicted/masked_image/"
    in_path = "./data/imgs/100.jpg"
    for filename in os.listdir(src):
        with torch.no_grad():
            net = UNet(n_channels=3, n_classes=5)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            net.to(device=device)
            net.load_state_dict(torch.load("epoch70.pth", map_location=device))
            net.eval()
            img = Image.open(src + filename)
            trans = torchvision.transforms.PILToTensor()
            mask_pred = net(torch.div(trans(img).float().unsqueeze(dim=0).to(device), 255))
            mask_pred = torch.nn.functional.softmax(mask_pred, dim=1)
            mask_pred = mask_pred.squeeze(dim=0)
            # print(mask_pred)

            # mapping labels to colour
            colormap = {
                0: (0, 0, 0),
                1: (128, 128, 0),
                2: (128, 0, 0),
                3: (0, 128, 0),
                4: (0, 0, 128)
            }
            mask_pred = mask_pred.cpu()
            mask = torch.zeros([512, 512, 3])
            ids = np.argmax(mask_pred, axis=0)
            colormap_arr = np.array([[0, 0, 0], [128, 128, 0], [128, 0, 0], [0, 128, 0], [0, 0, 128]])

            # Mask prediction
            for i, index in enumerate(ids):
                for j, index2 in enumerate(index):
                    mask[i, j, :] = torch.tensor(colormap[index2.item()])

            trans2 = torchvision.transforms.ToPILImage()
            img_mask = trans2(np.uint8(mask))

            # Visualizing segemntaion
            predictions = (ids).numpy()
            input = imgviz.color.rgb2gray(asarray(img))
            label_names = ['0:background', '1:road', '2:building', '3:tree', '4:sea']
            labelviz_withimg = imgviz.label2rgb(predictions, colormap=colormap_arr, img=input, label_names=label_names,
                                                font_size=25, loc='rb')
            masked_image = Image.fromarray(labelviz_withimg, 'RGB')
            # masked_image.show()

            # Concatenating images
            cat1 = Image.new('RGB', (img.width + masked_image.width, img.height))
            cat1.paste(img, (0, 0))
            cat1.paste(masked_image, (img.width, 0))

            cat2 = Image.new('RGB', (cat1.width + img_mask.width, img_mask.height))
            cat2.paste(cat1, (0, 0))
            cat2.paste(img_mask, (cat1.width, 0))

            # Saving
            cat2.save(outfolder + filename)
            print(filename + "done")
