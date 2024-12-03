import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask
import matplotlib.pyplot as plt

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input-folder', '-i', metavar='INPUT_FOLDER', required=True,
                        help='Folder containing input images')
    parser.add_argument('--output-folder', '-o', metavar='OUTPUT_FOLDER', required=True,
                        help='Folder to save output masks')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    
    return parser.parse_args()

def mask_to_image(mask: np.ndarray, mask_values):
    out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)
    for i, v in enumerate(mask_values):
        out[mask == i] = v
    return Image.fromarray(out)

if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # 入力フォルダー内の画像ファイルを取得
    in_files = [os.path.join(args.input_folder, f) for f in os.listdir(args.input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    out_folder = args.output_folder
    os.makedirs(out_folder, exist_ok=True)

    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    for filename in in_files:
        logging.info(f'Predicting image {filename} ...')
        img = Image.open(filename)

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        if not args.no_save:
            out_filename = os.path.join(out_folder, os.path.splitext(os.path.basename(filename))[0] + '.jpg')
            result = mask_to_image(mask, mask_values)
            result.save(out_filename, format='JPEG')
            logging.info(f'Mask saved to {out_filename}')

        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            
            # 入力画像とマスクを重ねて表示
            plt.figure(figsize=(10, 5))
            
            # 入力画像
            plt.subplot(1, 3, 1)
            plt.imshow(img)
            plt.title("Input Image")
            plt.axis("off")
            
            # 出力マスク
            plt.subplot(1, 3, 2)
            plt.imshow(mask, cmap='gray')
            plt.title("Predicted Mask")
            plt.axis("off")
            
            # 重ね合わせ
            plt.subplot(1, 3, 3)
            plt.imshow(img)
            plt.imshow(mask, cmap='jet', alpha=0.5)  # alphaで透明度を調整
            plt.title("Overlay")
            plt.axis("off")
            
            plt.tight_layout()
            plt.show()
