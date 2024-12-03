import argparse
import logging
import os

import cv2
import numpy as np
import torch
from PIL import Image

from unet import UNet
from utils.data_loading import BasicDataset
from utils.utils import plot_img_and_mask

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
        output = torch.nn.functional.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()

def mask_to_image(mask: np.ndarray, mask_values):
    out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)
    for i, v in enumerate(mask_values):
        out[mask == i] = v
    return Image.fromarray(out)

def process_video(video_path, output_path, net, device, scale_factor=1, out_threshold=0.5):
    # 動画の読み込み
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 出力動画の設定
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logging.info(f'Total frames: {frame_count}')

    # フレームごとに処理
    for _ in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        # OpenCVのBGRフレームをPILのRGBに変換
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # フレームのマスクを予測
        mask = predict_img(net, pil_img, device, scale_factor, out_threshold)
        
        # マスクを画像として保存するための変換
        mask_img = mask_to_image(mask, mask_values=[0, 255])  # 白黒マスクに変換
        mask_img = mask_img.resize((width, height))  # 元の解像度に合わせてリサイズ

        # OpenCV用に再変換
        mask_frame = np.array(mask_img)
        mask_frame = cv2.cvtColor(mask_frame, cv2.COLOR_GRAY2BGR)  # 3チャンネルに変換

        # フレームとマスクを重ねる
        overlay = cv2.addWeighted(frame, 0.7, mask_frame, 0.3, 0)  # 元フレームとマスクを重ね合わせ

        # 結果を書き込み
        out.write(overlay)

    # リソースの解放
    cap.release()
    out.release()
    logging.info(f'Processed video saved to {output_path}')

if __name__ == '__main__':
    # 引数の設定
    parser = argparse.ArgumentParser(description='Process a video to predict masks for each frame')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE', help='Model file path')
    parser.add_argument('--input-video', '-i', required=True, help='Path to the input video')
    parser.add_argument('--output-video', '-o', required=True, help='Path to save the output video')
    parser.add_argument('--scale', '-s', type=float, default=1.0, help='Scale factor for input frames')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5, help='Threshold for mask prediction')
    args = parser.parse_args()

    # モデルの読み込み
    net = UNet(n_channels=3, n_classes=2, bilinear=False)  # 必要に応じて変更
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    logging.info(f'Loading model {args.model}')
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)
    logging.info('Model loaded!')

    # 動画の処理
    process_video(args.input_video, args.output_video, net, device, args.scale, args.mask_threshold)
