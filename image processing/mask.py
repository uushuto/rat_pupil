import json
import numpy as np
from PIL import Image
import os
import cv2

def json_to_mask(json_file, output_path):
    with open(json_file) as f:
        data = json.load(f)

    # マスクサイズを取得
    img_height = data['imageHeight']
    img_width = data['imageWidth']

    # バイナリマスクの作成（背景を0、オブジェクトを1として設定）
    mask = np.zeros((img_height, img_width), dtype=np.uint8)

    for shape in data['shapes']:
        points = shape['points']
        polygon = np.array(points, dtype=np.int32)
        mask = cv2.fillPoly(mask, [polygon], 1)

    # マスクを保存
    mask_img = Image.fromarray(mask * 255)  # 255倍して白黒画像に
    mask_img.save(output_path)

# 使用例
input_folder = ''  # JSONファイルがあるフォルダ
output_folder = ''  # マスク画像を保存するフォルダ

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for json_file in os.listdir(input_folder):
    if json_file.endswith('.json'):
        json_path = os.path.join(input_folder, json_file)
        mask_output_path = os.path.join(output_folder, json_file.replace('.json', '.png'))
        json_to_mask(json_path, mask_output_path)
