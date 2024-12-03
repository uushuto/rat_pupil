import cv2
import numpy as np
import os

# フォルダパス
input_folder = 'value'  # 元画像が含まれるフォルダ
mask_folder = 'valu'    # 予測マスク画像が含まれるフォルダ
output_folder = 'out'  # 出力画像を保存するフォルダ

# 出力フォルダが存在しない場合は作成
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# フォルダ内のすべての画像を処理
for filename in os.listdir(input_folder):
    # 画像のパスと対応するマスク画像のパス
    image_path = os.path.join(input_folder, filename)
    mask_path = os.path.join(mask_folder, filename)

    # 元画像とマスク画像の読み込み
    image = cv2.imread(image_path)
    if image is None:
        print(f"元画像が見つかりません: {image_path}")
        continue
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"マスク画像が見つかりません: {mask_path}")
        continue

    # マスク画像を元画像と同じサイズにリサイズ
    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # マスクの閾値処理（0か1のバイナリマスクを作成）
    _, binary_mask = cv2.threshold(mask_resized, 127, 255, cv2.THRESH_BINARY)

    # 重ね合わせ用にカラーに変換
    color_mask = cv2.applyColorMap(binary_mask, cv2.COLORMAP_JET)  # カラーマップの変更可

    # マスクを重ねる（0.5は透明度の調整）
    overlay = cv2.addWeighted(image, 0.7, color_mask, 0.3, 0)  # 0.7: 元画像の透明度, 0.3: マスクの透明度

    # 重ね合わせ画像の保存
    output_path = os.path.join(output_folder, f"overlay_{filename}")
    cv2.imwrite(output_path, overlay)
    print(f"保存しました: {output_path}")

print("すべての画像が処理されました。")
