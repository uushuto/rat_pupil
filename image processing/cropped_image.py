import os
import json
from PIL import Image

# 入力フォルダと出力フォルダを指定
input_folder = "moto"  # 画像とJSONファイルがあるフォルダ
output_folder = "rat_tri"  # トリミングされた画像の保存先フォルダ

# 出力フォルダが存在しない場合は作成
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# フォルダ内のファイルを走査
for filename in os.listdir(input_folder):
    if filename.endswith(".json"):  # JSONファイルを見つけた場合
        json_path = os.path.join(input_folder, filename)
        image_path = os.path.join(input_folder, filename.replace(".json", ".jpg"))  # 対応する画像ファイル
        output_path = os.path.join(output_folder, filename.replace(".json", "_cropped.jpg"))  # 出力画像ファイル

        # JSONファイルの読み込み
        with open(json_path) as f:
            data = json.load(f)

        # 画像ファイルが存在する場合のみ処理
        if os.path.exists(image_path):
            # 画像の読み込み
            image = Image.open(image_path)
            original_width, original_height = image.size

            # 矩形の座標を取得 (最初の矩形アノテーションを使用)
            rect_points = data['shapes'][0]['points']
            (x1, y1), (x2, y2) = rect_points

            # 小数の場合があるため、座標を整数に変換
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # 矩形で囲んだ部分をトリミング
            cropped_image = image.crop((x1, y1, x2, y2))

            # 元のサイズに合わせて中央に配置するための背景画像（黒）を作成
            new_image = Image.new("RGB", (original_width, original_height), (0, 0, 0))
            new_image.paste(cropped_image, ((original_width - cropped_image.width) // 2, 
                                            (original_height - cropped_image.height) // 2))

            # トリミングして元サイズで保存
            new_image.save(output_path)
            print(f"Trimming completed for {image_path} and saved to {output_path}")

print("All images processed.")
