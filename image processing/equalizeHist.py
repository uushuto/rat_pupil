import cv2
import os

# 入力フォルダと出力フォルダを指定
input_folder = 'rat_tri'  # 画像があるフォルダ
output_folder = 'rat01'  # 処理後の画像を保存するフォルダ

# 出力フォルダが存在しない場合は作成
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# フォルダ内のファイルを走査
for filename in os.listdir(input_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):  # 対応する画像フォーマット
        # 画像の読み込み
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"{filename} は画像ではありません。スキップします。")
            continue

        # カラー画像の場合、グレースケールに変換
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # ヒストグラム均等化を適用
        equalized_image = cv2.equalizeHist(gray_image)

        # グレースケールの画像をBGRに変換して保存（カラーチャンネルに変換）
        bgr_equalized_image = cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2BGR)

        # 出力ファイルパスの設定
        output_image_path = os.path.join(output_folder, filename)

        # 処理後の画像を保存
        cv2.imwrite(output_image_path, bgr_equalized_image)

        print(f"{filename} にヒストグラム均等化を適用して保存しました。")

print("フォルダ内のすべての画像にヒストグラム均等化を適用しました。")

