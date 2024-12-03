#瞳孔楕円近似
import json
import numpy as np
import cv2
import os
import csv

# 入力フォルダーと出力フォルダーの指定
input_folder = ""  # JSONファイルが保存されているフォルダー
output_folder = ""  # 楕円描画結果を保存するフォルダー
csv_output_path = ""  # CSV出力先

# 出力フォルダーが存在しない場合は作成
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# CSVファイルを作成してヘッダーを書き込む
with open(csv_output_path, mode="w", newline="") as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["ファイル名", "中心_x", "中心_y", "長軸", "短軸", "回転角度", "推定瞳孔直径"])

    # フォルダー内のすべてのJSONファイルに対して処理
    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):
            json_path = os.path.join(input_folder, filename)

            # JSONファイルの読み込み
            with open(json_path) as f:
                data = json.load(f)

            # JSONファイルから画像サイズを取得
            image_width = int(data["imageWidth"])
            image_height = int(data["imageHeight"])

            # 瞳孔を囲むポリゴンの座標を取得（例: 最初のshapeを使用）
            polygon_points = np.array(data['shapes'][0]['points'], dtype=np.float32)

            # 楕円にフィットさせる (OpenCVのfitEllipseを使用)
            ellipse = cv2.fitEllipse(polygon_points)

            # 楕円のパラメータを取得
            (center, (major_axis, minor_axis), angle) = ellipse

            if major_axis < minor_axis:
                print(f"警告: ファイル {filename} で長軸 ({major_axis}) が短軸 ({minor_axis}) よりも小さいです。値を入れ替えます。")
                major_axis, minor_axis = minor_axis, major_axis

            # 楕円のパラメータを表示
            print(f"ファイル: {filename}")
            print(f"中心: {center}")
            print(f"長軸 (major axis): {major_axis}")
            print(f"短軸 (minor axis): {minor_axis}")
            print(f"回転角度: {angle}")
            print(f"推定される瞳孔の直径: {major_axis}\n")

            # 結果をCSVに書き込む
            csv_writer.writerow([filename, center[0], center[1], major_axis, minor_axis, angle, major_axis])

            # 楕円を描画する画像の作成（元の画像サイズで空の画像を作成）
            image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
            polygon_points_int = polygon_points.astype(np.int32)

            # ポリゴンを描画
            cv2.polylines(image, [polygon_points_int], isClosed=True, color=(0, 255, 0), thickness=2)

            # 楕円を描画
            cv2.ellipse(image, ellipse, (255, 0, 0), 2)

            # 描画結果をJPG形式で保存
            output_image_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_ellipse.jpg")
            cv2.imwrite(output_image_path, image)

print("すべてのファイルの処理が完了し、結果がCSVに保存されました。")