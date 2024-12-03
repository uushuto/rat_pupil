import cv2
import os

"""
動画からxフレームごとに画像を保存するコード
"""

# setting -------------------------------------------------------------------

# 動画ファイルのパス
video_path = 'rat0825_1.avi'  

# 保存するフォルダ名
save_dir = 'moto1' 

# 保存する番号の最初の値 (例：save_number = 10, xxxx0010)
save_number = 0

# 何フレームごとに画像を保存するか
save_frame = 200

# ----------------------------------------------------------------------------

exists = os.path.exists(video_path)
if not exists:
    print("エラー : video_pathが存在しません.")
    exit()

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 動画を読み込む
cap = cv2.VideoCapture(video_path)

# フレーム番号の初期化
frame_number = 0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 総フレーム数を取得

# フレームを処理して画像を保存
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 指定されたフレームごとに画像を保存
    if frame_number % save_frame == 0:
        save_filename = os.path.join(save_dir, f'rad{save_number:05d}.jpg')
        
        # 画像を保存
        cv2.imwrite(save_filename, frame)
        save_number += 1

    frame_number += 1

# リソースの解放
cap.release()

print("動画から画像を抽出し、保存が完了しました。")
