import cv2
import numpy as np

# 動画を読み込む（AVI形式のファイル）
video_path = '.avi'  # 入力動画のパス
cap = cv2.VideoCapture(video_path)

# 動画のフレームレートとサイズを取得
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 出力用の動画ファイルの設定（AVI形式で保存）
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # AVI形式のコーデック（XVID）
out = cv2.VideoWriter('output_video1.avi', fourcc, fps, (frame_width, frame_height))

# 明るさとコントラストの係数
brightness = 50 # 明るさ調整値 (範囲例: -100 〜 100)
contrast = 3  # コントラスト調整値 (通常は1.0がデフォルト)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 明るさとコントラストを調整
    adjusted_frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)

    # 調整されたフレームを出力動画に書き込み
    out.write(adjusted_frame)

    # 調整されたフレームをリアルタイムで表示（オプション）
    cv2.imshow('Adjusted Video', adjusted_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# リソースを解放
cap.release()
out.release()
cv2.destroyAllWindows()