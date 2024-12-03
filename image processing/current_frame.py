import cv2

# 動画のパスを指定
video_path = '.avi'
output_path = ''

# 動画を読み込む
cap = cv2.VideoCapture(video_path)

# 動画のフレームレートとフレーム数を取得
fps = int(cap.get(cv2.CAP_PROP_FPS))  # フレームレート（FPS）
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 総フレーム数

# 動画の幅と高さを取得
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 出力する動画の設定
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 動画のコーデック
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# トリミングしたい範囲のフレーム番号を指定
start_time = 7  # 開始時間（秒）
end_time = 54  # 終了時間（秒）

start_frame = start_time * fps  # 開始フレーム
end_frame = end_time * fps  # 終了フレーム

# フレームを開始フレームに移動
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

# 動画のトリミングを実行
current_frame = start_frame
while current_frame < end_frame:
    ret, frame = cap.read()
    if not ret:
        break

    # フレームを書き出す
    out.write(frame)
    current_frame += 1

# リソースの解放
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"動画が {start_time}秒 から {end_time}秒 の間でトリミングされました。")
