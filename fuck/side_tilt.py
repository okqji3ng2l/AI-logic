"""
側向人體傾斜 & 駝背偵測
相容 MediaPipe 0.10+ (Tasks API)
啟動: python side_tilt.py
按 Q 離開
"""
import cv2
import math
import os
import urllib.request

# ── 自動下載模型 ────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "pose_landmarker_lite.task")
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/"
    "pose_landmarker_lite.task"
)

def download_model(url, path):
    print("正在下載 Pose 模型（僅首次需要）...")
    tmp = path + ".tmp"
    try:
        urllib.request.urlretrieve(url, tmp)
        # 驗證檔案大小 > 1 MB
        if os.path.getsize(tmp) < 1_000_000:
            raise RuntimeError("下載檔案過小，可能不完整")
        os.replace(tmp, path)
        print(f"模型下載完成：{path}")
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(f"模型下載失敗：{e}")

if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1_000_000:
    download_model(MODEL_URL, MODEL_PATH)

# ── 初始化 MediaPipe Tasks API ──────────────────────────
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

BaseOptions   = mp_python.BaseOptions
PoseLandmarker       = mp_vision.PoseLandmarker
PoseLandmarkerOptions = mp_vision.PoseLandmarkerOptions
VisionRunningMode     = mp_vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    min_pose_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
landmarker = PoseLandmarker.create_from_options(options)

# ── Pose landmark 索引 ──────────────────────────────────
IDX = {
    "l_ear": 7,  "r_ear": 8,
    "l_shoulder": 11, "r_shoulder": 12,
    "l_hip": 23,      "r_hip": 24,
    "l_knee": 25,     "r_knee": 26,
    "l_ankle": 27,    "r_ankle": 28,
}
# 骨架連線（用於繪製灰色背景骨架）
CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),  # 手臂
    (11,23),(12,24),(23,24),                   # 軀幹
    (23,25),(25,27),(24,26),(26,28),            # 腿
    (7,11),(8,12),                              # 耳→肩
]

# ── 工具函數 ────────────────────────────────────────────
def to_px(lm, h, w):
    return int(lm.x * w), int(lm.y * h)

def get_side(lm_list, h, w):
    """選可見度較高的側邊，回傳 (ear, shoulder, hip, knee, ankle), side_str"""
    l = [lm_list[IDX[k]] for k in ("l_ear","l_shoulder","l_hip","l_knee","l_ankle")]
    r = [lm_list[IDX[k]] for k in ("r_ear","r_shoulder","r_hip","r_knee","r_ankle")]
    l_vis = l[1].visibility + l[2].visibility
    r_vis = r[1].visibility + r[2].visibility
    chosen = l if l_vis >= r_vis else r
    side   = "LEFT" if l_vis >= r_vis else "RIGHT"
    return [to_px(p, h, w) for p in chosen], side

def angle_from_vertical(top, bot):
    """兩點連線與垂直線夾角（正=向前傾）"""
    dx = top[0] - bot[0]
    dy = bot[1] - top[1]
    return math.degrees(math.atan2(dx, dy))

def three_point_angle(a, b, c):
    """b 點的夾角（度）"""
    ax, ay = a[0]-b[0], a[1]-b[1]
    cx, cy = c[0]-b[0], c[1]-b[1]
    dot = ax*cx + ay*cy
    mag = math.hypot(ax,ay) * math.hypot(cx,cy)
    if mag == 0: return 180.0
    return math.degrees(math.acos(max(-1.0, min(1.0, dot/mag))))

def tilt_label(angle):
    if angle >  10: return f"FORWARD  {angle:+.1f}deg",  (0, 100, 255)
    if angle < -10: return f"BACKWARD {angle:+.1f}deg",  (255, 100, 0)
    return              f"UPRIGHT  {angle:+.1f}deg",  (0, 220, 80)

def hunch_label(spine_ang, head_ratio):
    if spine_ang < 140 or head_ratio > 0.12:
        return 2, f"HUNCHED      spine={spine_ang:.0f}deg", (0, 60, 255)
    if spine_ang < 155 or head_ratio > 0.05:
        return 1, f"SLIGHT HUNCH spine={spine_ang:.0f}deg", (0, 180, 255)
    return     0, f"GOOD POSTURE spine={spine_ang:.0f}deg", (0, 220, 80)

# ── 主迴圈 ──────────────────────────────────────────────
cap = cv2.VideoCapture(0)
print("偵測中... 按 Q 離開")
timestamp_ms = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    h, w = frame.shape[:2]

    # MediaPipe 偵測
    import mediapipe as mp
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    timestamp_ms += 33

    output = frame.copy()

    if result.pose_landmarks:
        lm = result.pose_landmarks[0]   # 第一個人

        # ── 繪製背景骨架 ──────────────────────────────
        pts_all = {i: to_px(lm[i], h, w) for i in range(33)}
        for a_idx, b_idx in CONNECTIONS:
            cv2.line(output, pts_all[a_idx], pts_all[b_idx], (80,80,80), 1)
        for pt in pts_all.values():
            cv2.circle(output, pt, 3, (100,100,100), -1)

        # ── 取側向關鍵點 ──────────────────────────────
        pts, side = get_side(lm, h, w)
        ear, shoulder, hip, knee, ankle = pts

        # ── 計算指標 ──────────────────────────────────
        trunk_angle = angle_from_vertical(shoulder, hip)
        body_angle  = angle_from_vertical(shoulder, ankle)
        spine_angle = three_point_angle(ear, shoulder, hip)
        head_ratio  = (ear[0] - shoulder[0]) / w   # 正 = 頭在肩前

        tilt_text,  tilt_color  = tilt_label(trunk_angle)
        h_level, hunch_text, hunch_color = hunch_label(spine_angle, head_ratio)

        # ── 繪製側向身體線 ────────────────────────────
        neck_color = hunch_color if h_level > 0 else (255, 255, 0)
        cv2.line(output, ear,      shoulder, neck_color,   3)
        cv2.line(output, shoulder, hip,      tilt_color,   3)
        cv2.line(output, hip,      knee,     (0, 200, 255), 2)
        cv2.line(output, knee,     ankle,    (0, 200, 255), 2)

        # 傾斜弧（髖部）
        cv2.ellipse(output, hip, (55,55), 0,
                    min(-90, -90+int(trunk_angle)), max(-90, -90+int(trunk_angle)),
                    (0, 220, 255), 2)
        # 脊椎弧（肩膀）
        dev = int(180 - spine_angle)
        cv2.ellipse(output, shoulder, (45,45), 0, -180, -180+dev, (0,200,255), 2)

        # 關鍵點
        for pt, col in zip([ear, shoulder, hip, knee, ankle],
                           [hunch_color, hunch_color, tilt_color, tilt_color, tilt_color]):
            cv2.circle(output, pt, 7, col, -1)
            cv2.circle(output, pt, 7, (255,255,255), 1)

        # ── HUD ───────────────────────────────────────
        overlay = output.copy()
        cv2.rectangle(overlay, (0,0), (370,170), (20,20,20), -1)
        cv2.addWeighted(overlay, 0.65, output, 0.35, 0, output)

        cv2.putText(output, "SIDE TILT & POSTURE",
                    (12,26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200,200,200), 1)
        cv2.putText(output, tilt_text,
                    (12,58), cv2.FONT_HERSHEY_SIMPLEX, 0.78, tilt_color, 2)
        cv2.putText(output, hunch_text,
                    (12,93), cv2.FONT_HERSHEY_SIMPLEX, 0.78, hunch_color, 2)
        cv2.putText(output, f"Trunk:{trunk_angle:+.1f}  Body:{body_angle:+.1f}  [{side}]",
                    (12,123), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180,220,255), 1)
        cv2.putText(output, f"Head offset:{int(head_ratio*w):+d}px  Spine:{spine_angle:.0f}deg",
                    (12,148), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180,180,180), 1)

    else:
        cv2.putText(output, "No person — face sideways",
                    (20,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (80,80,255), 2)

    cv2.imshow("Side Tilt & Posture  [Q] quit", output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
landmarker.close()
cv2.destroyAllWindows()
