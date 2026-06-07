import cv2
import mediapipe as mp
import math

# 初始化 MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    # 轉換顏色空間
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        # 取得關鍵點座標 (以右側為例: Right Shoulder = 12, Right Hip = 24)
        landmarks = results.pose_landmarks.landmark
        
        # 取得肩膀與髖部座標 (影像正規化座標需乘以寬高)
        h, w, _ = frame.shape
        shoulder = (int(landmarks[12].x * w), int(landmarks[12].y * h))
        hip = (int(landmarks[24].x * w), int(landmarks[24].y * h))

        # 計算角度
        # dx, dy 是相對於垂直線的位移
        angle = math.degrees(math.atan2(shoulder[0] - hip[0], hip[1] - shoulder[1]))
        
        # 繪製視覺化
        cv2.line(image, shoulder, hip, (0, 255, 0), 3)
        cv2.putText(image, f"Angle: {int(angle)} deg", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # 繪製骨架
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow('Side Lean Detection', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()