import cv2
import mediapipe as mp
import numpy as np
import time
import csv
from datetime import datetime



mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class AttentionTracker:
    def __init__(self):
        
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=50,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )
        self.face_detector = mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.7
        )
        
       
        self.LEFT_EYE = [33, 133]          # Outer and inner corners
        self.RIGHT_EYE = [362, 263]
        self.LEFT_IRIS = [468]             # Center of left iris
        self.RIGHT_IRIS = [473]            # Center of right iris
        
      
        self.EXPRESSION_LANDMARKS = {
            'left_mouth': 61,
            'right_mouth': 291,
            'top_lip': 13,
            'bottom_lip': 14,
            'left_eyebrow_inner': 55,
            'right_eyebrow_inner': 285,
            'left_eyebrow_outer': 65,
            'right_eyebrow_outer': 295,
            'nose_tip': 1,
            'chin': 152
        }
        
      
        self.HEAD_POSE_LANDMARKS = [1, 152, 33, 263, 61, 291]
        self.MODEL_POINTS = np.array([
            (0.0, 0.0, 0.0),          # Nose tip
            (0.0, -330.0, -65.0),     # Chin
            (-225.0, 170.0, -135.0),  # Left eye left corner
            (225.0, 170.0, -135.0),   # Right eye right corner
            (-150.0, -150.0, -125.0), # Left mouth corner
            (150.0, -150.0, -125.0)   # Right mouth corner
        ], dtype=np.float64)
        
        
        self.attention_data = {}
        self.last_attention_time = time.time()
        self.attention_threshold = 3  
        self.attention_warning = False
        
        
        self.SMILE_THRESH = 0.30
        self.SAD_MOUTH_SLOPE_THRESH = 0.015
        self.SAD_EYEBROW_HEIGHT = 0.05
        self.ANGRY_EYEBROW_DISTANCE = 0.06
        self.ANGRY_EYEBROW_HEIGHT = 0.035
        
        self.COLORS = {
            "attentive": (0, 255, 0),        # Green
            "looking_away": (0, 165, 255),   # Orange
            "distracted": (0, 0, 255),       
            "engaged": (0, 255, 255),       
            "confused": (255, 0, 0),        
            "bored": (128, 0, 128)          
        }
    
    def distance(self, p1, p2):
        
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    
    def get_gaze_ratio(self, eye_corner1, eye_corner2, iris_center):
        
        eye_width = self.distance(eye_corner1, eye_corner2)
        iris_dist = self.distance(eye_corner1, iris_center)
        ratio = iris_dist / eye_width if eye_width != 0 else 0
        return ratio
    
    def analyze_gaze(self, landmarks, face_id):
        
       
        left_eye_outer = landmarks[self.LEFT_EYE[0]]
        left_eye_inner = landmarks[self.LEFT_EYE[1]]
        right_eye_inner = landmarks[self.RIGHT_EYE[1]]
        right_eye_outer = landmarks[self.RIGHT_EYE[0]]
        left_iris = landmarks[self.LEFT_IRIS[0]]
        right_iris = landmarks[self.RIGHT_IRIS[0]]
        
        
        left_ratio = self.get_gaze_ratio(left_eye_outer, left_eye_inner, left_iris)
        right_ratio = self.get_gaze_ratio(right_eye_inner, right_eye_outer, right_iris)
        
        
        if left_ratio < 0.35 or right_ratio < 0.35:
            gaze_status = "Looking Left"
            attention = "distracted"
        elif left_ratio > 0.65 or right_ratio > 0.65:
            gaze_status = "Looking Right"
            attention = "distracted"
        elif abs(left_ratio - 0.5) > 0.15 or abs(right_ratio - 0.5) > 0.15:
            gaze_status = "Looking Away"
            attention = "looking_away"
        else:
            gaze_status = "Attentive"
            attention = "attentive"
        
        
        current_time = time.time()
        if face_id not in self.attention_data:
            self.attention_data[face_id] = {
                "attention_start": current_time,
                "last_attentive": current_time
            }
        
        if attention == "attentive":
            self.attention_data[face_id]["last_attentive"] = current_time
            self.attention_warning = False
        else:
           
            if current_time - self.attention_data[face_id]["last_attentive"] > self.attention_threshold:
                attention = "distracted"
                self.attention_warning = True
        
        return gaze_status, attention
    
    def analyze_expression(self, landmarks):
        """Determine facial expression"""
       
        points = {}
        for name, idx in self.EXPRESSION_LANDMARKS.items():
            points[name] = landmarks[idx]
        
        
        face_height = self.distance(points['nose_tip'], points['chin'])
        
       
        mouth_width = self.distance(points['left_mouth'], points['right_mouth'])
        mouth_open = self.distance(points['top_lip'], points['bottom_lip'])
        mouth_ratio = mouth_open / mouth_width
        
        
        eyebrow_distance = self.distance(points['left_eyebrow_inner'], points['right_eyebrow_inner'])
        eyebrow_height_l = (points['nose_tip'].y - points['left_eyebrow_inner'].y) / face_height
        eyebrow_height_r = (points['nose_tip'].y - points['right_eyebrow_inner'].y) / face_height
        eyebrow_avg_height = (eyebrow_height_l + eyebrow_height_r) / 2
        
       
        mouth_corner_diff_y = points['left_mouth'].y - points['right_mouth'].y
        
        
        if mouth_ratio > self.SMILE_THRESH:
            expression = "Smiling"
            engagement = "engaged"
        elif eyebrow_avg_height > self.SAD_EYEBROW_HEIGHT and abs(mouth_corner_diff_y) > self.SAD_MOUTH_SLOPE_THRESH:
            expression = "Sad"
            engagement = "confused"
        elif eyebrow_distance < self.ANGRY_EYEBROW_DISTANCE and eyebrow_avg_height < self.ANGRY_EYEBROW_HEIGHT:
            expression = "Angry"
            engagement = "distracted"
        else:
            expression = "Neutral"
            engagement = "bored"
        
        return expression, engagement
    
    def analyze_head_pose(self, landmarks, frame, camera_matrix, dist_coeffs):
        h, w = frame.shape[:2]
        
        
        image_points = np.array([
            [landmarks[i].x * w, landmarks[i].y * h] for i in self.HEAD_POSE_LANDMARKS
        ], dtype=np.float64)
        
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.MODEL_POINTS, image_points, camera_matrix, dist_coeffs)
        
        
        nose_end_point3D = np.array([[0, 0, 1000.0]])
        nose_end_point2D, _ = cv2.projectPoints(
            nose_end_point3D, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        
        
        p1 = (int(image_points[0][0]), int(image_points[0][1]))  # Nose tip
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
        cv2.line(frame, p1, p2, (0, 255, 0), 2)
        
        
        x_coords = [int(pt[0]) for pt in image_points]
        y_coords = [int(pt[1]) for pt in image_points]
        x_min, x_max = min(x_coords) - 20, max(x_coords) + 20
        y_min, y_max = min(y_coords) - 20, max(y_coords) + 20
        return (x_min, y_min, x_max, y_max)
    
    def track_attention(self):
        
        csv_file = open("attention_log.csv", mode='w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Timestamp", "Student_ID", "Gaze_Status", "Attention", 
                             "Expression", "Engagement", "Final_Status"])
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
       
        total_students = 0
        attentive_students = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
      
            focal_length = w
            center = (w/2, h/2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float64)
            dist_coeffs = np.zeros((4, 1))
            
           
            total_students = 0
            attentive_students = 0
            
           
            results = self.face_mesh.process(rgb)
            if results.multi_face_landmarks:
                for face_id, face_landmarks in enumerate(results.multi_face_landmarks):
                    total_students += 1
                    landmarks = face_landmarks.landmark
                    
                    
                    gaze_status, attention = self.analyze_gaze(landmarks, face_id)
                    
                    
                    expression, engagement = self.analyze_expression(landmarks)
                    
                   
                    if attention == "attentive" and engagement == "engaged":
                        overall_status = "Focused"
                        status_color = self.COLORS["attentive"]
                    elif attention == "distracted":
                        overall_status = "Distracted"
                        status_color = self.COLORS["distracted"]
                    elif engagement == "confused":
                        overall_status = "Confused"
                        status_color = self.COLORS["confused"]
                    else:
                        overall_status = "Neutral"
                        status_color = (200, 200, 200)
                    
                    if attention == "attentive":
                        attentive_students += 1
                    
                    
                    bbox = self.analyze_head_pose(landmarks, frame, camera_matrix, dist_coeffs)
                    
                   
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), status_color, 2)
                    
                    
                    status_text = f"Student {face_id+1}: {overall_status}"
                    cv2.putText(frame, status_text, (bbox[0], bbox[1]-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                    
                  
                    for idx in [self.LEFT_IRIS[0], self.RIGHT_IRIS[0]]:
                        cx, cy = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
                        cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)
                    
                   
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    csv_writer.writerow([
                        timestamp,
                        face_id + 1,
                        gaze_status,
                        attention,
                        expression,
                        engagement,
                        overall_status
                    ])
            
            
            if total_students > 0:
                attention_percent = (attentive_students / total_students) * 100
                attention_text = f"Class Attention: {attention_percent:.1f}%"
                cv2.putText(frame, attention_text, (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                
                if self.attention_warning:
                    warning_text = "Warning: Student distraction detected!"
                    cv2.putText(frame, warning_text, (w//2-250, 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
           
            cv2.imshow("Student Attention Tracker", frame)
            
            
            if cv2.waitKey(5) & 0xFF == 27:
                break
        
        
        cap.release()
        csv_file.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    tracker = AttentionTracker()
    tracker.track_attention()
