import cv2
import mediapipe as mp
import csv
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Etiqueta de la acción
#
# Cambiar segun la accion del video a analizar
#
# Acciones:
# 1: CaminarFrente (Toma frontal)
# 2: CaminarLado (Toma lateral)
# 3: CaminarEspalda (Toma desde atras)
# 4: Sentarse
# 5: Pararse
# 6: Girarse

action_label = ""

# Ruta del video
video_path = '../Videos/Caminando_Devuelta.mp4'
cap = cv2.VideoCapture(video_path)



with open('./DataBase/database_pose_******.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    
    header = ['label']
    for i in range(33):
        header += [f'x{i}', f'y{i}', f'z{i}']
    writer.writerow(header)

    # Procesamos cada fotograma del video
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break

        # Convertimos el fotograma de BGR a RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Procesamos el fotograma con mediapipe para detectar las articulaciones
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            # Extraemos las coordenadas x, y, z de cada articulación
            landmarks = results.pose_landmarks.landmark
            # Etiqueta del movimiento
            row = [action_label]
            
            # Agregamos las coordenadas de los 33 puntos clave obtenidos de las articulaciones

            # NOTA:
            # Valores y su significado:

            # 0 - nose
            # 1 - left eye (inner)
            # 2 - left eye
            # 3 - left eye (outer)
            # 4 - right eye (inner)
            # 5 - right eye
            # 6 - right eye (outer)
            # 7 - left ear
            # 8 - right ear
            # 9 - mouth (left)
            # 10 - mouth (right)
            # 11 - left shoulder
            # 12 - right shoulder
            # 13 - left elbow
            # 14 - right elbow
            # 15 - left wrist
            # 16 - right wrist
            # 17 - left pinky
            # 18 - right pinky
            # 19 - left index
            # 20 - right index
            # 21 - left thumb
            # 22 - right thumb
            # 23 - left hip
            # 24 - right hip
            # 25 - left knee
            # 26 - right knee
            # 27 - left ankle
            # 28 - right ankle
            # 29 - left heel
            # 30 - right heel
            # 31 - left foot index
            # 32 - right foot index

            for landmark in landmarks:
                row.append(landmark.x)
                row.append(landmark.y)
                row.append(landmark.z)
            
            writer.writerow(row)

# Liberamos recursos usados
cap.release()
cv2.destroyAllWindows()