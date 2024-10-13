import cv2
import mediapipe as mp
import csv
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Etiqueta de la acción
action_label = "Sentandose_04"

# Ruta del video
video_path = 'Entrega-1/Videos/Sentado-04.mp4'
cap = cv2.VideoCapture(video_path)

with open('Entrega-1/VideosDataSet/database_pose_Sentandose04.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    
    header = ['label']
    for i in range(33):
        header += [f'x{i}', f'y{i}', f'z{i}']
    writer.writerow(header)

    # Variables para agrupar cada 5 fotogramas
    landmarks_buffer = []
    buffer_size = 5

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
            frame_landmarks = []

            # Agregamos las coordenadas de los 33 puntos clave obtenidos de las articulaciones
            for landmark in landmarks:
                frame_landmarks.append([landmark.x, landmark.y, landmark.z])
            
            # Añadimos los puntos clave del fotograma al buffer
            landmarks_buffer.append(frame_landmarks)

            # Si tenemos suficientes fotogramas en el buffer
            if len(landmarks_buffer) == buffer_size:
                # Calculamos el promedio de las coordenadas de cada punto clave a lo largo de los últimos 5 fotogramas
                averaged_landmarks = np.mean(landmarks_buffer, axis=0)
                
                # Creamos la fila para escribir en el CSV
                row = [action_label]
                for avg_landmark in averaged_landmarks:
                    row.extend(avg_landmark)
                
                writer.writerow(row)

                # Limpiamos el buffer
                landmarks_buffer = []

    # Si quedan fotogramas en el buffer al final, promediarlos y escribirlos
    if len(landmarks_buffer) > 0:
        averaged_landmarks = np.mean(landmarks_buffer, axis=0)
        row = [action_label]
        for avg_landmark in averaged_landmarks:
            row.extend(avg_landmark)
        
        writer.writerow(row)

# Liberamos recursos usados
cap.release()
cv2.destroyAllWindows()