import cv2
import mediapipe as mp
import csv
import numpy as np
import os

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Ruta de la carpeta con los videos
video_folder_path = 'Entrega-1/Videos/'

# Función para obtener la etiqueta de acción según el nombre del archivo de video
def get_action_label(filename):
    if "CaminandoE" in filename:
        return "Caminando_Espalda"
    elif "CaminandoF" in filename:
        return "Caminando_Frente"
    elif "Giro" in filename:
        return "Giro"
    elif "Parado" in filename:
        return "Parado"
    elif "Sentado" in filename:
        return "Sentado"
    elif "Quieto" in filename:
        return "Quieto"
    elif "CaminandoL" in filename:
        return "Caminando_Lado"
    elif "Parandose" in filename:
        return "Parandose"
    elif "Sentandose" in filename:
        return "Sentandose"
    else:
        return "Desconocido"  # Por si algún archivo no coincide con ninguna etiqueta

# Procesa cada archivo de video en la carpeta
for video_filename in os.listdir(video_folder_path):
    # Verifica que sea un archivo de video (por ejemplo, termina en .mp4)
    if video_filename.endswith('.mp4'):
        # Ruta completa del video
        video_path = os.path.join(video_folder_path, video_filename)
        
        # Determina la etiqueta de acción basada en el nombre del archivo
        action_label = get_action_label(video_filename)
        
        # Abre el archivo de video
        cap = cv2.VideoCapture(video_path)
        
        # Genera el nombre del archivo CSV de salida
        csv_filename = f"database_pose_{video_filename.split('.')[0]}.csv"
        csv_path = os.path.join(video_folder_path, "../VideosDataSet", csv_filename)
        
        # Abre el archivo CSV para escritura
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            
            # Escribe la cabecera del CSV
            header = ['label']
            for i in range(33):
                header += [f'x{i}', f'y{i}', f'z{i}']
            writer.writerow(header)

            # Variables para agrupar cada 5 fotogramas
            landmarks_buffer = []
            buffer_size = 5

            # Procesa cada fotograma del video
            while cap.isOpened():
                ret, frame = cap.read()
                
                if not ret:
                    break

                # Convierte el fotograma de BGR a RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Procesa el fotograma con Mediapipe para detectar las articulaciones
                results = pose.process(rgb_frame)

                if results.pose_landmarks:
                    # Extrae las coordenadas x, y, z de cada articulación
                    landmarks = results.pose_landmarks.landmark
                    frame_landmarks = []

                    # Agrega las coordenadas de los 33 puntos clave obtenidos de las articulaciones
                    for landmark in landmarks:
                        frame_landmarks.append([landmark.x, landmark.y, landmark.z])
                    
                    # Añade los puntos clave del fotograma al buffer
                    landmarks_buffer.append(frame_landmarks)

                    # Si tenemos suficientes fotogramas en el buffer
                    if len(landmarks_buffer) == buffer_size:
                        # Calcula el promedio de las coordenadas de cada punto clave a lo largo de los últimos 5 fotogramas
                        averaged_landmarks = np.mean(landmarks_buffer, axis=0)
                        
                        # Crea la fila para escribir en el CSV
                        row = [action_label]
                        for avg_landmark in averaged_landmarks:
                            row.extend(avg_landmark)
                        
                        writer.writerow(row)

                        # Limpia el buffer
                        landmarks_buffer = []

            # Si quedan fotogramas en el buffer al final, promediarlos y escribirlos
            if len(landmarks_buffer) > 0:
                averaged_landmarks = np.mean(landmarks_buffer, axis=0)
                row = [action_label]
                for avg_landmark in averaged_landmarks:
                    row.extend(avg_landmark)
                
                writer.writerow(row)

        # Libera recursos usados por el video actual
        cap.release()

# Libera recursos de Mediapipe
pose.close()
cv2.destroyAllWindows()
