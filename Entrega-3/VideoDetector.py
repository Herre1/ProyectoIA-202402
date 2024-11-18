import cv2
import mediapipe as mp
import numpy as np
import joblib

# Cargar el modelo entrenado
model = joblib.load('Entrega-3/action_classifier_99_features_nuevo.pkl')

# Inicializar MediaPipe y OpenCV
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)  # Usar 0 para cámara web

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convertir la imagen a RGB y procesar con MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        # Convertir de nuevo a BGR para mostrar en OpenCV
        image_bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            # Dibujar los puntos de las articulaciones en la imagen
            mp_drawing.draw_landmarks(image_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Extraer las coordenadas (x, y, z) de cada uno de los 33 puntos clave
            landmarks = results.pose_landmarks.landmark
            frame_landmarks = []

            # Recopilar coordenadas x, y, z de cada punto
            for landmark in landmarks:
                frame_landmarks.extend([landmark.x, landmark.y, landmark.z])

            # Asegurarse de que frame_landmarks tiene exactamente 99 características
            frame_landmarks = np.array(frame_landmarks).reshape(1, -1)  # (1, 99) para pasar al modelo

            # Realizar la predicción
            prediction = model.predict(frame_landmarks)
            action = prediction[0]

            # Mostrar la acción en la ventana de video
            cv2.putText(image_bgr, f'Accion: {action}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Mostrar la imagen en tiempo real
        cv2.imshow("Deteccion de actividad en tiempo real", image_bgr)

        # Salir al presionar 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
