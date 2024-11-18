# Sistema de anotación de video  (Primera entrega)

Integrantes :
 - Manuel Herrera A00381987
 - Kevin Loachamin A00382106
 -Juan Esteban Brawn A00381715


## 1. Pregunta de investigación
Pregunta principal: ¿Cómo podemos desarrollar un sistema preciso en tiempo real para el reconocimiento de la actividad humana y el seguimiento de las articulaciones mediante técnicas de visión artificial?

Subpreguntas:
- ¿Cuáles son las características más efectivas para distinguir entre diferentes actividades (caminar, girar, sentarse, estar de pie)?
- ¿Cómo podemos rastrear y medir con precisión los ángulos de las articulaciones y las inclinaciones laterales en tiempo real?
- ¿Cuál es el equilibrio óptimo entre la precisión del modelo y el rendimiento en tiempo real?

## 2. Tipo de problema
Este es un problema multifacético que combina:
1. Clasificación de videos (aprendizaje supervisado)
2. Estimación de posturas (visión por computadora)
3. Análisis de series temporales (para seguimiento de movimientos)

## 3. Metodología: Adaptación de CRISP-DM

### Comprensión empresarial
- Contexto: Necesidad de reconocimiento automático de actividades en varios campos (asistencia sanitaria, deportes, ergonomía)
- Objetivos: Desarrollar un sistema que pueda:
- Clasificar actividades específicas en tiempo real
- Realizar un seguimiento de los movimientos articulares y la postura
- Proporcionar mediciones precisas de ángulos

### Comprensión de los datos
- Plan de recopilación de datos:
1. Grabar videos de varios sujetos realizando actividades específicas
2. Garantizar la diversidad de sujetos, perspectivas y velocidades de movimiento
3. Utilizar entornos interiores y exteriores para lograr solidez

- Requisitos iniciales de datos:
- Mínimo de 20 sujetos
- 5 repeticiones de cada actividad por sujeto
- Múltiples ángulos de cámara (frontal, lateral, 45 grados)

### Preparación de datos
1. Video Preprocesamiento:
- Extracción de fotogramas
- Estandarización de la resolución
- Normalización de la iluminación

2. Estrategia de anotación:
- Utilizar LabelStudio para la anotación manual
- Definir directrices claras de anotación
- Implementar control de calidad para las anotaciones

3. Extracción de características:
- Coordenadas de las articulaciones mediante MediaPipe
- Características derivadas (ángulos, velocidades)
- Secuencias temporales

## 4. Métricas de rendimiento
1. Clasificación de la actividad:
- Exactitud
- Precisión
- Recordatorio
- Puntuación F1
- Matriz de confusión

2. Seguimiento de articulaciones:
- Precisión media promedio (mAP)
- Error de ángulo de articulación (grados)
- Estabilidad del seguimiento

3. Rendimiento en tiempo real:
- Fotogramas por segundo (FPS)
- Latencia

## 5. Consideraciones éticas
1. Privacidad:
- Consentimiento para la recopilación de datos
- Almacenamiento seguro de datos de vídeo
- Anonimización de sujetos

2. Mitigación de sesgos:
- Sujetos diversos piscina
- Consideración de diferentes tipos de cuerpo y movilidades
- Sensibilidad cultural en las definiciones de actividad

3. Transparencia:
- Documentación clara de las limitaciones del sistema
- Resultados explicables

## 6. Próximos pasos
1. Semana 11-12:
- Configurar el entorno de recopilación de datos
- Comenzar las sesiones de grabación de video
- Implementar la integración inicial de MediaPipe

2. Semana 13:
- Completar la recopilación de datos
- Terminar la anotación
- Comenzar la extracción de características

3. Semana 14:
- Comenzar el entrenamiento del modelo
- Implementar la canalización de procesamiento en tiempo real

## 7. Estrategias adicionales de adquisición de datos
1. Aumento de datos:
- Rotación, escalado y volteo de videos existentes
- Variaciones de velocidad
- Ajustes de iluminación

2. Generación de datos sintéticos:
- Usar modelos 3D para generar datos de entrenamiento adicionales
- Simular diferentes ángulos y movimientos de cámara

3. Conjuntos de datos públicos:
- Identificar e incorporar conjuntos de datos públicos relevantes de actividad humana
- Garantizar la compatibilidad y la alineación del preprocesamiento