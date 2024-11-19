import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import mediapipe as mp
import cv2
import base64
from io import BytesIO
import random

app = Flask(__name__)

# Configuración de la carpeta de carga
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Tamaño máximo de 16MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Crear carpeta si no existe

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def rotate_image_180(image):
    """Rota la imagen 180 grados."""
    return np.rot90(image, k=2)  # Rotar 180 grados

def draw_keypoints(image, landmarks, key_points=None):
    """Dibuja los puntos clave en una imagen."""
    height, width = image.shape
    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap='gray')

    for idx, landmark in enumerate(landmarks):
        if key_points and idx not in key_points:
            continue

        x = int(landmark.x * width)
        y = int(landmark.y * height)
        plt.plot(x, y, 'rx')

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def process_image(image, landmarks, key_points):
    flipped_image = np.flip(image, axis=1)  # Invertir horizontalmente
    brightened_image = np.clip(random.uniform(1.5, 2) * image, 0, 255)  # Aumentar brillo
    rotated_image = rotate_image_180(image)  # Rotar 180 grados

    flipped = draw_keypoints(flipped_image, landmarks, key_points)
    brightened = draw_keypoints(brightened_image, landmarks, key_points)
    rotated = draw_keypoints(rotated_image, landmarks, key_points)

    return flipped, brightened, rotated

def analyze_face(image_path):
    try:
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )

        image = cv2.imread(image_path)
        if image is None:
            raise Exception("No se pudo cargar la imagen.")
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        results = face_mesh.process(rgb_image)
        if not results.multi_face_landmarks:
            raise Exception("No se detectó un rostro en la imagen.")

        landmarks = results.multi_face_landmarks[0].landmark
        key_points = [33, 133, 362, 263, 1, 61, 291, 199, 94, 0, 24, 130, 359, 288, 378]

        original = draw_keypoints(gray_image, landmarks, key_points)
        flipped, brightened, rotated = process_image(gray_image, landmarks, key_points)

        return {
            "original": original,
            "flipped": flipped,
            "brightened": brightened,
            "rotated": rotated
        }
    except Exception as e:
        return {"error": str(e)}

@app.route('/')
def home():
    images = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if allowed_file(f)]
    return render_template('index.html', images=images)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'files' not in request.files or not request.files.getlist('files'):
            return jsonify({'error': 'No se cargaron archivos.'}), 400

        images_data = {}

        for file in request.files.getlist('files'):
            if not allowed_file(file.filename):
                return jsonify({'error': 'Tipo de archivo no válido.'}), 400

            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            images = analyze_face(filepath)
            images_data[filename] = images

        return jsonify({'success': True, 'images': images_data})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
