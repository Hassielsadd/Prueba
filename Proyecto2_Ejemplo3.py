from flask import Flask, render_template, request, redirect, send_file
import cv2
import numpy as np
import sqlite3
import os
from datetime import datetime

app = Flask(__name__)
MARKER_REAL_SIZE_CM = 5.0
VERDE_BAJO = np.array([36, 25, 25])
VERDE_ALTO = np.array([86, 255, 255])
PIXELS_POR_CM = 20
CM2_POR_PIXEL = 1 / (PIXELS_POR_CM ** 2)
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

os.makedirs("Evidencia_Hojas", exist_ok=True)
conn = sqlite3.connect("datos_hojas.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS mediciones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fecha TEXT,
    area_cm2 REAL
)''')
conn.commit()

def detectar_aruco(frame):
    corners, ids, _ = cv2.aruco.detectMarkers(frame, ARUCO_DICT)
    if ids is not None and len(corners) > 0:
        c = corners[0][0]
        pix_lado = np.linalg.norm(c[0] - c[1])
        escala = MARKER_REAL_SIZE_CM / pix_lado
        return escala, corners, ids
    return None, None, None

def calcular_area_foliar(frame):
    escala, corners, ids = detectar_aruco(frame)
    blur = cv2.GaussianBlur(frame, (7, 7), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    mascara = cv2.inRange(hsv, VERDE_BAJO, VERDE_ALTO)
    kernel = np.ones((5, 5), np.uint8)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_px = sum(cv2.contourArea(c) for c in contornos)

    area_cm2 = area_px * (escala ** 2) if escala else area_px * CM2_POR_PIXEL
    texto = f"Área: {area_cm2:.2f} cm²" if escala else f"Área (manual): {area_cm2:.2f} cm²"

    cv2.drawContours(frame, contornos, -1, (0, 255, 0), 2)
    if escala:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    cv2.putText(frame, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return frame, area_cm2

@app.route('/', methods=['GET', 'POST'])
def index():
    area_cm2 = None
    if request.method == 'POST':
        file = request.files['image']
        if file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            path = f"Evidencia_Hojas/original_{timestamp}.jpg"
            file.save(path)
            frame = cv2.imread(path)
            procesado, area_cm2 = calcular_area_foliar(frame)
            cv2.imwrite(f"static/original.jpg", frame)
            cv2.imwrite(f"static/procesado.jpg", procesado)

            fecha = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cursor.execute("INSERT INTO mediciones (fecha, area_cm2) VALUES (?, ?)", (fecha, area_cm2))
            conn.commit()

            return render_template("index.html", area=round(area_cm2, 2),
                                   original='static/original.jpg',
                                   procesado='static/procesado.jpg')
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
