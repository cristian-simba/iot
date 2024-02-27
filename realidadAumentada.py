import cv2
import numpy as np
import cv2.aruco as aruco

import requests

token = 'gCuq9JhucGltecNJ6RVZKHlryR6FF3Wc'

# Pines Virtuales
pin_virtual_humedad = 'V1'
pin_virtual_temperatura = 'V2'
pin_virtual_luz = 'V3'

blynk_api_humedad = f'https://blynk.cloud/external/api/get?token={token}&{pin_virtual_humedad}'
blynk_api_temperatura = f'https://blynk.cloud/external/api/get?token={token}&{pin_virtual_temperatura}'
blynk_api_luz = f'https://blynk.cloud/external/api/get?token={token}&{pin_virtual_luz}'

parametros = cv2.aruco.DetectorParameters()
aruco_diccionario = aruco.getPredefinedDictionary(aruco.DICT_6X6_100)

# Camara
captura = cv2.VideoCapture(0)
ip = 'http://192.168.3.195:4747/video'
captura.open(ip)

# Cargar imagen
imagen = cv2.imread('clima.jpg')

while True:
    # Peticiones a Blynk
    valor_sensor_humedad = requests.get(blynk_api_humedad)
    valor_sensor_temperatura = requests.get(blynk_api_temperatura)
    valor_sensor_luz = requests.get(blynk_api_luz)

    # Obtener los valores de Blynk
    valorHumedad = valor_sensor_humedad.text
    valorTemperatura = valor_sensor_temperatura.text
    valorLuz = valor_sensor_luz.text

    # Otras configuraciones
    lectura, frame = captura.read()
    cuadro_gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = aruco.ArucoDetector(aruco_diccionario, parametros)
    esquinas, identificador, puntosRechazados = detector.detectMarkers(cuadro_gris)
    
    if identificador is not None:
        aruco.drawDetectedMarkers(frame, esquinas, identificador)

        for i in range(len(identificador)):
            marker_corners = esquinas[i][0]
            x,y,w,h = cv2.boundingRect(marker_corners)
            imagen_sobrepuesta = cv2.resize(imagen, (w,h))

            frame[y:y+h, x:x+w] = imagen_sobrepuesta
            
            texto = f'Humedad: {valor_sensor_humedad} Temperatura: {valor_sensor_temperatura} Luz: {valor_sensor_luz}'
            cv2.putText(frame, texto, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            # cv2.putText(frame, "Humedad: " + valorHumedad + "Temperatura: ", valorTemperatura + "Luz: ", valorLuz, (x,y,-10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),2)
        
        cv2.imshow('Aruco', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

captura.release()  
cv2.destroyAllWindows
