"""
import cv2
import numpy as np

# Carregar a imagem
img = cv2.imread('1.mp4')

# Obter a largura e a altura da imagem
height, width, _ = img.shape

# Carregar o modelo YOLO
net = cv2.dnn.readNet('yolov3.weights', 'darknet\cfg\yolov3.cfg')

# Obter os nomes das classes
with open("darknet\data\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Obter os nomes das camadas de saída
layer_names = net.getLayerNames()

output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Construir um blob da imagem
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)

# Realizar a detecção de objetos
outs = net.forward(output_layers)

# Inicializar listas para as caixas delimitadoras, confianças e IDs de classe
class_ids = []
confidences = []
boxes = []

# Processar as saídas
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Obter as coordenadas e dimensões da caixa delimitadora
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Obter as coordenadas do canto superior esquerdo da caixa delimitadora
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Realizar a supressão não máxima para eliminar caixas delimitadoras redundantes
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

# Inicializar um dicionário para contar os objetos
object_counter = {}

# Processar as caixas delimitadoras restantes
for i in indices:
    box = boxes[i]
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]

    # Obter o rótulo da classe
    label = str(classes[class_ids[i]])

    # Contar o objeto
    if label in object_counter:
        object_counter[label] += 1
    else:
        object_counter[label] = 1

    # Desenhar a caixa delimitadora e o rótulo na imagem
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Imprimir o número de objetos
for object_name, count in object_counter.items():
    print(f'Número de {object_name}s: {count}')

# Mostrar a imagem
cv2.imshow('Imagem', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
"""
import cv2
import numpy as np

# Carregar o modelo YOLO
net = cv2.dnn.readNet('yolov3.weights', 'darknet\cfg\yolov3.cfg')

# Obter os nomes das classes
with open("darknet\data\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Obter os nomes das camadas de saída
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Abrir o vídeo
cap = cv2.VideoCapture('1.mp4')

while cap.isOpened():
    # Ler o quadro atual
    ret, img = cap.read()
    if not ret:
        break

    # Obter a largura e a altura da imagem
    height, width, _ = img.shape

    # Construir um blob da imagem
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    # Realizar a detecção de objetos
    outs = net.forward(output_layers)

    # Inicializar listas para as caixas delimitadoras, confianças e IDs de classe
    class_ids = []
    confidences = []
    boxes = []

    # Processar as saídas
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Obter as coordenadas e dimensões da caixa delimitadora
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Obter as coordenadas do canto superior esquerdo da caixa delimitadora
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Realizar a supressão não máxima para eliminar caixas delimitadoras redundantes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

    # Inicializar um dicionário para contar os objetos
    object_counter = {}

    # Processar as caixas delimitadoras restantes
    for i in indices:
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]

        # Obter o rótulo da classe
        label = str(classes[class_ids[i]])

        # Contar o objeto
        if label in object_counter:
            object_counter[label] += 1
        else:
            object_counter[label] = 1

        # Desenhar a caixa delimitadora e o rótulo na imagem
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Imprimir o número de objetos
    for object_name, count in object_counter.items():
        print(f'Número de {object_name}s: {count}')

    # Mostrar a imagem
    cv2.imshow('Vídeo', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
"""

import cv2
import numpy as np

# Carregar o modelo YOLO
net = cv2.dnn.readNet('yolov3.weights', 'darknet\cfg\yolov3.cfg')

# Obter os nomes das classes
with open("darknet\data\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Obter os nomes das camadas de saída
nomes_camadas = net.getLayerNames()
camadas_saida = [nomes_camadas[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Abrir o vídeo
cap = cv2.VideoCapture('1.mp4')

while cap.isOpened():
    # Ler o quadro atual
    ret, img = cap.read()
    if not ret:
        break

    # Obter a largura e a altura da imagem
    altura, largura, _ = img.shape

    # Construir um blob da imagem
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    # Realizar a detecção de objetos
    saidas = net.forward(camadas_saida)

    # Inicializar listas para as caixas delimitadoras, confianças e IDs de classe
    ids_classes = []
    confiancas = []
    caixas = []

    # Processar as saídas
    for saida in saidas:
        for deteccao in saida:
            pontuacoes = deteccao[5:]
            id_classe = np.argmax(pontuacoes)
            confianca = pontuacoes[id_classe]
            if confianca > 0.5:
                # Obter as coordenadas e dimensões da caixa delimitadora
                centro_x = int(deteccao[0] * largura)
                centro_y = int(deteccao[1] * altura)
                w = int(deteccao[2] * largura)
                h = int(deteccao[3] * altura)

                # Obter as coordenadas do canto superior esquerdo da caixa delimitadora
                x = int(centro_x - w / 2)
                y = int(centro_y - h / 2)

                caixas.append([x, y, w, h])
                confiancas.append(float(confianca))
                ids_classes.append(id_classe)

    # Realizar a supressão não máxima para eliminar caixas delimitadoras redundantes
    indices = cv2.dnn.NMSBoxes(caixas, confiancas, 0.5, 0.3)

    # Inicializar um dicionário para contar os objetos
    contador_objetos = {}

    # Processar as caixas delimitadoras restantes
    for i in indices:
        caixa = caixas[i]
        x = caixa[0]
        y = caixa[1]
        w = caixa[2]
        h = caixa[3]

        # Obter o rótulo da classe
        rotulo = str(classes[ids_classes[i]])

        # Contar o objeto
        if rotulo in contador_objetos:
            contador_objetos[rotulo] += 1
        else:
            contador_objetos[rotulo] = 1

        # Desenhar a caixa delimitadora e o rótulo na imagem
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, rotulo, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Imprimir o número de objetos
    for nome_objeto, contagem in contador_objetos.items():
        print(f'Número de {nome_objeto}s: {contagem}')

    # Mostrar a imagem
    cv2.imshow('Vídeo', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()