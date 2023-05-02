import cv2
import torch
import pandas as pd

#Importando el modelo de YOLOv5 -> small para agilizar el proceso
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
    return model

#Filtro de los bounding boxes mediante pandas, se utilizan dataframes para guardar valores de la matriz de las predicciones
def get_bboxes(preds: object):
    # xmin, ymin, xmax, ymax
    df = preds.pandas().xyxy[0]
    #df = df[df["confidence"] >= 0.2]
    #df = df[df["name"] == "car"]
    return df[["xmin", "ymin", "xmax", "ymax"]].values.astype(int)


def detector(cap: object):

    model = load_model()

    while cap.isOpened():
        status, frame = cap.read()
        if not status:
            break

        preds = model(frame) #poner a predecir el modelo bajo los estandares de mi video, resolucion, calidad, etc
        ######### EDITABLE SPACE PRINT #########
        bboxes = get_bboxes(preds)

        for box in bboxes:
            cv2.rectangle(img=frame, pt1=(box[0], box [1]), pt2=(box[2],box[3]), color=(255, 0, 0), thickness=1)
        
        ######### /EDITABLE SPACE PRINT #########

        cv2.imshow("frame", frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    cap.release()


if __name__ == '__main__':
    cap = cv2.VideoCapture("data/traffic.mp4")

    detector(cap)
