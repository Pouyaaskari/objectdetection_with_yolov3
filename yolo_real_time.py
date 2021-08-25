import cv2
import numpy as np

cap=cv2.VideoCapture(0)
classFile="coco.names"
className=[]
confidencethreshold=0.5
nms_threshold=0.3

with open(classFile,"rt") as f:
    className=f.read().rstrip("\n").split("\n")

modelConfig="yolov3.cfg"
modelweights="yolov3.weights"
net=cv2.dnn.readNetFromDarknet(modelConfig,modelweights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def find_object(outputs,image):
    ht,wt,ct=image.shape
    bbox = []
    classIds=[]
    confs=[]

    for output in outputs:
        for detection in output:
            scores=detection[5:]
            classId=np.argmax(scores)
            confidence=scores[classId]
            if confidence > confidencethreshold:
                w,h=int(detection[2]*wt),int(detection[3]*ht)
                x,y = int((detection[0]*wt)-w/2) , int((detection[1]*ht)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))
    indices=cv2.dnn.NMSBoxes(bbox, confs, confidencethreshold, nms_threshold)
    for i in indices:
        i=i[0]
        box=bbox[i]
        x,y,w,h=box[0],box[1],box[2],box[3]
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(image,f"{className[classIds[i]].upper()} {int(confs[i]*100)}%",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),2)




while True:
    success,image=cap.read()
    blob=cv2.dnn.blobFromImage(image,1/255,(320,320),[0,0,0],1,crop=False)
    net.setInput(blob)

    layerNames=net.getLayerNames()
    outputNames=[layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
    outputs=net.forward(outputNames)
    find_object(outputs,image)





    cv2.imshow("Image",image)
    if cv2.waitKey(1) & 0xFF==ord("q"):
        break

cap.release()
cv2.destroyAllWindows()