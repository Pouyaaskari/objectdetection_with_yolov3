import cv2
import numpy as np

colors = np.random.uniform(0, 255, size= (80, 3))
print(colors)

def load_data_and_preprocess(address):
    img = cv2.imread(address)
    h, w = img.shape[:2]
    preprocessed_image = cv2.dnn.blobFromImage(img, scalefactor= 1/255, size= (416, 416), 
                                               swapRB = True, crop = False)
    
    return img, preprocessed_image, h, w

def read_model_and_label(label_address, weights_address, config_address):
    labels = open(label_address).read().strip().split("\n")
    
    net = cv2.dnn.readNet(weights_address, config_address)

    return labels, net

def inference(pre_processed_image, net):
    net.setInput(pre_processed_image)
    output_layers = ["yolo_82", "yolo_94", "yolo_106"]
    predictions = net.forward(output_layers)

    print(np.array(predictions)[2].shape)
    return predictions

def post_processing(predictions, w, h):
    classIDs = []
    confidences = []
    boxes = []

    for layer in predictions:
        for detected_object in layer:
            scores = detected_object[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > 0.:
                box = detected_object[0:4]*np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - width/2)
                y = int(centerY - height/2)

                classIDs.append(classID)
                confidences.append(float(confidence))
                boxes.append([x, y, int(width), int(height)])
    return classIDs, confidences, boxes

def show_results(img, classIDs, confidences, boxes, labels):
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.5)

    for i in idxs.flatten():
        (x, y) = boxes[i][0], boxes[i][1]
        (w, h) = boxes[i][2], boxes[i][3]

        cv2.rectangle(img, (x, y), (x+w, y+h), colors[i], 3)
        text = "{}:{:.2f}".format(labels[classIDs[i]], confidences[i])
        cv2.rectangle(img, (x, y-45), (x+150, y), colors[i], -1)
        cv2.putText(img, text, (x, y-10),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
    cv2.imwrite("pic_yolo_1.jpg", img)
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img, pre_img, h, w = load_data_and_preprocess("test.jpg")

labels, net = read_model_and_label(r"C:\Users\pouya_pc\Desktop\Yolo3\coco.names",
                     r"C:\Users\pouya_pc\Desktop\Yolo3\yolov3.weights", r"C:\Users\pouya_pc\Desktop\Yolo3\yolov3.cfg")

predictions = inference(pre_img, net)
classIDs, confidences, boxes = post_processing(predictions, w, h)
show_results(img, classIDs, confidences, boxes, labels)
