import super_gradients
import cv2

yolo_nas = super_gradients.training.models.get("yolo_nas_pose_l", pretrained_weights="coco_pose").cuda()
img = cv2.imread("fall.webp")
model_predictions  = yolo_nas.predict(img, conf=0.5)

prediction = model_predictions[0].prediction # One prediction per image - Here we work with 1 image, so we get the first.

bboxes = prediction.bboxes_xyxy # [Num Instances, 4] List of predicted bounding boxes for each object 
poses  = prediction.poses       # [Num Instances, Num Joints, 3] list of predicted joints for each detected object (x,y, confidence)
scores = prediction.scores      # [Num Instances] - Confidence value for each predicted instance



for i in range(len(poses)): 
    for c in poses[i]: 
        img = cv2.circle(img, (int(c[0]), int(c[1])), 2, (0, 0, 255), 
                    thickness=-1, lineType=cv2.FILLED) 
    # ind = poses.index(pose)
    box = bboxes[i]
    img = cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color=(255, 255,255), thickness=2)
    
    head_y = poses[i][0][1]
    back_y = (poses[i][8][1] + poses[i][11][1])/2
    pos = (int(box[0]-5), int(box[1]-10))
    if head_y >= back_y:
        text = "fall"
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        text_w, text_h = text_size
        cv2.rectangle(img, pos, (pos[0] + text_w, pos[1] + text_h), (255,255,255), -1)
        img = cv2.putText(img, text,  (pos[0], pos[1] + text_h + 1 - 1), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,0), thickness=2)
        print("fall")
    else:
        text = "not fall"
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        text_w, text_h = text_size
        cv2.rectangle(img, pos, (pos[0] + text_w, pos[1] + text_h), (255,255,255), -1)
        img = cv2.putText(img, text,  (pos[0], pos[1] + text_h + 1 - 1), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,0), thickness=2)
        print("not fall")

cv2.imwrite("ot.jpg", img)
    

