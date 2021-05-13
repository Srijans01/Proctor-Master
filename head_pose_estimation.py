import cv2
import numpy as np
import math
from face_detector import get_face_detector, find_faces
from face_landmarks import get_landmark_model, detect_marks
import tkinter as tk


import pyttsx3 
import speech_recognition as sr 


import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Add,
    Concatenate,
    Conv2D,
    Input,
    Lambda,
    LeakyReLU,
    UpSampling2D,
    ZeroPadding2D,
    BatchNormalization
)
from tensorflow.keras.regularizers import l2
import wget


engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')

def speak(audio):
    engine.say(audio)
    engine.runAndWait()


def load_darknet_weights(model, weights_file):
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)
  
    layers = ['yolo_darknet',
            'yolo_conv_0',
            'yolo_output_0',
            'yolo_conv_1',
            'yolo_output_1',
            'yolo_conv_2',
            'yolo_output_2']

    for layer_name in layers:
        sub_model = model.get_layer(layer_name)
        for i, layer in enumerate(sub_model.layers):
          
            
            if not layer.name.startswith('conv2d'):
                continue

            batch_norm = None
            if i + 1 < len(sub_model.layers) and \
                    sub_model.layers[i + 1].name.startswith('batch_norm'):
                batch_norm = sub_model.layers[i + 1]

            filters = layer.filters
            size = layer.kernel_size[0]
            in_dim = layer.input_shape[-1]

            if batch_norm is None:
                conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)
            else:
                bn_weights = np.fromfile(
                    wf, dtype=np.float32, count=4 * filters)
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]

            conv_shape = (filters, in_dim, size, size)
            conv_weights = np.fromfile(
                wf, dtype=np.float32, count=np.product(conv_shape))

            conv_weights = conv_weights.reshape(
                conv_shape).transpose([2, 3, 1, 0])

            if batch_norm is None:
                layer.set_weights([conv_weights, conv_bias])
            else:
                layer.set_weights([conv_weights])
                batch_norm.set_weights(bn_weights)

    assert len(wf.read()) == 0, 'failed to read all data'
    wf.close()
    
def draw_outputs(img2, outputs, class_names):
    
    boxes, objectness, classes, nums = outputs
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    wh = np.flip(img2.shape[0:2])
    for i in range(nums):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        img2 = cv2.rectangle(img2, x1y1, x2y2, (255, 0, 0), 2)
        img2 = cv2.putText(img2, '{} {:.4f}'.format(
            class_names[int(classes[i])], objectness[i]),
            x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
    return img2

yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)],
                        np.float32) / 416

yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
    
def DarknetConv(x, filters, kernel_size, strides=1, batch_norm=True):
    
    if strides == 1:
        padding = 'same'
    else:
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)  
        padding = 'valid'
        
    x = Conv2D(filters=filters, kernel_size=kernel_size,
               strides=strides, padding=padding,
               use_bias=not batch_norm, kernel_regularizer=l2(0.0005))(x)
    
    if batch_norm:
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
    return x

def DarknetResidual(x, filters):

    prev = x
    x = DarknetConv(x, filters // 2, 1)
    x = DarknetConv(x, filters, 3)
    x = Add()([prev, x])
    return x
  
  
def DarknetBlock(x, filters, blocks):

    x = DarknetConv(x, filters, 3, strides=2)
    for _ in range(blocks):
        x = DarknetResidual(x, filters)
    return x

def Darknet(name=None):
    
    x = inputs = Input([None, None, 3])
    x = DarknetConv(x, 32, 3)
    x = DarknetBlock(x, 64, 1)
    x = DarknetBlock(x, 128, 2)  
    x = x_36 = DarknetBlock(x, 256, 8)  
    x = x_61 = DarknetBlock(x, 512, 8)
    x = DarknetBlock(x, 1024, 4)
    return tf.keras.Model(inputs, (x_36, x_61, x), name=name)

def YoloConv(filters, name=None):

    def yolo_conv(x_in):
        if isinstance(x_in, tuple):
            inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
            x, x_skip = inputs

            x = DarknetConv(x, filters, 1)
            x = UpSampling2D(2)(x)
            x = Concatenate()([x, x_skip])
        else:
            x = inputs = Input(x_in.shape[1:])

        x = DarknetConv(x, filters, 1)
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, filters, 1)
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, filters, 1)
        return Model(inputs, x, name=name)(x_in)
    return yolo_conv

def YoloOutput(filters, anchors, classes, name=None):

    def yolo_output(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, anchors * (classes + 5), 1, batch_norm=False)
        x = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2],
                                            anchors, classes + 5)))(x)
        return tf.keras.Model(inputs, x, name=name)(x_in)
    return yolo_output

def yolo_boxes(pred, anchors, classes):
    

    grid_size = tf.shape(pred)[1]
    box_xy, box_wh, objectness, class_probs = tf.split(
        pred, (2, 2, 1, classes), axis=-1)

    box_xy = tf.sigmoid(box_xy)
    objectness = tf.sigmoid(objectness)
    class_probs = tf.sigmoid(class_probs)
    pred_box = tf.concat((box_xy, box_wh), axis=-1) 

    grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  

    box_xy = (box_xy + tf.cast(grid, tf.float32)) / \
        tf.cast(grid_size, tf.float32)
    box_wh = tf.exp(box_wh) * anchors

    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

    return bbox, objectness, class_probs, pred_box

def yolo_nms(outputs, anchors, masks, classes):
    b, c, t = [], [], []

    for o in outputs:
        b.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
        c.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
        t.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))

    bbox = tf.concat(b, axis=1)
    confidence = tf.concat(c, axis=1)
    class_probs = tf.concat(t, axis=1)

    scores = confidence * class_probs
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
        scores=tf.reshape(
        scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
        max_output_size_per_class=100,
        max_total_size=100,
        iou_threshold=0.5,
        score_threshold=0.6
    )

    return boxes, scores, classes, valid_detections


def YoloV3(size=None, channels=3, anchors=yolo_anchors,
           masks=yolo_anchor_masks, classes=80):
  
    x = inputs = Input([size, size, channels], name='input')

    x_36, x_61, x = Darknet(name='yolo_darknet')(x)

    x = YoloConv(512, name='yolo_conv_0')(x)
    output_0 = YoloOutput(512, len(masks[0]), classes, name='yolo_output_0')(x)

    x = YoloConv(256, name='yolo_conv_1')((x, x_61))
    output_1 = YoloOutput(256, len(masks[1]), classes, name='yolo_output_1')(x)

    x = YoloConv(128, name='yolo_conv_2')((x, x_36))
    output_2 = YoloOutput(128, len(masks[2]), classes, name='yolo_output_2')(x)

    boxes_0 = Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], classes),
                     name='yolo_boxes_0')(output_0)
    boxes_1 = Lambda(lambda x: yolo_boxes(x, anchors[masks[1]], classes),
                     name='yolo_boxes_1')(output_1)
    boxes_2 = Lambda(lambda x: yolo_boxes(x, anchors[masks[2]], classes),
                     name='yolo_boxes_2')(output_2)

    outputs = Lambda(lambda x: yolo_nms(x, anchors, masks, classes),
                     name='yolo_nms')((boxes_0[:3], boxes_1[:3], boxes_2[:3]))

    return Model(inputs, outputs, name='yolov3')

def weights_download(out='models/yolov3.weights'):
    _ = wget.download('https://pjreddie.com/media/files/yolov3.weights', out='models/yolov3.weights')
    

yolo = YoloV3()
load_darknet_weights(yolo, 'models/yolov3.weights') 






def get_2d_points(img2, rotation_vector, translation_vector, camera_matrix, val):
    point_3d = []
    dist_coeffs = np.zeros((4,1))
    rear_size = val[0]
    rear_depth = val[1]
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))
    
    front_size = val[2]
    front_depth = val[3]
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)
    
    (point_2d, _) = cv2.projectPoints(point_3d,
                                      rotation_vector,
                                      translation_vector,
                                      camera_matrix,
                                      dist_coeffs)
    point_2d = np.int32(point_2d.reshape(-1, 2))
    return point_2d

def draw_annotation_box(img2, rotation_vector, translation_vector, camera_matrix,
                        rear_size=300, rear_depth=0, front_size=500, front_depth=400,
                        color=(255, 255, 0), line_width=2):
    
    rear_size = 1
    rear_depth = 0
    front_size = img2.shape[1]
    front_depth = front_size*2
    val = [rear_size, rear_depth, front_size, front_depth]
    point_2d = get_2d_points(img2, rotation_vector, translation_vector, camera_matrix, val)
    cv2.polylines(img2, [point_2d], True, color, line_width, cv2.LINE_AA)
    cv2.line(img2, tuple(point_2d[1]), tuple(
        point_2d[6]), color, line_width, cv2.LINE_AA)
    cv2.line(img2, tuple(point_2d[2]), tuple(
        point_2d[7]), color, line_width, cv2.LINE_AA)
    cv2.line(img2, tuple(point_2d[3]), tuple(
        point_2d[8]), color, line_width, cv2.LINE_AA)
    
    
def head_pose_points(img2, rotation_vector, translation_vector, camera_matrix):

    rear_size = 1
    rear_depth = 0
    front_size = img2.shape[1]
    front_depth = front_size*2
    val = [rear_size, rear_depth, front_size, front_depth]
    point_2d = get_2d_points(img2, rotation_vector, translation_vector, camera_matrix, val)
    y = (point_2d[5] + point_2d[8])//2
    x = point_2d[2]
    
    return (x, y)

#eye tracker
def eye_on_mask(mask, side, shape):

    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    l = points[0][0]
    t = (points[1][1]+points[2][1])//2
    r = points[3][0]
    b = (points[4][1]+points[5][1])//2
    return mask, [l, t, r, b]

def find_eyeball_position(end_points, cx, cy):

    x_ratio = (end_points[0] - cx)/(cx - end_points[2])
    y_ratio = (cy - end_points[1])/(end_points[3] - cy)
    if x_ratio > 3:
        return 1
    elif x_ratio < 0.33:
        return 2
    elif y_ratio < 0.33:
        return 3
    else:
        return 0

    
def contouring(thresh, mid, img2, end_points, right=False):
    
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    try:
        cnt = max(cnts, key = cv2.contourArea)
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        if right:
            cx += mid
        cv2.circle(img2, (cx, cy), 4, (0, 0, 255), 2)
        pos = find_eyeball_position(end_points, cx, cy)
        return pos
    except:
        pass
    
def process_thresh(thresh):

    thresh = cv2.erode(thresh, None, iterations=2) 
    thresh = cv2.dilate(thresh, None, iterations=4) 
    thresh = cv2.medianBlur(thresh, 3) 
    thresh = cv2.bitwise_not(thresh)
    return thresh

def print_eye_pos(img2, left, right):
    
    if left == right and left != 0:
        text = ''
        if left == 1:
            print('Looking left')
            text = 'Looking left'
        elif left == 2:
            print('Looking right')
            text = 'Looking right'
        elif left == 3:
            print('Looking up')
            text = 'Looking up'
        font = cv2.FONT_HERSHEY_SIMPLEX 
        cv2.putText(img2, text, (30, 30), font,  
                   1, (0, 255, 255), 2, cv2.LINE_AA) 

left = [36, 37, 38, 39, 40, 41]
right = [42, 43, 44, 45, 46, 47]




face_model = get_face_detector()
landmark_model = get_landmark_model()
cap = cv2.VideoCapture(0)
ret, img2 = cap.read()
size = img2.shape
font = cv2.FONT_HERSHEY_SIMPLEX 
thresh = img2.copy()
kernel = np.ones((9, 9), np.uint8)

model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                        ])


focal_length = size[1]
center = (size[1]/2, size[0]/2)
camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )

def nothing(x):
    pass
cv2.createTrackbar('threshold', 'image', 75, 255, nothing)

j=0
jc=0
while True:
    ret, img2 = cap.read()
    if ret == True:
        faces = find_faces(img2, face_model)
        for face in faces:
            marks = detect_marks(img2, landmark_model, face)
            image_points = np.array([
                                    marks[30],     # Nose tip
                                    marks[8],     # Chin
                                    marks[36],     # Left eye left corner
                                    marks[45],     # Right eye right corne
                                    marks[48],     # Left Mouth corner
                                    marks[54]      # Right mouth corner
                                ], dtype="double")
            dist_coeffs = np.zeros((4,1)) 
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_UPNP)

            
            (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
            
            for p in image_points:
                cv2.circle(img2, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
            
            
            p1 = ( int(image_points[0][0]), int(image_points[0][1]))
            p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
            x1, x2 = head_pose_points(img2, rotation_vector, translation_vector, camera_matrix)

            cv2.line(img2, p1, p2, (0, 255, 255), 2)
            cv2.line(img2, tuple(x1), tuple(x2), (255, 255, 0), 2)
            try:
                m = (p2[1] - p1[1])/(p2[0] - p1[0])
                ang1 = int(math.degrees(math.atan(m)))
            except:
                ang1 = 90
                
            try:
                m = (x2[1] - x1[1])/(x2[0] - x1[0])
                ang2 = int(math.degrees(math.atan(-1/m)))
            except:
                ang2 = 90


            m=0
            img21 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            img21= cv2.resize(img21, (320, 320))
            img21 = img21.astype(np.float32)
            img21 = np.expand_dims(img21, 0)
            img21 = img21 / 255
            class_names = [c.strip() for c in open("models/classes.TXT").readlines()]
            boxes, scores, classes, nums = yolo(img21)
            count=0
            for i in range(nums[0]):
                if int(classes[0][i] == 0):
                    count +=1
                if int(classes[0][i] == 67):
                    print('Mobile Phone detected')
                    speak("Mobile Phone detected")
                    m+=1
            if(m==1):
                speak("Please Don't use Phone will giving exam")
                speak(m)
            if(m==3):
                cv2.destroAllWindows()
                cap.release()
            if count == 0:
                print('No person detected')
                speak("No person detected")
            elif count > 1: 
                print('More than one person detected')
                speak("More than one person detected")
            

            img2 = draw_outputs(img2, (boxes, scores, classes, nums), class_names)
            
            
            rects = find_faces(img2, face_model)
    
            for rect in rects:
                shape = detect_marks(img2, landmark_model, rect)
                mask = np.zeros(img2.shape[:2], dtype=np.uint8)
                mask, end_points_left = eye_on_mask(mask, left, shape)
                mask, end_points_right = eye_on_mask(mask, right, shape)
                mask = cv2.dilate(mask, kernel, 5)
                
                eyes = cv2.bitwise_and(img2, img2, mask=mask)
                mask = (eyes == [0, 0, 0]).all(axis=2)
                eyes[mask] = [255, 255, 255]
                mid = (shape[42][0] + shape[39][0]) // 2
                eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
                threshold = cv2.getTrackbarPos('threshold', 'image')
                _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
                thresh = process_thresh(thresh)
                
                eyeball_pos_left = contouring(thresh[:, 0:mid], mid, img2, end_points_left)
                eyeball_pos_right = contouring(thresh[:, mid:], mid, img2, end_points_right, True)
                print_eye_pos(img2, eyeball_pos_left, eyeball_pos_right)

            if ang1 >= 48:
                print('Head down')
                # cv2.putText(img2, 'Head down', (30, 30), font, 2, (255, 255, 128), 3)
                speak("Head Down")
                j=int(j)
                j+=1
                j=str(j)
                cv2.putText(img2, 'Warning No.:'+j, (30, 50), font, 2, (255, 255, 128), 3)
                # cv2.putText(img2, 'Head left', (90, 30), font, 1 , (255, 255, 128), 3)
                j=int(j)
                speak("Warning Number")
                speak(j)
            elif ang1 <= -48:
                print('Head up')
                # cv2.putText(img2, 'Head up', (30, 30), font, 2, (255, 255, 128), 3)
                speak("Head up")
                j=int(j)
                j+=1
                j=str(j)
                cv2.putText(img2, 'Warning No.:'+j, (30, 50), font, 2, (255, 255, 128), 3)
                j=int(j)
                speak("Warning Number")
                speak(j)

            if ang2 >= 48:
                print('Head right')
                # cv2.putText(img2, 'Head right', (90, 30), font, 2, (255, 255, 128), 3)
                speak("Head Right")
                j=int(j)
                j+=1
                j=str(j)
                cv2.putText(img2, 'Warning No.:'+j, (30, 50), font, 2, (255, 255, 128), 3)
                j=int(j)
                speak("Warning Number")
                speak(j)

            elif ang2 <= -48:
                print('Head left')
                # cv2.putText(img2, 'Head left', (90, 30), font, 2, (255, 255, 128), 3)
                speak("Head Left")
                j=int(j)
                j+=1
                j=str(j)
                cv2.putText(img2, 'Warning No.:'+j, (30, 50), font, 2, (255, 255, 128), 3)
                j=int(j)
                speak("Warning Number")
                speak(j)
                
            if j==4 and jc==0:
                cv2.putText(img2, 'Final Warning', (30, 100), font, 1, (255, 255, 128), 3)
                speak("Final Warning , After this your Exam will be cancelled")
                jc+=1
            if j==5:
                cv2.destroAllWindows()
                cap.release()
            cv2.putText(img2, str(ang1), tuple(p1), font, 2, (128, 255, 255), 3)
            cv2.putText(img2, str(ang2), tuple(x1), font, 2, (255, 255, 128), 3)
        cv2.imshow('img2', img2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cv2.destroyAllWindows()
cap.release()