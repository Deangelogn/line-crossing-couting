import cv2
import numpy as np
import time
from multiprocessing import Process, Manager
from person import Person
from zone import Zone

ID_COUNTER = -1
CROSSING_COUNTER = 0

def searchOnList(local, person_list):
    x1, y1, x2, y2 = local
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    for i, person in enumerate(person_list):
        (px1, py1, px2, py2) = local
        if (px1 <= cx <= px2) and (py1 <= cy <= py2):
            return i
    return None

def intersectionTest(y, coordinateYLineIn, coordinateYlineOut):
    absolutelyDif = abs(y - coordinateYLineIn)

    if ((absolutelyDif <= 10) and (y < coordinateYlineOut)):
        return 1
    else:
        return 0

def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = class_id
    color = COLORS[0]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def generate_id():
    global ID_COUNTER
    ID_COUNTER += 1

    if ID_COUNTER > 100000000:
        ID_COUNTER = 0

    return ID_COUNTER


def find_person(person_location, people_list):
    x1, y1, x2, y2 = person_location
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    for i, old_person in enumerate(people_list):
        px1, py1, px2, py2 = old_person.get_last_position()

        if (px1 <= cx <= px2) and (py1 <= cy <= py2):
            return i

    return None


def processing_yolo(input_process, output_process):
    weights = "yolov3.weights"
    config = "yolov3.cfg"
    scale = 0.00392
    net = cv2.dnn.readNet(weights, config)

    while True:
        if len(input_process) == 0:
            continue

        frame = input_process[0]

        blob = cv2.dnn.blobFromImage(frame, scale, (416,416), (0,0,0),
                                     True, crop=False)
        net.setInput(blob)
        outs = net.forward(get_output_layers(net))

        output_process.append(outs)
        input_data.pop()


process_input = Manager()
process_output = Manager()
input_data = process_input.list()
output_data = process_output.list()

classes = None
classes_file = "yolov3.txt"

with open(classes_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

yolo_process = Process(target=processing_yolo, args=(input_data, output_data))
yolo_process.start()

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
cap = cv2.VideoCapture("/home/eu/Desktop/videos_test/vid2.mp4")
_, frame = cap.read()

transition_region = (200, 210)
indices = []
people_list = []

Width = frame.shape[1]
Height = frame.shape[0]
zone = Zone(location=200, th=5, type="horizontal", size_limit=Width)

normal_color = (0, 0, 0)
detected_color = (0, 255, 255)

vid = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (Width, Height))

while True:
    ret, frame = cap.read()

    if not ret:
        break

    if len(input_data)==0:
        input_data.append(frame)

    if len(output_data) > 0:
        current_people = []
        outs = output_data.pop()

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if (confidence > 0.5) and (class_id == 0):
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w // 2
                    y = center_y - h // 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

                    # checking people
                    x2 = x + w
                    y2 = y + h
                    person_location = (x, y, x2, y2)

                    person_id = find_person(person_location, people_list)

                    if person_id is None:
                        new_person = Person()
                        new_person.set_id(generate_id())
                        new_person.add_location(person_location)
                        current_people.append(new_person)
                    else:
                        old_person = people_list[person_id]
                        old_person.add_location(person_location)
                        current_people.append(old_person)

        people_list = current_people

        # Use this to remove multiple detections
        # indices =\
        #     cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # Draw the transition zone
    zone_start, zone_end = zone.get_limits()
    cv2.rectangle(frame, zone_start, zone_end, (0, 0, 255), -1)

    for i, p in enumerate(people_list):
        location = p.get_last_position()
        left, top, right, bottom = location
        cx, cy = (left + right) // 2, (top + bottom) // 2

        cv2.putText(frame, '{}'.format(p.get_id()), (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        events = zone.analyse_trajectory_in_zone(p.get_center_trajectory())

        if not p.get_check():
            if ("OuP" in events) or ("CrP" in events):
                print("{} crossed the line".format(p.get_id()))
                CROSSING_COUNTER += 1
                people_list[i].set_check()

            # Draw a black rectangle for people the crossed the line
            cv2.rectangle(frame, (left, top), (right, bottom),
                          normal_color, 1)
        else:
            # Draw a yellow rectangle for people the crossed the line
            cv2.rectangle(frame, (left, top), (right, bottom),
                          detected_color, 1)

    # Show the number of people that crossed the line from top-down direction
    cv2.putText(frame, '{}'.format(CROSSING_COUNTER), (20, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    cv2.imshow("object detection", frame)
    vid.write(frame)
    key = cv2.waitKey(100)

    if key == ord('q'):
        break

cap.release()
vid.release()
cv2.destroyAllWindows()
yolo_process.terminate()
