import cv2 as cv 
import numpy as np

# Distance constants 
KNOWN_DISTANCE = 37 #INCHES.Distance from camera to object measured
PERSON_WIDTH = 16 #INCHES
MOBILE_WIDTH = 3.0 #INCHES
CHAIR_WIDTH = 7.0  # INCHES
CUP_WIDTH = 3.5  # INCHES

# Object detector constant 
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3

# colors for object detected
COLORS = [(255,0,0),(255,0,255),(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN =(0,255,0)
BLACK =(0,0,0)
# defining fonts 
FONTS = cv.FONT_HERSHEY_COMPLEX

# getting class names from classes.txt file 
class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]
#  setting up opencv net
yoloNet = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')

yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

# object detector funciton /method
def object_detector(image):
    classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    # creating empty list to add objects data
    data_list =[]
    for (classid, score, box) in zip(classes, scores, boxes):
        # define color of each, object based on its class id 
        color= COLORS[int(classid) % len(COLORS)]
    
        label = "%s : %f" % (class_names[classid[0]], score)

        # draw rectangle on and label on object
        cv.rectangle(image, box, color, 2)
        cv.putText(image, label, (box[0], box[1]-14), FONTS, 0.5, color, 2)
    
        # getting the data 
        # 1: class name  2: object width in pixels, 3: position where have to draw text(distance)
        if classid == 0: # person class id 
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
        elif classid == 67: # phone
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2)])
        elif classid == 56: # chair
            data_list.append(
                [class_names[classid[0]], box[2], (box[0], box[1]-2)])
        elif classid == 41:  # cup
            data_list.append(
                [class_names[classid[0]], box[2], (box[0], box[1]-2)])
    return data_list

'''
    Tính toán Độ dài tiêu cự (khoảng cách giữa ống kính đến cảm biến CMOS)
    :param1 Measure_Distance(int): Là khoảng cách được đo từ đối tượng đến Camera trong khi Chụp ảnh tham chiếu
    :param2 Real_Width(int): Chiều rộng thực tế của đối tượng, trong thế giới thực (như Chiều rộng người = 16 inch)
    :param3 width_in_rf(int): chiều rộng của đối tượng trong khung/hình ảnh trong trường hợp trong hình ảnh tham chiếu 
    :retrun Focal_Length(Float):
    '''
def focal_length_finder (measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width

    return focal_length


# distance finder function
'''
    Ước tính khoảng cách giữa đối tượng và máy ảnh
    :param1 Focal_length(float): trả về bởi hàm Focal_Length_Finder
    :param2 real_object_width(int): Đó là Chiều rộng thực tế của đối tượng, trong thế giới thực (như Chiều rộng người = 16 inch)
    :param3 object_Width_Frame(int): chiều rộng của đối tượng trong ảnh (webcam)
    :return Khoảng cách(float): khoảng cách Ước tính
'''
def distance_finder(focal_length, real_object_width, width_in_frame):
    distance = (real_object_width * focal_length) / width_in_frame
    return distance


# reading the reference image from dir
ref_mobile = cv.imread('dataset/images2.png')
ref_person = cv.imread('dataset/images1.png')
ref_chair = cv.imread('dataset/images5.png')
ref_cup = cv.imread('dataset/images3.png')

mobile_data = object_detector(ref_mobile)
mobile_width_in_rf = mobile_data[1][1]

person_data = object_detector(ref_person)
person_width_in_rf = person_data[0][1]

chair_data = object_detector(ref_chair)
chair_width_in_rf = chair_data[0][1]

cup_data = object_detector(ref_cup)
cup_width_in_rf = cup_data[1][1]

print(
    f"Person width in pixels : {person_width_in_rf} mobile width in pixel: {mobile_width_in_rf} chair width in pixel: {chair_width_in_rf} cup width in pixel: {cup_width_in_rf}")

# finding focal length 
focal_person = focal_length_finder(28, PERSON_WIDTH, person_width_in_rf)

focal_mobile = focal_length_finder(20, MOBILE_WIDTH, mobile_width_in_rf)

focal_chair = focal_length_finder(74, CHAIR_WIDTH, chair_width_in_rf)

focal_cup = focal_length_finder(20, CUP_WIDTH, cup_width_in_rf)
cap = cv.VideoCapture(0)
while True:
    ret, frame = cap.read()

    data = object_detector(frame) 
    for d in data:
        if d[0] =='person':
            distance = distance_finder(focal_person, PERSON_WIDTH, d[1])
            x, y = d[2]
        elif d[0] =='cell phone':
            distance = distance_finder(focal_mobile, MOBILE_WIDTH, d[1])
            x, y = d[2]
        elif d[0] == 'chair':
            distance = distance_finder(focal_chair, CHAIR_WIDTH, d[1])
            x, y = d[2]
        elif d[0] == 'cup':
            distance = distance_finder(focal_cup, CUP_WIDTH, d[1])
            x, y = d[2]
        cv.rectangle(frame, (x, y-3), (x+150, y+23),BLACK,-1 )
        cv.putText(frame, f'Dis: {round(distance*0.0254,2)} m',
                   (x+5, y+13), FONTS, 0.48, GREEN, 2)

    cv.imshow('frame',frame)
    
    key = cv.waitKey(1)
    if key ==ord('q'):
        break
cv.destroyAllWindows()
cap.release()

