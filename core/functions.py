import os
import cv2
import math
import random
import numpy as np
import tensorflow as tf
import pytesseract
from core.utils import read_class_names
from core.config import cfg
from itertools import combinations


# function to count objects, can return total classes or count per class
def count_objects(data, by_class=False, allowed_classes=list(read_class_names(cfg.YOLO.CLASSES).values())):
    boxes, scores, classes, num_objects = data

    # create dictionary to hold count of objects
    counts = dict()

    # if by_class = True then count objects per class
    if by_class:
        class_names = read_class_names(cfg.YOLO.CLASSES)

        # loop through total number of objects found
        for i in range(num_objects):
            # grab class index and convert into corresponding class name
            class_index = int(classes[i])
            class_name = class_names[class_index]
            if class_name in allowed_classes:
                counts[class_name] = counts.get(class_name, 0) + 1
            else:
                continue

    # else count total objects found
    else:
        counts['total object'] = num_objects

    return counts


# function for cropping each detection and saving as new image
def crop_objects(img, data, path, allowed_classes):
    boxes, scores, classes, num_objects = data
    class_names = read_class_names(cfg.YOLO.CLASSES)
    # create dictionary to hold count of objects for image name
    counts = dict()
    for i in range(num_objects):
        # get count of class for part of image name
        class_index = int(classes[i])
        class_name = class_names[class_index]
        if class_name in allowed_classes:
            counts[class_name] = counts.get(class_name, 0) + 1
            # get box coords
            xmin, ymin, xmax, ymax = boxes[i]
            # crop detection from image (take an additional 5 pixels around all edges)
            cropped_img = img[int(ymin) - 5:int(ymax) + 5, int(xmin) - 5:int(xmax) + 5]
            # construct image name and join it to path for saving crop properly
            img_name = class_name + '_' + str(counts[class_name]) + '.png'
            img_path = os.path.join(path, img_name)
            # save image
            cv2.imwrite(img_path, cropped_img)
        else:
            continue


# function to run general Tesseract OCR on any detections
def ocr(img, data):
    boxes, scores, classes, num_objects = data
    class_names = read_class_names(cfg.YOLO.CLASSES)
    for i in range(num_objects):
        # get class name for detection
        class_index = int(classes[i])
        class_name = class_names[class_index]
        # separate coordinates from box
        xmin, ymin, xmax, ymax = boxes[i]
        # get the subimage that makes up the bounded region and take an additional 5 pixels on each side
        box = img[int(ymin) - 5:int(ymax) + 5, int(xmin) - 5:int(xmax) + 5]
        # grayscale region within bounding box
        gray = cv2.cvtColor(box, cv2.COLOR_RGB2GRAY)
        # threshold the image using Otsus method to preprocess for tesseract
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        # perform a median blur to smooth image slightly
        blur = cv2.medianBlur(thresh, 3)
        # resize image to double the original size as tesseract does better with certain text size
        blur = cv2.resize(blur, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        # run tesseract and convert image text to string
        try:
            text = pytesseract.image_to_string(blur, config='--psm 11 --oem 3')
            print("Class: {}, Text Extracted: {}".format(class_name, text))
        except:
            text = None


def is_close(p1, p2):
    """
    #================================================================
    # 1. Purpose : Calculate Euclidean Distance between two points
    #================================================================
    :param:
    p1, p2 = two points for calculating Euclidean Distance
    :return:
    dst = Euclidean Distance between two 2d points
    """
    dst = math.sqrt(p1 ** 2 + p2 ** 2)
    # =================================================================#
    return dst


def cvDrawBoxes(data, img):
    """
    :param:
    detections = total detections in one frame
    img = image from detect_image method of darknet
    :return:
    img with bbox
    """
    # ================================================================
    # 3.1 Purpose : Filter out Persons class from detections and get
    #           bounding box centroid for each person detection.
    # ================================================================
    boxes, scores, classes, num_objects = data
    class_names = read_class_names(cfg.YOLO.CLASSES)

    if num_objects.size > 0:  # At least 1 detection in the image and check detection presence in a frame
        centroid_dict = dict()  # Function creates a dictionary and calls it centroid_dict
        objectId = 0  # We initialize a variable called ObjectId and set it to 0
        for i in range(num_objects):  # In this if statement, we filter all the detections for persons only
            # grab class index and convert into corresponding class name
            class_index = int(classes[i])
            class_name = class_names[class_index]
            # Check for the only person name tag
            if class_name == 'person':
                xmin, ymin, xmax, ymax = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
                # calculate middle points x,y
                x = int(round((xmin + xmax) / 2))
                y = int(round((ymin + ymax) / 2))

                centroid_dict[objectId] = (x, y, xmin, ymin, xmax,
                                           ymax)  # Create dictionary of tuple with 'objectId' as the index center points and bbox
                objectId += 1  # Increment the index for each detection
                print('centroid', centroid_dict)
        # =================================================================#

        # =================================================================
        # 3.2 Purpose : Determine which person bbox are close to each other
        # =================================================================
        red_zone_list = []  # List containing which Object id is in under threshold distance condition.
        red_line_list = []
        for (id1, p1), (id2, p2) in combinations(centroid_dict.items(),
                                                 2):  # Get all the combinations of close detections, #List of multiple items - id1 1, points 2, 1,3
            dx, dy = p1[0] - p2[0], p1[1] - p2[1]  # Check the difference between centroid x: 0, y :1
            distance = is_close(dx, dy)
            print('distance', distance)  # Calculates the Euclidean distance
            if distance < 500:  # Set our social distance threshold - If they meet this condition then..
                print('DISTANCE TOO CLOSE')
                if id1 not in red_zone_list:
                    red_zone_list.append(id1)  # Add Id to a list
                    red_line_list.append(p1[0:2])  # Add points to the list
                if id2 not in red_zone_list:
                    red_zone_list.append(id2)  # Same for the second id
                    red_line_list.append(p2[0:2])

        for idx, box in centroid_dict.items():  # dict (1(key):red(value), 2 blue)  idx - key  box - value
            if idx in red_zone_list:  # if id is in red zone list
                cv2.rectangle(img, (box[2], box[3]), (box[4], box[5]), (255, 0, 0),
                              2)  # Create Red bounding boxes  #starting point, ending point size of 2
            else:
                cv2.rectangle(img, (box[2], box[3]), (box[4], box[5]), (0, 255, 0), 2)  # Create Green bounding boxes
            # =================================================================#

            # =================================================================
        # 3.3 Purpose : Display Risk Analytics and Show Risk Indicators
        # =================================================================
        text = "People at Risk: %s" % str(len(red_zone_list))  # Count People at Risk
        location = (10, 25)  # Set the location of the displayed text
        cv2.putText(img, text, location, cv2.FONT_HERSHEY_SIMPLEX, 1, (246, 86, 86), 2, cv2.LINE_AA)  # Display Text

        for check in range(0, len(red_line_list) - 1):  # Draw line between nearby bboxes iterate through redlist items
            start_point = red_line_list[check]
            end_point = red_line_list[check + 1]
            check_line_x = abs(end_point[0] - start_point[0])  # Calculate the line coordinates for x
            check_line_y = abs(end_point[1] - start_point[1])  # Calculate the line coordinates for y
            if (check_line_x < 75) and (
                    check_line_y < 25):  # If both are We check that the lines are below our threshold distance.
                cv2.line(img, start_point, end_point, (255, 0, 0), 2)  # Only above the threshold lines are displayed.
        # =================================================================#
    return img

