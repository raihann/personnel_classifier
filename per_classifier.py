import cv2
import numpy as np

def create_mask(source_image, lower_bound, upper_bound):

    # convert image frame to HSV color space
    hsv_img = cv2.cvtColor(source_image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
    # filter to remove noise from the mask
    kernel_open = np.ones((5, 5))
    kernel_close = np.ones((20, 20))
    mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    mask_close = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel_close)
    return mask

def box_filter(boxes, scores, classes, min_score, target_class):

    # filter boxes array based on minimum score and target class
    filtered_boxes = np.empty((0,4), float)
    for x,y,z in zip(boxes, scores, classes):
        if y > min_score and z == target_class:
            filtered_boxes = np.append(filtered_boxes, [x], axis=0)
    return filtered_boxes

def vest_classifier(display_image, boxes, width, height):
               
    # color segmentation and detection
    height, width, _ = display_image.shape
    red_detect = [False]*40
    green_detect = [False]*40
    red_lower_bound = np.array([0, 200, 32])
    red_upper_bound = np.array([20, 255, 255])
    green_lower_bound = np.array([25, 128, 70])
    green_upper_bound = np.array([40, 255, 255])
    red_mask = create_mask(display_image, red_lower_bound, red_upper_bound)
    green_mask = create_mask(display_image, green_lower_bound, green_upper_bound)
    _, red_conts, h = cv2.findContours(red_mask.copy(), cv2.RETR_EXTERNAL, \
    cv2.CHAIN_APPROX_SIMPLE)
    _, green_conts, h = cv2.findContours(green_mask.copy(), cv2.RETR_EXTERNAL, \
    cv2.CHAIN_APPROX_SIMPLE)
    red_filtered_conts = [c for c in red_conts if cv2.contourArea(c) > 256.0]
    green_filtered_conts = [c for c in green_conts if cv2.contourArea(c) > 256.0]
    red_rects = [cv2.boundingRect(f) for f in red_filtered_conts]
    green_rects = [cv2.boundingRect(f) for f in green_filtered_conts]
                
    for i,r in enumerate(red_rects):
        for j,b in enumerate(boxes):
            print "b[0] = " + str(b[0])
            box_left = int(b[1]*width)
            box_top = int(b[0]*height)
            box_right = int(b[3]*width)
            box_bottom = int(b[2]*height)
            print(b[0], b[1], b[2], b[3], " ", r[0], r[1], r[0]+r[2], r[1]+r[3])
            if ((b[1]*width < r[0]) and (b[0]*height < r[1]) and \
            (b[3]*width > r[0]+r[2]) and \
            (b[2]*height > r[1]+r[3])):
                cv2.rectangle(display_image, (int(b[1]*width), int(b[0]*height)), 
                (int(b[3]*width), int(b[2]*height)), (0,0,255), 2)
                # assign appropriate label
                scale = 1
                red_detect[j] = True
                label_text = "supervisor"
                label_background_color = (0, 0, 100)
                label_text_color = (255, 255, 255)  # white text

                label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                label_left = box_left
                label_top = box_top - label_size[1]
                if (label_top < 1):
                    label_top = 1
                    label_right = label_left + label_size[0]
                    label_bottom = label_top + label_size[1]
                    cv2.rectangle(display_image, (label_left - 1, label_top - 1), (label_right + 1,
                    label_bottom + 1), label_background_color, -1)

                    # label text above the box
                    cv2.putText(display_image, label_text, \
                    (label_left, label_bottom), cv2.FONT_HERSHEY_SIMPLEX, \
                    0.5, label_text_color, 1)

                
    for i,r in enumerate(green_rects):
        for j,b in enumerate(boxes):
            box_left = int(b[1]*width)
            box_top = int(b[0]*height)
            box_right = int(b[3]*width)
            box_bottom = int(b[2]*height)
            label_text = "unauthorized person"

            print(b[0], b[1], b[2], b[3], " ", r[0], r[1], r[0]+r[2], r[1]+r[3])
            if (b[1]*width < r[0]) and (b[0]*height < r[1]) and \
            (b[3]*width > r[0]+r[2]) and \
            (b[2]*height > r[1]+r[3]):
                label_text = "worker"
                cv2.rectangle(display_image, (int(b[1]*width), int(b[2]*height)),
                (int(b[3]*width), int(b[2]*height)), (0,255,0), 2)

                # assign appropriate label
                scale = 1
                green_detect[j] = True
                label_background_color = (0, 100, 0)
                label_text_color = (255, 255, 255)  # white text

                label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                label_left = box_left
                label_top = box_top - label_size[1]
                if (label_top < 1):
                    label_top = 1
                label_right = label_left + label_size[0]
                label_bottom = label_top + label_size[1]
                cv2.rectangle(display_image, (label_left - 1, label_top - 1), \
                (label_right + 1, label_bottom + 1), label_background_color, -1)

                # label text above the box
                cv2.putText(display_image, label_text, (label_left, label_bottom), \
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_text_color, 1)


    label_text = "unauthorized_person"
    for j,b in enumerate(boxes):
        if not (red_detect[j] or green_detect[j]):
            box_left = int(b[1]*width)
            box_top = int(b[0]*height)
            box_right = int(b[3]*width)
            box_bottom = int(b[2]*height)

            cv2.rectangle(display_image, (box_left, box_top), (box_right, box_bottom), \
            (255, 255, 100), 2)
            # assign appropriate label
            scale = 1
            label_background_color = (100, 0, 0)
            label_text_color = (255, 255, 255)  # white text

            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            label_left = box_left
            label_top = box_top - label_size[1]
            if (label_top < 1):
                label_top = 1
            label_right = label_left + label_size[0]
            label_bottom = label_top + label_size[1]
            cv2.rectangle(display_image, (label_left - 1, label_top - 1), \
            (label_right + 1, label_bottom + 1), label_background_color, -1)

            # label text above the box
            cv2.putText(display_image, label_text, (label_left, label_bottom), \
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_text_color, 1)

            print(red_rects)
                
            #cv2.drawContours(display_image, filtered_conts, -1, (0, 0, 255),3)
            #cv2.imshow("red mask", red_mask)
            #cv2.imshow("green_mask", green_mask)
              
    return display_image, red_mask, green_mask        

