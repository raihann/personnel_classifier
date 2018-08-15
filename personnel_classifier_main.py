#--------------------------------------------------------------
# Author:     Rai Hann
#             Chief Architect, Glocol Networks
# Created:    July 12, 2018
# Name:       Computer Vision Personnel Classifier
# Platform:   iWave q7 RZ/G1M Development Board
# License:    MIT (see license file)
#--------------------------------------------------------------

import os
import cv2
import time
import urllib
import argparse
import multiprocessing
import numpy as np
import tensorflow as tf

from utils.app_utils import FPS, WebcamVideoStream
from multiprocessing import Queue, Pool
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from per_classifier_2  import box_filter
from per_classifier_2 import vest_classifier

# Globols (will be dealt with in a later refactored source revision)

CWD_PATH = os.getcwd()
count = 0
p_count = 0

# Path to frozen detection graph. This is the actual model that is used for the object detection.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
PATH_TO_CKPT = os.path.join(CWD_PATH, 'object_detection', MODEL_NAME, 'frozen_inference_graph.pb')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(CWD_PATH, 'object_detection', 'data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def filter_boxes(min_score, boxes, scores, classes, categories):
    """Return boxes with a confidence >= `min_score`"""
    n = len(classes)
    idxs = []
    for i in range(n):
        if classes[i] in categories and scores[i] >= min_score:
            idxs.append(i)
    
    filtered_boxes = boxes[idxs, ...]
    filtered_scores = scores[idxs, ...]
    filtered_classes = classes[idxs, ...]
    return filtered_boxes, filtered_scores, filtered_classes

def detect_objects(image_np, sess, detection_graph):
    # Initiated object detection (persons)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]

    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    print scores
    print classes
    print boxes
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)

#    filtered_boxes, filtered_scores, filtered_classes = filter_boxes(50.0, np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes), categories)
    
    global count
    count = 0 
    global red_rects
    red_detect = [] 

    s_classes = np.squeeze(classes)
    s_scores = np.squeeze(scores)
    s_boxes = np.squeeze(boxes)
    for i in range(len(s_classes)):
        if s_classes[i] == 1.0 and s_scores[i] > 0.30:
            count = count + 1
    print 'people count = ' + str(count)
    print s_classes


    filtered_boxes = box_filter(s_boxes, s_scores, s_classes, 0.2, 1)
    print "filtered boxes"
    print filtered_boxes

    # get image shape
    height, width, _ = image_np.shape

    # using boxed people detections, classify person based on vest color if any
    display_image, green_rects, red_rects = vest_classifier(image_np, filtered_boxes, width, height)

    print "green_rects = " + str(green_rects)
    print "red_rects   = " + str(red_rects)
    # Visualization of the results of a detection.
    """vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=2) """
    return display_image


def worker(input_q, output_q, myqueue):

    # creates worker process for parallel detections
    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    fps = FPS().start()
    while True:
        #fps.update()
        frame = input_q.get()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output_q.put(detect_objects(frame_rgb, sess, detection_graph))
        myqueue.put(count)

    fps.stop()
    sess.close()

def printCount():
    print str(count)



def cam_stream(url_source):

    # capture video frames from remote VLC stream

    stream=urllib.urlopen(url_source)
    bytes=''
    while True:
        bytes+=stream.read(1024)
        a = bytes.find('\xff\xd8')
        b = bytes.find('\xff\xd9')
        if a!=-1 and b!=-1:
            jpg = bytes[a:b+2]
            bytes= bytes[b+2:]
            frame = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8),1)
            return frame



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', '--source', dest='video_source', type=int,
                        default=0, help='Device index of the camera.')
    parser.add_argument('-wd', '--width', dest='width', type=int,
                        default=480, help='Width of the frames in the video stream.')
    parser.add_argument('-ht', '--height', dest='height', type=int,
                        default=360, help='Height of the frames in the video stream.')
    parser.add_argument('-num-w', '--num-workers', dest='num_workers', type=int,
                        default=2, help='Number of workers.')
    parser.add_argument('-q-size', '--queue-size', dest='queue_size', type=int,
                        default=5, help='Size of the queue.')
    args = parser.parse_args()

    logger = multiprocessing.log_to_stderr()
    logger.setLevel(multiprocessing.SUBDEBUG)


    input_q = Queue(maxsize=args.queue_size)
    output_q = Queue(maxsize=args.queue_size)
    myqueue = Queue(4)

    pool = Pool(args.num_workers, worker, (input_q, output_q, myqueue))
    
    # Will stream from local webcam (currently needs debugging)
    #video_capture = WebcamVideoStream(src=args.video_source,
    #                                  width=args.width,
    #                                  height=args.height).start()

    # stream from CMOS camera (poor color stability prevents proper color segmentation)
    #video_capture = cv2.VideoCapture(0)

    # grab frames from remote VLC source (edit IP address to accomodate your LAN)
    source_url = "http://192.168.1.27:8080"
    frame = cam_stream(source_url)
    
    #frame = video_capture.read()    # read frame to get shape
    frame_height, frame_width, _ =  frame.shape
    out = cv2.VideoWriter('outvid.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),20 ,(frame_width, frame_height))

    b_mask = np.zeros((frame_height, frame_width, 3), np.uint8)
    b_mask[:] = (35, 0, 20)


    fps = FPS().start()

    while True:  # fps._numFrames < 120
        frame = cam_stream(source_url)
        #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
        #frame = cv2.subtract(frame, b_mask)
        #frame = cv2.cvtColor(frame, cv2.COLOR_LAB2RGB)
        input_q.put(frame)
        p_count = myqueue.get()

        t = time.time()

        output_rgb = cv2.cvtColor(output_q.get(), cv2.COLOR_RGB2BGR)
        #cv2.rectangle(output_rgb, (220, 10), (365, 30), (255, 0, 0), -1)
        #cv2.putText(output_rgb, "count = "+str(p_count), (230, 25), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
        cv2.imshow('Video', output_rgb)
        out.write(output_rgb)
        fps.update()

        print('[INFO] elapsed time: {:.2f}'.format(time.time() - t))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fps.stop()
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

    pool.terminate()
    video_capture.stop()
    cv2.destroyAllWindows()
