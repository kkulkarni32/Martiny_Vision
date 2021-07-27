"""trt_yolo.py

This script demonstrates how to do real-time object detection with
TensorRT optimized YOLO engine.
"""


import os
import time
import argparse
import numpy as np
import boto3
import datetime

import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver

from utils.yolo_classes import get_cls_dict
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import get_input_shape, TrtYOLO

from Shadow_Detection import shadow_detect


WINDOW_NAME = 'TrtYOLODemo'
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('People_Count')
objects = [0,1,15,16,25] #person,bike,cat,dog,umbrella
store_data = True
store_aws = True

# Args to pass
# store_data, apply pink mask, images, freq_images, videos, freq_videos, aws related stuff like cred, tables etc
def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with TensorRT optimized '
            'YOLO model on Jetson')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument(
        '-c', '--category_num', type=int, default=80,
        help='number of object categories [80]')
    parser.add_argument(
        '-m', '--model', type=str, default="yolov4-416",
        help=('[yolov3|yolov3-tiny|yolov3-spp|yolov4|yolov4-tiny]-'
              '[{dimension}], where dimension could be a single '
              'number (e.g. 288, 416, 608) or WxH (e.g. 416x256)'))
    parser.add_argument(
        '-l', '--letter_box', action='store_true',
        help='inference with letterboxed image [False]')
    args = parser.parse_args()
    return args

def loop_and_detect(cam, trt_yolo, conf_th, vis):
    """Continuously capture images from camera and do object detection.

    # Arguments
      cam: the camera instance (video source).
      trt_yolo: the TRT YOLO object detector instance.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
    """

    

    full_scrn = False
    fps = 0.0
    tic = time.time()
    start= time.time()
    shadow_flag = True
    vid_stop = True
    store_img = 0
    vid_count = 0
    vid_time = [10,10,10,15,15,15,15,20,20,20,30] #Variable for video time interval, which are chosen at random
    frames = 0
    vid_flag = False
    while True:

        people_count = 0
        shadow_count = 0
        bike_count = 0
        pet_count = 0
        umbrella_count = 0
        
        

        timer_1 = time.time()
        frames+=1

        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break
        img = cam.read()
        img_cp = img.copy()
        if img is None:
            print("No image found")
            break
        boxes, confs, clss = trt_yolo.detect(img, conf_th)
        img = vis.draw_bboxes(img, boxes, confs, clss)
        img = show_fps(img, fps)

        clss_list = clss.tolist()
        boxes_list = boxes.tolist()

        if (int(timer_1-start)%900==0): # Making sure that shadow image is loaded every 15 mintues. 
            print("Loading shadow data")
            shadow = cv2.imread("./Shadow_Map/Shadow.jpg")
            shadow = cv2.resize(shadow, (640,480))
            

            # This is to apply white mask for all the gray pixel points which are caused due to python3 conversion 
            shadow[shadow>10]=255

            # Applying Pink mask
            shadow[(shadow==[255,255,255]).all(axis=2)] = [193,182,255]

            
            # shadow_flag =False
            cv2.imwrite("./Shadow_Images/Shadow.jpg", img_cp)
            time.sleep(1) # Making sure to not load shadow map again within a second

        #Blending both the images
        img = cv2.addWeighted(img, 0.7,shadow, 0.4, 0)
        
        
    



        for index, value in enumerate(clss_list):
            if value ==0:
                people_count+=1
                # print(boxes_list[index])
            

                pt1 = boxes_list[index][0:2]
                pt2 = boxes_list[index][2:4]

                shade = shadow_detect(img_cp, shadow, pt1, pt2, 0.5)

                if shade:
                    shadow_count+=1

                confs_ped = confs[index]
                box_ped = boxes[index]
                stack = np.hstack((box_ped,confs_ped))
                # print(stack)

            elif value==1:
                bike_count+=1

            elif value==15 or value==16:
                pet_count+=1
            
            elif value==45:
                umbrella_count+=1

        # print("Number of people", people_count)
        # print("Number in shadow", shadow_count)
        img = cv2.putText(img, "No. of people: {0}".format(people_count), (550,20), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0,0,0), 1, cv2.LINE_AA)
        img = cv2.putText(img, "People in shade: {0}".format(shadow_count), (550,35), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0,0,0), 1, cv2.LINE_AA)
        cv2.imshow(WINDOW_NAME, img)

        if store_aws and people_count>0:
            timestamp = datetime.datetime.now().replace(microsecond=0).isoformat()
            try:
                table.put_item(   
                Item={
                'Time': timestamp,
                'No of people': people_count,
                'No of people in shade': shadow_count,
                'Bikes' : bike_count,
                'Umbrellas' : umbrella_count,
                'No of pets' : pet_count

                        }
                )

            except:
                print("unable to insert data in AWS table")

            if frames %30==0:
            # This checks if there is pedestrian every 10 seconds, and stores if there is.
                # timestamp = datetime.datetime.now().replace(microsecond=0).isoformat()
                cv2.imwrite("Test/Images/Image_" + str(timestamp) + ".jpg", img_cp) # use img for images with bounding boxes, img_cp without bounding boxes
                np.save("Test/Images/Shadow_" + str(timestamp) + ".npy", shadow)
                np.save("Test/Images/bbox_" + str(timestamp) + ".npy", stack)
                
                store_img+=1



                if(store_data and store_img%10==0):
                    # This copies data from folder to aws and then delete them from local system
                    print("sending data to AWS")
                    os.system("aws s3 cp ./Test/Images/ s3://martinyvision/ --recursive")
                    os.system("rm ./Test/Images/*")

            
                
                if store_data and vid_stop and frames%60==0:
                    print("Storing videos")
                    vid_flag = True
                    vid_stop = False
                    vid_start = time.time()
                    choice = np.random.choice(vid_time)
                    vid = cv2.VideoWriter("Test/Videos/Video_" + str(datetime.datetime.now().replace(microsecond=0).isoformat()) + ".mp4", 
                                cv2.VideoWriter_fourcc(*'MP4V'),
                                12, (img.shape[1], img.shape[0]))

            timer_2=time.time()

            # This will make sure the next frame is after 1 second, hence calculating and storing data every second
            if(timer_2-timer_1<1):
                time.sleep(1-(timer_2-timer_1))


        
        

        if vid_flag:
            
            # This logic starts to store video, untill the length of the video equals to one of the random choices
            vid.write(img)
            vid_end = time.time()

            if int(vid_end-vid_start+1)%choice==0: #not sure why I added +1?
                vid_count+=1
                vid.release()
                vid_stop = True
                vid_flag = False

                if(store_data and vid_count%10==0):
                    # print("video command ###################################")
                    os.system("aws s3 cp ./Test/Videos/ s3://martinyvision/ --recursive")
                    os.system("rm ./Test/Videos/*")



        toc = time.time()
        curr_fps = 1.0 / (toc - tic)
        # calculate an exponentially decaying average of fps number
        fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
        tic = toc
        key = cv2.waitKey(1)

        
        


        if key == 27:  # ESC key: quit program
            if vid_stop==False:
                vid.release()
            break

        elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            full_scrn = not full_scrn
            set_display(WINDOW_NAME, full_scrn)

        


def main():
    args = parse_args()
    if args.category_num <= 0:
        raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
    if not os.path.isfile('yolo/%s.trt' % args.model):
        raise SystemExit('ERROR: file (yolo/%s.trt) not found!' % args.model)

    try:
        cam = Camera(args)
        
        if not cam.isOpened():
            raise SystemExit('ERROR: failed to open camera!')

        cls_dict = get_cls_dict(args.category_num)
        vis = BBoxVisualization(cls_dict)
        h, w = get_input_shape(args.model)
        trt_yolo = TrtYOLO(args.model, (h, w), args.category_num, args.letter_box)

        open_window(
            WINDOW_NAME, 'Camera TensorRT YOLO Demo',
            cam.img_width, cam.img_height)

        try:
            loop_and_detect(cam, trt_yolo, conf_th=0.3, vis=vis)
        except KeyboardInterrupt:
            cam.release()
            cv2.destroyAllWindows()

    except KeyboardInterrupt:
            cam.release()
            cv2.destroyAllWindows()

        
    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
