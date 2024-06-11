import streamlit as st
import pandas as pd
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import os
#import tensorflow as tf
#import tensorflow_hub as hub
import time ,sys
from streamlit_embedcode import github_gist
import urllib.request
import urllib
import moviepy.editor as moviepy
import cv2
import numpy as np
import time
import sys


def object_detection_image():
    st.title('SpeedBump and Pothole Detection for Images')
    st.subheader("""
    This object detection project takes in an image and outputs the image with bounding boxes created around the objects in the image
    """)
    d = st.markdown("""
    Please scroll down to see the processed image."""
    )
    file = st.file_uploader('Upload Image', type = ['jpg','png','jpeg'])
    if file!= None:
        img1 = Image.open(file)
        img2 = np.array(img1)

        st.image(img1, caption = "Uploaded Image")
        my_bar = st.progress(0)
        confThreshold =st.slider('Confidence', 0, 100, 50)
        nmsThreshold= st.slider('Threshold', 0, 100, 20)
        #classNames = []
        whT = 416
        # url = "https://raw.githubusercontent.com/zhoroh/ObjectDetection/master/labels/coconames.txt"
        # f = urllib.request.urlopen(url)
        # classNames = [line.decode('utf-8').strip() for  line in f]
        f = open(r'obj.names','r')
        lines = f.readlines()
        classNames = [line.strip() for line in lines]
        config_path = r'yolov4-custom.cfg'
        weights_path = r'yolov4-custom_final.weights'
        net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        def findObjects(outputs,img):
            hT, wT, cT = img2.shape
            bbox = []
            classIds = []
            confs = []
            for output in outputs:
                for det in output:
                    scores = det[5:]
                    classId = np.argmax(scores)
                    confidence = scores[classId]
                    if confidence > (confThreshold/100):
                        w,h = int(det[2]*wT) , int(det[3]*hT)
                        x,y = int((det[0]*wT)-w/2) , int((det[1]*hT)-h/2)
                        bbox.append([x,y,w,h])
                        classIds.append(classId)
                        confs.append(float(confidence))
        
            indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold/100, nmsThreshold/100)
            obj_list=[]
            confi_list =[]
            #drawing rectangle around object
            for i in indices:
                i = i
                box = bbox[i]
                x, y, w, h = box[0], box[1], box[2], box[3]
                # print(x,y,w,h)
                cv2.rectangle(img2, (x, y), (x+w,y+h), (240, 54 , 230), 2)
                #print(i,confs[i],classIds[i])
                obj_list.append(classNames[classIds[i]].upper())
                
                confi_list.append(int(confs[i]*100))
                cv2.putText(img2,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
                          (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (240, 0, 240), 2)
            df= pd.DataFrame(list(zip(obj_list,confi_list)),columns=['Object Name','Confidence'])
            if st.checkbox("Show Object's list" ):
                
                st.write(df)
            if st.checkbox("Show Confidence bar chart" ):
                st.subheader('Bar chart for confidence levels')
                
                st.bar_chart(df["Confidence"])
           
        blob = cv2.dnn.blobFromImage(img2, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
        net.setInput(blob)
        layersNames = net.getLayerNames()
        outputNames = [layersNames[i-1] for i in net.getUnconnectedOutLayers()]
        outputs = net.forward(outputNames)
        findObjects(outputs,img2)
    
        st.image(img2, caption='Processed Image.')
        
        cv2.waitKey(0)
        
        cv2.destroyAllWindows()
        my_bar.progress(100)



def object_detection_video():
    #object_detection_video.has_beenCalled = True
    #pass
    CONFIDENCE = 0.2
    SCORE_THRESHOLD = 0.0
    IOU_THRESHOLD = 0.5
    config_path = r'yolov4-custom.cfg'
    weights_path = r'yolov4-custom_final.weights'
    font_scale = 1
    thickness = 1
    # url = "https://raw.githubusercontent.com/zhoroh/ObjectDetection/master/labels/coconames.txt"
    # f = urllib.request.urlopen(url)
    # labels = [line.decode('utf-8').strip() for  line in f]
    f = open(r'obj.names','r')
    lines = f.readlines()
    labels = [line.strip() for line in lines]
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    st.title("SpeedBump and Pothole Detection for Videos")
    st.subheader("""
    This object detection project takes in a video and outputs the video with bounding boxes created around the objects in the video 
    """
    )
    uploaded_video = None
    uploaded_video = st.file_uploader("Upload Video", type = ['mp4','mpeg','mov'])
    if uploaded_video != None:
        
        vid = uploaded_video.name
        with open(vid, mode='wb') as f:
            f.write(uploaded_video.read()) # save video to disk

        st_video = open(vid,'rb')
        video_bytes = st_video.read()
        st.write("Uploaded Video")
        st.video(video_bytes)
        #video_file = 'street.mp4'
        cap = cv2.VideoCapture(vid)
        _, image = cap.read()
        h, w = image.shape[:2]
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        #out = cv2.VideoWriter(output_name, cv2.VideoWriter_fourcc#(*'avc3'), fps, insize)

        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        out = cv2.VideoWriter("detected_video.mp4", fourcc, 20.0, (w, h))
        count = 0

        while True:
            _, image = cap.read()
            if _ != False:
                h, w = image.shape[:2]
                blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
                net.setInput(blob)
                start = time.perf_counter()
                layer_outputs = net.forward(ln)
                time_took = time.perf_counter() - start
                per =count/total
                count +=1
                d = st.markdown(f"""
                Video in process. Please wait. {per*100:.2f}%"""
                )
                e = st.progress(per)
                print(f"Time took: {count}", time_took)
                boxes, confidences, class_ids = [], [], []

                # loop over each of the layer outputs
                for output in layer_outputs:
                    # loop over each of the object detections
                    for detection in output:
                        # extract the class id (label) and confidence (as a probability) of
                        # the current object detection
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        # discard weak predictions by ensuring the detected
                        # probability is greater than the minimum probability
                        if confidence > CONFIDENCE:
                            # scale the bounding box coordinates back relative to the
                            # size of the image, keeping in mind that YOLO actually
                            # returns the center (x, y)-coordinates of the bounding
                            # box followed by the boxes' width and height
                            box = detection[:4] * np.array([w, h, w, h])
                            (centerX, centerY, width, height) = box.astype("int")

                            # use the center (x, y)-coordinates to derive the top and
                            # and left corner of the bounding box
                            x = int(centerX - (width / 2))
                            y = int(centerY - (height / 2))

                            # update our list of bounding box coordinates, confidences,
                            # and class IDs
                            boxes.append([x, y, int(width), int(height)])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)

                # perform the non maximum suppression given the scores defined before
                idxs = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, IOU_THRESHOLD)

                font_scale = 0.6
                thickness = 1

                # ensure at least one detection exists
                if len(idxs) > 0:
                    # loop over the indexes we are keeping
                    for i in idxs.flatten():
                        # extract the bounding box coordinates
                        x, y = boxes[i][0], boxes[i][1]
                        w, h = boxes[i][2], boxes[i][3]
                        # draw a bounding box rectangle and label on the image
                        color = [int(c) for c in colors[class_ids[i]]]
                        cv2.rectangle(image, (x, y), (x + w, y + h), color=color, thickness=thickness)
                        text = f"{labels[class_ids[i]]}: {confidences[i]:.2f}"
                        # calculate text width & height to draw the transparent boxes as background of the text
                        (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
                        text_offset_x = x
                        text_offset_y = y - 5
                        box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
                        overlay = image.copy()
                        cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)
                        # add opacity (transparency to the box)
                        image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
                        # now put the text (label: confidence %)
                        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=font_scale, color=(0, 0, 0), thickness=thickness)
                
                out.write(image)
                cv2.imshow("image", image)
                d.empty()
                e.empty() 
                if ord("q") == cv2.waitKey(1):
                    break
            else:
                break
        d.empty()
        
        # return "detected_video.mp4"
            
        cap.release()
        cv2.destroyAllWindows()

def object_detection_cam():
    #object_detection_video.has_beenCalled = True
    #pass
    CONFIDENCE = 0.3
    SCORE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.5
    config_path = r'yolov4-custom.cfg'
    weights_path = r'yolov4-custom_final.weights'
    font_scale = 1
    thickness = 1
    # url = "https://raw.githubusercontent.com/zhoroh/ObjectDetection/master/labels/coconames.txt"
    # f = urllib.request.urlopen(url)
    # labels = [line.decode('utf-8').strip() for  line in f]
    f = open(r'obj.names','r')
    lines = f.readlines()
    labels = [line.strip() for line in lines]
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    

    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    st.title("SpeedBump and Pothole Detection for WebCam")
    st.subheader("""
    Real-time detection 
    """
    )
    st.write("Tick 'RUN' to open camera. Untick to close camera.")

    run = st.checkbox('Run')
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)
    count = 0
    _, image = camera.read()
    h, w = image.shape[:2]
    h1, w1 = image.shape[:2]

    fps = 0
    prev_time = 0
    curr_time = 0
    st1, st2, st3 = st.columns(3)
    with st1:
        st.markdown("## Height")
        st1_text = st.markdown(f"{h1}")
    with st2:
        st.markdown("## Width")
        st2_text = st.markdown(f"{w1}")
    with st3:
        st.markdown("## FPS")
        st3_text = st.markdown(f"{fps}")

    while run:
        _, frame = camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if _ != False:
                h, w = frame.shape[:2]
                blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
                net.setInput(blob)
                start = time.perf_counter()
                layer_outputs = net.forward(ln)
                time_took = time.perf_counter() - start
                count +=1
                print(f"Time took: {count}", time_took)
                boxes, confidences, class_ids = [], [], []

                # loop over each of the layer outputs
                for output in layer_outputs:
                    # loop over each of the object detections
                    for detection in output:
                        # extract the class id (label) and confidence (as a probability) of
                        # the current object detection
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        # discard weak predictions by ensuring the detected
                        # probability is greater than the minimum probability
                        if confidence > CONFIDENCE:
                            # scale the bounding box coordinates back relative to the
                            # size of the image, keeping in mind that YOLO actually
                            # returns the center (x, y)-coordinates of the bounding
                            # box followed by the boxes' width and height
                            box = detection[:4] * np.array([w, h, w, h])
                            (centerX, centerY, width, height) = box.astype("int")

                            # use the center (x, y)-coordinates to derive the top and
                            # and left corner of the bounding box
                            x = int(centerX - (width / 2))
                            y = int(centerY - (height / 2))

                            # update our list of bounding box coordinates, confidences,
                            # and class IDs
                            boxes.append([x, y, int(width), int(height)])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)

                # perform the non maximum suppression given the scores defined before
                idxs = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, IOU_THRESHOLD)

                font_scale = 0.6
                thickness = 1

                # ensure at least one detection exists
                if len(idxs) > 0:
                    # loop over the indexes we are keeping
                    for i in idxs.flatten():
                        # extract the bounding box coordinates
                        x, y = boxes[i][0], boxes[i][1]
                        w, h = boxes[i][2], boxes[i][3]
                        # draw a bounding box rectangle and label on the image
                        color = [int(c) for c in colors[class_ids[i]]]
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color=color, thickness=thickness)
                        text = f"{labels[class_ids[i]]}: {confidences[i]:.2f}"
                        # calculate text width & height to draw the transparent boxes as background of the text
                        (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
                        text_offset_x = x
                        text_offset_y = y - 5
                        box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
                        overlay = frame.copy()
                        cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)
                        # add opacity (transparency to the box)
                        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
                        # now put the text (label: confidence %)
                        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=font_scale, color=(0, 0, 0), thickness=thickness)
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        st1_text.markdown(f"**{h}**")
        st2_text.markdown(f"**{w}**")
        st3_text.markdown(f"**{fps:.2f}**")
        FRAME_WINDOW.image(frame)
    else:
        st.write('Stopped')



def main():
    new_title = '<p style="font-size: 42px;">Welcome to SpeedBump and Pothole Detection App!</p>'
    read_me_0 = st.markdown(new_title, unsafe_allow_html=True)

    read_me = st.markdown("""
    This project was built using Streamlit and OpenCV 
    to demonstrate YOLO speedbump and pothole detection in images. videos(pre-recorded)
    and webcam."""
    )
    choice  = st.sidebar.selectbox("MODE",("About","Image","Video","WebCam"))
    #["Show Instruction","Landmark identification","Show the #source code", "About"]
    
    if choice == "Image":
        #st.subheader("Object Detection")
        read_me_0.empty()
        read_me.empty()
        #st.title('Object Detection')
        object_detection_image()
    elif choice == "Video":
        if os.path.exists('myvideo.mp4'):
            os.unlink('myvideo.mp4')
        if os.path.exists('detected_video.mp4'):
            os.unlink('detected_video.mp4')
        read_me_0.empty()
        read_me.empty()
        #object_detection_video.has_beenCalled = False
        object_detection_video()
        #if object_detection_video.has_beenCalled:
        try:
            clip = moviepy.VideoFileClip('detected_video.mp4')
            clip.write_videofile("myvideo.mp4")
            st_video = open('myvideo.mp4','rb')
            video_bytes = st_video.read()
            st.write("Detected Video") 
            st.video(video_bytes)
        except OSError:
            ''
        

    elif choice == "WebCam":
        read_me_0.empty()
        read_me.empty()
        #object_detection_video.has_beenCalled = False
        object_detection_cam()

    elif choice == "About":
        print()
        

if __name__ == '__main__':
		main()	