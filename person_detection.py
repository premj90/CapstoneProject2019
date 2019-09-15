# os.chdir("/Users/kashish/Downloads/yolov3")
import cv2
import numpy as np

import Summarise_clips
from yolov3 import yolo_opencv

print("ss")
# os.system("ls")
# Path  = "/Users/kashish/Desktop/s1.png"
config = "/Users/kashish/darknet/cfg/yolov3.cfg"
weights = "/Users/kashish/darknet/yolov3.weights"
classes = "/Users/kashish/Downloads/yolov3/yolov3.txt"

cap = cv2.VideoCapture('/Users/kashish/Downloads/new_croped.mp4')
ret, frame = cap.read()

flag_persons = False

if ret is True:

    run = True
else:
    run = False
skip_frames = 0
persons = 0

frames_clip = []
frames_time = []
output_list = []
output_time = []
buffer_frames = []
buffer_time = []

fps = cap.get(cv2.CAP_PROP_FPS)
timestamps = [cap.get(cv2.CAP_PROP_POS_MSEC)]
calc_timestamps = [0.0]

while (run):
    # Read a frame from the camera
    ret, frame = cap.read()

    # If the frame was properly read.
    if ret is True:

        ts = cap.get(cv2.CAP_PROP_POS_MSEC)
        cal_ts = calc_timestamps[-1] + 1000 / fps

        if (len(buffer_frames) < 30):
            buffer_frames.append(frame)
            buffer_time.append(ts)
        else:
            del buffer_frames[0]
            del buffer_time[0]
            buffer_frames.append(frame)
            buffer_time.append(ts)

        # print(skip_frames)

        if (skip_frames > 15):

            persons = yolo_opencv.detect_persons(frame, config, weights, classes)
            # print(persons)
            skip_frames = 0

            if (persons > 0):
                flag_persons = True
                print(ts)

                if (len(frames_clip) < 1):
                    frames_clip = frames_clip + buffer_frames
                    frames_time = frames_time + buffer_time
            else:
                flag_persons = False

        if flag_persons:
            millis = int(ts)
            seconds = int((millis / 1000) % 60)
            minutes = int((millis / (1000 * 60)) % 60)
            hours = int((millis / (1000 * 60 * 60)) % 24)
            time = str(hours) + ":" + str(minutes) + ":" + str(seconds)
            print(time)
            frames_time.append(ts)
            frames_clip.append(frame)
        else:

            if (len(frames_clip) > 10):
                output_time.append(frames_time)
                output_list.append(frames_clip)

            frames_clip = []
            frames_time = []

        # cv2.imshow("video",frame)
        skip_frames += 1

        key = cv2.waitKey(10) & 0xFF
    else:
        break

    if key == 27:
        break

print("Number of Clips ", len(output_list))

frame1 = cv2.imread('frame1.png')
summary_frames = Summarise_clips.summarise_clips(output_list[0:], output_time, frame1)

pathOut = "summary2.avi"
fps = 30.0
size = summary_frames[0].shape
size = (size[1], size[0])
print(size)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(pathOut, fourcc, fps, size)
frame_array = summary_frames
print(len(frame_array))
for i in range(len(frame_array)):
    # writing to a image array
    # cv2.imshow("frames", frame_array[i])
    # cv2.waitKey(0)
    print(i)
    f = frame_array[i] * 255
    f = np.uint8(f)

    out.write(f)
out.release()

"""

video_number = 0
for clip in output_list:
    video_number+=1

    pathOut = "video"+str(video_number)+".avi"
    fps = 30.0
    size = clip[0].shape
    size = (size[1], size[0])
    print(size)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(pathOut, fourcc, fps, size)
    frame_array = clip
    print(len(frame_array))
    for i in range(len(frame_array)):
        # writing to a image array
        #cv2.imshow("frames", frame_array[i])
        #cv2.waitKey(0)
        out.write(frame_array[i])
    out.release()
"""
cap.release()
