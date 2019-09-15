import cv2
import numpy as np


def alpha_blend(foreground, background, alpha):
    # foreground = cv2.imread("puppets.png")
    # background = cv2.imread("ocean.png")
    # alpha = cv2.imread("puppets_alpha.png")

    # Convert uint8 to float
    # print(type(foreground.dtype))
    foreground = foreground.astype(float)
    background = background.astype(float)
    print(alpha.shape)
    alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR)

    # Normalize the alpha mask to keep intensity between 0 and 1
    alpha = alpha.astype(float) / 255
    # print(alpha.shape)
    # Multiply the foreground with the alpha matte
    foreground = cv2.multiply(alpha, foreground)

    # Multiply the background with ( 1 - alpha )
    background = cv2.multiply(1.0 - alpha, background)

    # Add the masked foreground and background.
    outImage = cv2.add(foreground, background)
    return outImage / 255


def denoise(frame):
    frame = cv2.medianBlur(frame, 5)
    frame = cv2.GaussianBlur(frame, (5, 5), 0)

    return frame


frame1 = cv2.imread('frame1.png')
cap = cv2.VideoCapture('video1.avi')
ret, frame = cap.read()

cap2 = cv2.VideoCapture('video2.avi')
ret2, frame2 = cap2.read()

cap3 = cv2.VideoCapture('video3.avi')
ret3, frame3 = cap3.read()

frame_array = []
frame1 = cv2.imread("frame1.png")

fps = 30.0
size = frame.shape
size = (size[1], size[0])
print(size)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, fps, size)

if not cap.isOpened:
    print('Unable to open: ')
    exit(0)
## [capture]
backSub = cv2.createBackgroundSubtractorMOG2()
kernel = np.ones((8, 8), np.uint8)

while True:
    if (ret == None or ret2 == None or ret3 == None):
        break
    ret, frame = cap.read()
    ret2, frame2 = cap2.read()
    ret3, frame3 = cap3.read()
    if (frame is None or frame2 is None or frame3 is None):
        break

    ## [apply]
    # update the background model
    fgMask = backSub.apply(denoise(frame))
    ## [apply]
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)
    ret, fgMask = cv2.threshold(fgMask, 80, 255, cv2.THRESH_BINARY)
    fgMask = cv2.dilate(fgMask, None, iterations=2)

    fgMask2 = backSub.apply(denoise(frame2))
    ## [apply]
    fgMask2 = cv2.morphologyEx(fgMask2, cv2.MORPH_OPEN, kernel)
    ret2, fgMask2 = cv2.threshold(fgMask2, 80, 255, cv2.THRESH_BINARY)
    fgMask2 = cv2.dilate(fgMask2, None, iterations=2)

    fgMask3 = backSub.apply(denoise(frame3))
    ## [apply]
    fgMask3 = cv2.morphologyEx(fgMask3, cv2.MORPH_OPEN, kernel)
    ret, fgMask3 = cv2.threshold(fgMask3, 80, 255, cv2.THRESH_BINARY)
    fgMask3 = cv2.dilate(fgMask3, None, iterations=2)
    print(fgMask.dtype)
    image = alpha_blend(frame, frame1, fgMask)
    image = alpha_blend(frame2, image * 255, fgMask2)
    image = alpha_blend(frame3, image * 255, fgMask3)

    data = image * 255
    img = np.uint8(data)
    out.write(img)

    ## [display_frame_number]
    # get the frame number and write it on the current frame
    cv2.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
    cv2.putText(frame, str(cap.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    ## [display_frame_number]

    ## [show]
    # show the current frame and the fg masks
    # cv2.imshow('Frame', frame)
    cv2.imshow('FG Mask', fgMask)
    ## [show]

    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
out.release()
cap.release()
cap2.release()
cap3.release()
