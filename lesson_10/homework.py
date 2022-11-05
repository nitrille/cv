'''
## Homework 10

In this homework, you are going to use and compare two different trackers (of your liking) and compare the results.

### Step 1
Decide what video you are going to use for this homework, select an object and generate the template. You can use any video you want (your own, from Youtube, etc.)
and track any object you want (e.g. a car, a pedestrian, etc.).

### Step 2
Initialize a tracker (e.g. KCF).

### Step 3
Run the tracker on the video and the selected object. Run the tracker for around 10-15 frames.

### Step 4
For each frame, print the bounding box on the image and save it.

### Step 5
Select a different tracker (e.g. CSRT) and repeat steps 2, 3 and 4.

### Step 6
Compare the results:
* Do you see any differences? If so, what are they?
* Does one tracker perform better than the other? In what way?
'''

# Imports
import cv2

# Path to video file
VIDEO = './data/test.mov'

# OpenCV trackers
OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "mil": cv2.TrackerMIL_create,
}

# Defined type of trackers
TRACKER_1 = 'csrt'
TRACKER_2 = 'boosting'

# Defined colors of trackers
TRACKER_1_COLOR = (255, 0, 0)
TRACKER_2_COLOR = (0, 0, 255)

# Print OpenCV version (test using 4.6.0)
print('OpenCV version: ', cv2.__version__)

# Create instances of trackers
tracker1 = OPENCV_OBJECT_TRACKERS[TRACKER_1]()
tracker2 = OPENCV_OBJECT_TRACKERS[TRACKER_2]()

# Create video stream
vs = cv2.VideoCapture(VIDEO)

# Select bounding box of object
def get_bbox(vs):
    # Read first frame from video stream
    frame = vs.read()
    frame = frame[1]

    # Show frame size
    (height, width) = frame.shape[:2]
    print(f'Frame size: {width}x{height}')

    # Present frame
    cv2.imshow('Frame', frame)

    print('Please select bounding box')

    # Select bounding box
    bbox = cv2.selectROI('Frame', frame, fromCenter=False, showCrosshair=True)

    # Destroys the window showing image
    cv2.destroyAllWindows()

    # Trackers initialization
    tracker1.init(frame, bbox)
    tracker2.init(frame, bbox)

    return bbox

# Object bounding box which will be tracker
bbox = get_bbox(vs)

print(f'Bounding box: {bbox}')
print(f'Press "q" for exit')
print(f'Press any key to see next frame...')

# loop over frames from the video stream
while True:
    # Get video frame
    frame = vs.read()
    frame = frame[1]

    # Stop loop
    if frame is None:
        vs.release()
        break

    # Track the objet using both trackers
    (success1, box1) = tracker1.update(frame)
    (success2, box2) = tracker2.update(frame)

    # Check to see if the tracking was a success using tracker 1
    if success1:
        (x, y, w, h) = [int(v) for v in box1]
        cv2.rectangle(frame, (x, y), (x + w, y + h), TRACKER_1_COLOR, 2)

    # Check to see if the tracking was a success using tracker 2
    if success2:
        (x, y, w, h) = [int(v) for v in box2]
        cv2.rectangle(frame, (x, y), (x + w, y + h), TRACKER_2_COLOR, 2)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(0)

    # Enter "q" for exit
    if key == ord("q"):
        vs.release()
        cv2.destroyAllWindows()
        break

    cv2.destroyAllWindows()

'''
Compare the results:
* Do you see any differences? If so, what are they?
    
    ** Yes, they have differences. **

    # csrt - has resizing, good accuracy
    # kcf - static bounding box, handle slightly lower object tracking accuracy 
    # mil - looks similar to kcf 

* Does one tracker perform better than the other? In what way?
    
    ** All trackers has different characteristics and have different purposes. CSRT is looks better then others for detecting people which are resizing. **
'''
