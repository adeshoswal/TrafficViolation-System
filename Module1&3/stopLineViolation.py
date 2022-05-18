import cv2
import time
import sys
import numpy as np

def build_model(is_cuda):
    net = cv2.dnn.readNetFromONNX("best.onnx")
    if is_cuda:
        print("Attempty to use CUDA")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    else:
        print("Running on CPU")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net

INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
CONFIDENCE_THRESHOLD = 0.4

def detect(image, net):
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()
    #print(preds)
    return preds

def load_capture():
    capture = cv2.VideoCapture("videos/traffic_t11.mp4")
    return capture

def load_classes():
    class_list = []
    with open("classes.txt", "r") as f:
        class_list = [cname.strip() for cname in f.readlines()]
    return class_list

class_list = load_classes()

def wrap_detection(input_image, output_data):
    class_ids = []
    confidences = []
    boxes = []

    rows = output_data.shape[0]

    image_width, image_height, _ = input_image.shape

    x_factor = image_width / INPUT_WIDTH
    y_factor =  image_height / INPUT_HEIGHT

    for r in range(rows):
        row = output_data[r]
        confidence = row[4]
        if confidence >= 0.4:

            classes_scores = row[5:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if (classes_scores[class_id] > .25):

                confidences.append(confidence)

                class_ids.append(class_id)

                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item() 
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45) 

    result_class_ids = []
    result_confidences = []
    result_boxes = []

    for i in indexes:
        result_confidences.append(confidences[i])
        result_class_ids.append(class_ids[i])
        result_boxes.append(boxes[i])

    return result_class_ids, result_confidences, result_boxes

def format_yolov5(frame):

    row, col, _ = frame.shape
    _max = max(col, row)
    result = np.zeros((_max, _max, 3), np.uint8)
    result[0:row, 0:col] = frame
    return result
def doOverlap(l1, r1, l2, r2):
     
    # To check if either rectangle is actually a line
      # For example  :  l1 ={-1,0}  r1={1,1}  l2={0,-1}  r2={0,1}
    (l1x,l1y)=l1 
    (l2x,l2y)=l2
    (r1x,r1y)=r1
    (r2x,r2y)=r2
    if (l1x == r1x or l1y == r1y or l2x == r2x or l2y == r2y):
        # the line cannot have positive overlap
        return False
       
     
    # If one rectangle is on left side of other
    if(l1x >= r2x or l2x >= r1x):
        return False
 
    # If one rectangle is above other
    if(r1y <= l2y or r2y <= l1y):
        return False
    return True


# for explanation of functions onSegment(),
# orientation() and doIntersect()

# Define Infinite (Using INT_MAX
# caused overflow problems)
INT_MAX = 10000

# Given three collinear points p, q, r,
# the function checks if point q lies
# on line segment 'pr'
def onSegment(p:tuple, q:tuple, r:tuple) -> bool:
	
	if ((q[0] <= max(p[0], r[0])) &
		(q[0] >= min(p[0], r[0])) &
		(q[1] <= max(p[1], r[1])) &
		(q[1] >= min(p[1], r[1]))):
		return True
		
	return False

# To find orientation of ordered triplet (p, q, r).
# The function returns following values
# 0 --> p, q and r are collinear
# 1 --> Clockwise
# 2 --> Counterclockwise
def orientation(p:tuple, q:tuple, r:tuple) -> int:
	
	val = (((q[1] - p[1]) *
			(r[0] - q[0])) -
		((q[0] - p[0]) *
			(r[1] - q[1])))
			
	if val == 0:
		return 0
	if val > 0:
		return 1 # Collinear
	else:
		return 2 # Clock or counterclock

def doIntersect(p1, q1, p2, q2):
	
	# Find the four orientations needed for
	# general and special cases
	o1 = orientation(p1, q1, p2)
	o2 = orientation(p1, q1, q2)
	o3 = orientation(p2, q2, p1)
	o4 = orientation(p2, q2, q1)

	# General case
	if (o1 != o2) and (o3 != o4):
		return True
	
	# Special Cases
	# p1, q1 and p2 are collinear and
	# p2 lies on segment p1q1
	if (o1 == 0) and (onSegment(p1, p2, q1)):
		return True

	# p1, q1 and p2 are collinear and
	# q2 lies on segment p1q1
	if (o2 == 0) and (onSegment(p1, q2, q1)):
		return True

	# p2, q2 and p1 are collinear and
	# p1 lies on segment p2q2
	if (o3 == 0) and (onSegment(p2, p1, q2)):
		return True

	# p2, q2 and q1 are collinear and
	# q1 lies on segment p2q2
	if (o4 == 0) and (onSegment(p2, q1, q2)):
		return True

	return False

# Returns true if the point p lies
# inside the polygon[] with n vertices
def is_inside_polygon(points:list, p:tuple) -> bool:
	
	n = len(points)
	
	# There must be at least 3 vertices
	# in polygon
	if n < 3:
		return False
		
	# Create a point for line segment
	# from p to infinite
	extreme = (INT_MAX, p[1])
	count = i = 0
	
	while True:
		next = (i + 1) % n
		
		# Check if the line segment from 'p' to
		# 'extreme' intersects with the line
		# segment from 'polygon[i]' to 'polygon[next]'
		if (doIntersect(points[i],
						points[next],
						p, extreme)):
							
			# If the point 'p' is collinear with line
			# segment 'i-next', then check if it lies
			# on segment. If it lies, return true, otherwise false
			if orientation(points[i], p,
						points[next]) == 0:
				return onSegment(points[i], p,
								points[next])
								
			count += 1
			
		i = next
		
		if (i == 0):
			break
		
	# Return true if count is odd, false otherwise
	return (count % 2 == 1)
colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]

is_cuda = len(sys.argv) > 1 and sys.argv[1] == "cuda"

net = build_model(is_cuda)
capture = load_capture()

start1 = time.time()
start = time.time_ns()
frame_count = 0
total_frames = 0
fps = -1
writer=None
while True:

    _, frame = capture.read()
    if frame is None:
        print("End of stream")
        break
    print(total_frames)
    #inputImage = format_yolov5(frame)
    frame = cv2.resize(frame, (1000, 1000))
    inputImage=frame
    outs = detect(inputImage, net)

    class_ids, confidences, boxes = wrap_detection(inputImage, outs[0])

    frame_count += 1
    total_frames += 1

    for (classid, confidence, box) in zip(class_ids, confidences, boxes):
         (x, y) = (box[0], box[1])
         (w, h) = (box[2], box[3])

         l1=(x, y)
         r1=(x + w, y + h)
         l2=(0, 490)
         r2=(1300, 930)
         l3=(0, 770)
         r3=(1300, 650)
         polylist=[l2,r3,r2,l3]
         if is_inside_polygon(polylist,l1) or is_inside_polygon(polylist,r1):
             color = colors[int(classid) % len(colors)]
             cv2.rectangle(frame, box, color, 2)
             cv2.rectangle(frame, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
             cv2.putText(frame, class_list[classid], (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0))
    cv2.line(frame, (0, 770), (1300, 930), (0, 0, 0), 3, cv2.LINE_AA)
    cv2.line(frame, (0, 490), (1300, 650), (0, 0, 0), 3, cv2.LINE_AA)
    if frame_count >= 30:
        end = time.time_ns()
        fps = 1000000000 * frame_count / (end - start)
        frame_count = 0
        start = time.time_ns()
    
    if fps > 0:
        fps_label = "FPS: %.2f" % fps
        cv2.putText(frame, fps_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # check if the video writer is None
    if writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter("output/final1.mp4", fourcc, 30,
            (frame.shape[1], frame.shape[0]), True)
    #cv2.imshow("output", frame)
    writer.write(frame)
    if cv2.waitKey(1) > -1:
        print("finished by user")
        break

end1 = time.time()
print("The time of execution of above program is :", end1-start1)
print("Total frames: " + str(total_frames))