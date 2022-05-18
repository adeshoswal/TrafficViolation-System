Module 1 and 3 are integrated into single code file stopLineViolation

Module 1 is simply detection of vehicles and classifying them. We have trained Yolov5 model according to our requirement which was to detect and classify vehicles on Indian roads. Hence our weights are able to detect and classify vehicles on highly dense roads as well giving about 30 detections in one frame. Classes used: motorcycle,car,truck,bicycle,bus

The weight file of Module 1 is then used to make detections. Most of the videos and images in our dataset have faded stopline or no stopline. Hence in this method we are choosing a reference parallelogram which could be considered as a violation zone. The detections got are checked if they are intersecting with reference parallelogram. If they intersect then the vehicles are considered as violating. This reference parallelogram is chosen according to survillence camera video. We have chosen it manually as polylist=[l2,r3,r2,l3] where l2,r3,r2,l3 are points. 
For improvement in results:
	1)A vehicle is flagged as violator only if it is violating for a threshold number of frames.
	2)Indexes are maintained in order to avoid duplicate violations.
	
The Yolov5 model can be trained according to requirement. Then the weight file of .pt format needs to be converted to .onxx format so that cv2 functions could be used with onxx weight files. This conversion coiuld be done using export.py file provided by yolov5 community.
***While this conversion there should be no kinds of errors given by execution of export.py file. If errors occur then the weight file would give errors while execution of stopLineViolation.py file also
Make sure opencv version 4.5.4.60 is used if newer versions are not working.
