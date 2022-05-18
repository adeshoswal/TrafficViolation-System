python detect.py --weights weights/last.pt  --conf 0.40 --source snip1.JPG

Once the detected license plate is cropped run it through ocr.py file to get the license plate recongnized.
ocr.py file will require to configure with the google ocr and the api key will needed to be added once credits have been bought from google cloud vision.
For more details: https://cloud.google.com/vision/docs/ocr