import numpy as np
import cv2
import threading


def shadow_detect(img, shadow, pt1, pt2, roi_percent):

	shadow = shadow
	img = img
	#shadow = cv2.imread(shadow)
	# img = cv2.imread(img)

	# Converting into 0 and 255
	shadow = cv2.threshold(shadow, 100, 255, cv2.THRESH_BINARY)[1]

	area = (pt2[0]-pt1[0]+1) * (pt2[1]-pt1[1]+1) * roi_percent

	# print("Area of the person", area)

	confidence = int(pt1[1] + (pt2[1]-pt1[1])*roi_percent)

	# print("confidence", confidence)

	img2 = np.ndarray(shadow.shape)
	temp = cv2.rectangle(img2, (pt1[0],confidence), (pt2[0],pt2[1]), (255,255,255), -1)
	# print(temp.dtype)
	# print(img.dtype)
	# blend = cv2.addWeighted(img, 0.7, temp.astype(np.uint8), 0.3, 0)
	#cv2.imshow("sample",blend)
	#cv2.waitKey(0)
	iou = shadow*img2
	iou_area = iou[iou==65025].shape[0]

	# print("IOU Area", iou_area)

	if iou_area>=area:
		print("true")
		return True
	else:
		return False


#print(shadow_detect("./Shadow_Images/sample4.jpg", "./Shadow_Map/shadow.jpg", (181,224), (433, 529), 0.5))



