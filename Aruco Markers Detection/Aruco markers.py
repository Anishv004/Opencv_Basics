import cv2
import numpy as np
from cv2 import aruco
import os

# #Generating aruco markers
# for id in range(20):
#     marker_image=aruco.drawMarker(marker_dict,id,MARKER_SIZE)
#     cv2.imshow("img",marker_image)
#     cv2.imwrite(f"E:\Computer Vision\Aruco Markers Detection\markers\{id}.jpeg",marker_image)
#     # cv2.waitKey(0)
#     # break

# # We can also generate aruco markers from the website https://fodi.github.io/arucosheetgen/

#  ---------------------------------------

# # Detection of aruco markers
# marker_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
# MARKER_SIZE = 400  # pixels
# param_markers = aruco.DetectorParameters.create()

# # Using webcam
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     marker_corners, marker_IDs, reject = aruco.detectMarkers(       # parameters
#         gray_frame, marker_dict, parameters=param_markers
#     )
#     if marker_corners:
#         for ids, corners in zip(marker_IDs, marker_corners):
#             cv2.polylines(      # Draw boxes over the aruco marker
#                 frame, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv2.LINE_AA
#             )
#             # print(idqs," ", corners)
#             corners = corners.reshape(4, 2)
#             corners = corners.astype(int)
#             top_right = corners[0].ravel()
#             cv2.putText(        # Printing the text(id) over the detected aruco marker
#                 frame,
#                 f"id:{ids[0]}",
#                 top_right,
#                 cv2.FONT_HERSHEY_COMPLEX,
#                 1.3,
#                 (255, 0, 0),
#                 2,
#                 cv2.LINE_AA,
#             )
#     # print(marker_IDs)
#     cv2.imshow("frame", frame)
#     key = cv2.waitKey(1)
#     if key == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()



# --------------------------------

# Image augmentation

def image_augmentation(frame,src_image,dst_points):
    # augmenting the image over the aruco marker
    src_h,src_w=src_image.shape[:2] 
    frame_h,frame_w=frame.shape[:2]

    mask=np.zeros((frame_h,frame_w),dtype=np.uint8)
    src_points=np.array([[0,0],[src_w,0],[src_h,src_w],[0,src_h]])
    H,_=cv2.findHomography(srcPoints=src_points,dstPoints=dst_points)
    warp_image=cv2.warpPerspective(src_image,H,(frame_w,frame_h))
    cv2.imshow("warp image",warp_image)
    cv2.fillConvexPoly(mask,dst_points,255) # filling with white clr
    results=cv2.bitwise_and(warp_image,warp_image,frame,mask=mask)

def read_img(dir_path):
    # reading the list of images from a directory
    img_list=[]
    files=os.listdir(dir_path)
    for file in files:
        img_path=os.path.join(dir_path,file)
        image=cv2.imread(img_path)
        img_list.append(image)
        
    return img_list

marker_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
MARKER_SIZE = 400  # pixels
param_markers = aruco.DetectorParameters.create()

# aug_img=cv2.imread("E:\Computer Vision\Aruco Markers Detection\images\sample.jpg")
images_list=read_img("E:\Computer Vision\Aruco Markers Detection\images")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    marker_corners, marker_IDs, reject = aruco.detectMarkers(       # parameters
        gray_frame, marker_dict, parameters=param_markers
    )
    if marker_corners:
        for ids, corners in zip(marker_IDs, marker_corners):
            cv2.polylines(      # Draw boxes over the aruco marker
                frame, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv2.LINE_AA
            )
            # print(idqs," ", corners)
            corners = corners.reshape(4, 2)
            corners = corners.astype(int)
            if ids[0]<4:
                image_augmentation(frame,images_list[ids[0]],corners)
            else:
                image_augmentation(frame,images_list[0],corners)
            
    # print(marker_IDs)
    cv2.imshow("frame", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()




