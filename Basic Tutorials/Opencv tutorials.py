
import cv2
import numpy as np

print(cv2.__version__)  #checking version

# img=cv2.imread("E:\Computer Vision\IMG-20221019-WA0011.jpg",0)
# if the flag above is 1, loads a color image.
# if 0, it loads a grayscale image
# if -1, loads image as such including alpha channel
# print(img)
# returns a matrix of the pixels read

# cv2.imshow('image',img) ## image is the window to display the image, name can be anything
# # cv2.waitKey(5000) ## delay - 5000ms
# cv2.waitKey(0)  ## wait till we close the window
# # cv2.destroyAllWindows()

# # cv2.imwrite('E:\Computer Vision\img_copy.png',img) #Generating a copy of the generated 
# modified version of the image


# #Adding functionalities
# k=cv2.waitKey(0)
# if k==27:           #27 is the keyboard value for esc key
#     cv2.destroyAllWindows()
# elif k==ord('s'):       # if s is pressed, the content is saved
#     cv2.imwrite('E:\Computer Vision\img_copy.png',img)
#     cv2.destroyAllWindows()

#--------------------------------------

# # using webcam

# cap = cv2.VideoCapture(0)    
# # if we wanna use an already available video file, instead of 0, put the path of the file in the braces above

# print(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # returns frame width
# print(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # returns frame height

# cap.set(3,700) # sets the frame width to 700, 3 is the number associated with width
# cap.set(4, 700) # sets the frame height to 700, 4 is the number associated with heigth

# Note - the set function changes only the window resolution. The camera resolution can't be changed
#        It would remain the same as the standard camera resolution available

# fourcc=cv2.VideoWriter_fourcc(*'XVID')  # fourcc is something standard
# # to save the output file,
# out=cv2.VideoWriter('E:\Computer Vision\output_file_name.avi',
#                     fourcc,
#                     20.0, # frames per second
#                     (640,480)#size
#                     )

# while(True):
#     ret,frame=cap.read()
#     if ret==True:    # ret returns true if the given file is valid and opened, else it returns false,
#                      # for ex, if the given path is invalid
#         cv2.imshow('frame',frame)
#         out.write(frame)
#         if cv2.waitKey(1) & 0xFF==ord('Q'):
#             break
#     else:
#         break   

#     #Converting the live video feed into grayscale    
#     gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#     cv2.imshow('grayscale',gray)
#     if cv2.waitKey(1) & 0xFF==ord('Q'):
#         break   

#     # for getting various parameters of the videos like height, width, hue and stuff, 
#     # refer official documentation
# out.release()
# cap.release()
# cv2.destroyAllWindows()

# ----------------------------------

# # Drawing geometric shapes on images

# img=cv2.imread("E:\Computer Vision\IMG-20221019-WA0011.jpg",1)

# # instead of using an image, we can even create a blank image using numpy and use that for annotation

# # img=np.zeros([512,512,3],np.uint8)
# # syntax - zeros([height, width, 3], np.uint8) for black image

# # Drawing a line
# img=cv2.line(img,(0,0),(255,255),(255,0,0),5)
# # syntax - line(image_where_we_wanna_draw, starting coordinates, ending coordinates,
# # color of the line in BGR format, thickness of the line)

# # Note : It is BGR format and not RGB

# #Arrowed line
# img=cv2.arrowedLine(img,(0,0),(255,60),(255,0,0),5)

# # Rectangle
# img=cv2.rectangle(img,(100,200),(500,640),(255,245,24),5)
# # syntax - rectangle(image var, top left coordinate of rectangle, bottom right coordinate of rectangle, color, thickness)

# # if instead of thickness, we enter -1, the rectangle gets filled with the given color
# img=cv2.rectangle(img,(100,200),(500,640),(255,245,24),-1)

# #circle
# img=cv2.circle(img,(100,100),50,(0,145,130),-1)

# # Text insertion
# font=cv2.FONT_HERSHEY_SIMPLEX   
# img=cv2.putText(img,'Opencv Text Insertion',(10,500),font,1,(250,0,0),2)
# # syntax - putText(img, text to be inserted, starting coordinate, font, font size, color, thickness)


# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# --------------------------------

# Mouse click events

# events=[i for i in dir(cv2) if 'EVENT' in i] # Listing the events available in cv2
# print(events)

# def click_event(event, x, y, flags, param):     #standard function definition format to call any events
#     if event == cv2.EVENT_LBUTTONDOWN:
#         # printing the coordinate at any instance
#         print(x,', ' ,y)
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         strXY = str(x) + ', '+ str(y)
#         cv2.putText(img, strXY, (x, y), font, .5, (255, 255, 0), 2)
#         cv2.imshow('image', img)
#     if event == cv2.EVENT_RBUTTONDOWN:
#         # printing the colour at that instance in BGR format
#         blue = img[y, x, 0]  # y coordinate, x coordinate, zero represents the blue column
#         green = img[y, x, 1]
#         red = img[y, x, 2]
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         strBGR = str(blue) + ', '+ str(green)+ ', '+ str(red)
#         cv2.putText(img, strBGR, (x, y), font, .5, (0, 255, 255), 2)
#         cv2.imshow('image', img)

# #img = np.zeros((512, 512, 3), np.uint8)
# img = cv2.imread('E:\Computer Vision\IMG-20221019-WA0011.jpg')
# cv2.imshow('image', img)

# cv2.setMouseCallback('image', click_event)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ----------------------------------

# Drawing a point on mouse click event and connecting them subsequently

# def click_event(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         cv2.circle(img,(x,y),3,(0,0,255),-1)
#         points.append((x,y))
#         if len(points)>=2:
#             cv2.line(img,points[-1],points[-2],(255,0,0),5)
#         cv2.imshow('image', img)
    

# #img = np.zeros((512, 512, 3), np.uint8)
# img = cv2.imread('E:\Computer Vision\IMG-20221019-WA0011.jpg')
# cv2.imshow('image', img)
# points=[]
# cv2.setMouseCallback('image', click_event)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ---------------------------------------

# img = cv2.imread('E:\Computer Vision\IMG-20221019-WA0011.jpg')
# print(img.shape) # returns a tuple of number of rows, columns, and channels
# print(img.size) # returns Total number of pixels is accessed
# print(img.dtype) # returns Image datatype is obtained
# b,g,r=cv2.split(img) # output vector of arrays; the arrays themselves are reallocated.
# img=cv2.merge((b,g,r)) # The number of channels will be the total number of channels in the matrix array.

# img=cv2.resize(img,[512,512]) # resize the image
# # resize(src, no of rows and columns)

# dst = cv2.add(img, img2) # Calculates the per-element sum of two arrays or an array and a scalar.
# # Note - both the images must be of same size, if not resize them first

# dst = cv2.addWeighted(img, .2, img2, .8, 0) # Calculates the weighted sum of two arrays.
# # syntax - addWeighthed(src img 1, it's weightage, src img 2, it's weightage, gamma value)

# # cv2.imshow('image',dst)
# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ------------------------------

# img1 = np.zeros((250, 500, 3), np.uint8)
# img1 = cv2.rectangle(img1,(200, 0), (300, 100), (255, 255, 255), -1)
# img2 = cv2.imread("E:\Computer Vision\IMG-20221019-WA0011.jpg")

# bitAnd = cv2.bitwise_and(img2, img1)
# bitOr = cv2.bitwise_or(img2, img1)
# bitXor = cv2.bitwise_xor(img1, img2)
# bitNot1 = cv2.bitwise_not(img1)
# bitNot2 = cv2.bitwise_not(img2)

# cv2.imshow("img1", img1)
# cv2.imshow("img2", img2)
# cv2.imshow('bitAnd', bitAnd)
# cv2.imshow('bitOr', bitOr)
# cv2.imshow('bitXor', bitXor)
# cv2.imshow('bitNot1', bitNot1)
# cv2.imshow('bitNot2', bitNot2)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ----------------------

#changing the colour of the images using a trackbar

# import cv2 as cv

# def nothing(x):
#     print(x)

# # Create a black image, a window
# img = np.zeros((300,512,3), np.uint8)
# cv.namedWindow('image')

# cv.createTrackbar('B', 'image', 0, 255, nothing)
# cv.createTrackbar('G', 'image', 0, 255, nothing)
# cv.createTrackbar('R', 'image', 0, 255, nothing)

# switch = '0 : OFF\n 1 : ON'
# cv.createTrackbar(switch, 'image', 0, 1, nothing)

# while(1):
#     cv.imshow('image',img)
#     k = cv.waitKey(1) & 0xFF
#     if k == 27:
#         break

#     b = cv.getTrackbarPos('B', 'image')
#     g = cv.getTrackbarPos('G', 'image')
#     r = cv.getTrackbarPos('R', 'image')
#     s = cv.getTrackbarPos(switch, 'image')

#     if s == 0:
#        img[:] = 0
#     else:
#        img[:] = [b, g, r]


# cv.destroyAllWindows()

# Similarly, we can even convert a normal colored image into grayscale image using a trackbar

# ------------------------------------
