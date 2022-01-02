# # OpenCV Tutorial 01


# ## Load Libraries
import imutils
import cv2


# ## Loading and displaying an image
image = cv2.imread("images/img1.png")
h, w, d = image.shape
print("width={}, height={}, depth={}".format(w, h, d))

cv2.imshow("Image", image)
cv2.waitKey(0)


# ## Accessing individual pixels
B, G, R = image[100, 50]
print("R={}, G={}, B={}".format(R, G, B))


# ## Array slicing and cropping
roi = image[60:160, 320:420]
cv2.imshow("ROI", roi)
cv2.waitKey(0)


# ## Resizing images
# ### Fixed Resizing
resized = cv2.resize(image, (200, 200))
cv2.imshow("Fixed Resizing", resized)
cv2.waitKey(0)

# ### Aspect Ratio Resize
r = 300.0 / w
dim = (300, int(h * r))
resized = cv2.resize(image, dim)
cv2.imshow("Aspect Ratio Resize", resized)
cv2.waitKey(0)

# ### With imutils
resized = imutils.resize(image, width=300)
cv2.imshow("Imutils Resize", resized)
cv2.waitKey(0)


# ## Rotating an image
# ### With OpenCV
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, -45, 1.0)
rotated = cv2.warpAffine(image, M, (w, h))
cv2.imshow("OpenCV Rotation", rotated)
cv2.waitKey(0)

# ### With imutils
rotated = imutils.rotate(image, -45)
cv2.imshow("Imutils Rotation", rotated)
cv2.waitKey(0)

# ### Imutils bound rotation
rotated = imutils.rotate_bound(image, 45)
cv2.imshow("Imutils Bound Rotation", rotated)
cv2.waitKey(0)


# ## Smoothing an image
blurred = cv2.GaussianBlur(image, (13, 13), 0)
cv2.imshow("Blurred", blurred)
cv2.waitKey(0)


# ## Drawing on an image
# ### Draw rectangle
output = image.copy()
cv2.rectangle(output, (320, 60), (420, 160), (0, 0, 255), 2)
cv2.imshow("Rectangle", output)
cv2.waitKey(0)

# ### Draw circle
output = image.copy()
cv2.circle(output, (290, 130), 20, (255, 0, 0), -1)
cv2.imshow("Circle", output)
cv2.waitKey(0)

# ### Draw line
output = image.copy()
cv2.line(output, (60, 20), (400, 200), (0, 0, 255), 5)
cv2.imshow("Line", output)
cv2.waitKey(0)

# ### Draw text
output = image.copy()
cv2.putText(output, "OpenCV + Jurassic Park!!!", (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv2.imshow("Text", output)
cv2.waitKey(0)
