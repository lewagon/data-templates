import cv2

def annotate_face(img_np_array):
  """
  Detect and annotate faces with a red square.
  `img_np_array` should be a (width, height, 3) shape np.array
  """
  # Load the default cascade from OpenCV
  face_cascade = cv2.CascadeClassifier('face_rec/haarcascade_frontalface_default.xml')

  # Detect faces
  faces = face_cascade.detectMultiScale(img_np_array, 1.1, 4)

  # Draw rectangle around the faces
  for (x, y, w, h) in faces:
    cv2.rectangle(img_np_array, (x, y), (x+w, y+h), (0, 0, 255), 2)

  # return image numpy array
  return img_np_array
