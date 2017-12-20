from skimage.feature import corner_harris, corner_peaks
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.interpolation import geometric_transform
from sklearn.preprocessing import normalize
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import random
import math
from numpy.linalg import inv
import pdb
import cv2
import imageio


def FaceSwapping():

  # extract images for all three videos and perform mosaicing on them
  vid1 = imageio.get_reader("JonSnow.mp4", 'ffmpeg')
  # vid2 = imageio.get_reader("trimmed_clip_2.mp4", 'ffmpeg')
  # vid3 = imageio.get_reader("trimmed_clip_3.mp4", 'ffmpeg')

  # append each video's frame onto its assigned list
  pic_list1 = []
  # pic_list2 = []
  # pic_list3 = []
  for i, frame in enumerate(vid1):
    pic_list1.append(frame)
  # for i, frame in enumerate(vid2):
  #   pic_list2.append(frame)
  # for i, frame in enumerate(vid3):
  #   pic_list3.append(frame)

  # mosaic together all images and populate list of mosaic images
  # mosaic_list = []
  print("NUMBER OF FRAMES", len(pic_list1))
  for i in range(0, 250, 1):
    print ("number:  ", i)
    frame = pic_list1[i]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    F = faces.shape[0]
    bbox = np.zeros((F, 4, 2))
    for i, face in enumerate(faces):
      minx = face[1]
      miny = face[0]
      xlen = face[3]
      ylen = face[2]
      bbox[i, 0:2, 0] = minx
      bbox[i, 2:4, 0] = minx + xlen
      bbox[i, (0, 2), 1] = miny
      bbox[i, (1, 3), 1] = miny + ylen

    hullIndex = cv2.convexHull(points, returnPoints=False)

    curr_list = [pic_list1[i], pic_list2[i], pic_list3[i]]
    curr_mosaic = np.uint8(mosaic(curr_list))

    plt.imshow(np.uint8(curr_mosaic))
    plt.show()
    print("CURRENT FRAME MATRIX", curr_mosaic[:, :, 0])
    mosaic_list.append(curr_mosaic)

  imageio.mimsave('./ourvideo1.gif', mosaic_list)
  list_length = len(mosaic_list)
  mosaic_list = np.reshape(mosaic_list, (list_length, 1))

  return mosaic_list


mosaic_list = FaceSwapping()

if __name__ == "__main__":

  if len(sys.argv) > 1 and sys.argv[1] == "tkkagg":
    plt.switch_backend("tkagg")
  test_detect()
