import numpy as np
import cv2

def extract_cropped_image(image, top_left_x, top_left_y, bottom_right_x, bottom_right_y):
  return image[top_left_y:bottom_right_y, top_left_x:bottom_right_x].copy()

def removing_bottom_of_image(image):
  #Extracting top half of image for color histogram
  height = image.shape[0]
  width = image.shape[1]
  image_top_half = image
  image_top_half = image_top_half[0: int(height/2), 0:width]
  return image_top_half

def create_normalised_color_histogram(image): #No more flattening (more intuitive sense)
    histr = cv2.calcHist([image],[0,1,2], None, [16,16,16], [1,256,1,256,1,256]) #[1,256] to ignore the black pixels in calculations
    # histr = cv2.normalize(histr, histr, norm_type = cv2.NORM_L1).flatten() #Normalise color histogram
    histr = cv2.normalize(histr, histr, norm_type = cv2.NORM_L1).flatten() #Normalise color histogram #No flattining (gives similar results as with flattening but makes more intuitive sense)
    return histr
    # Here we compute histogram over 3 channels. Normally, we do it one by one so we can visualise them.
    # Now, we do it all together then normalise the histogram and flatten it for comparison

def hisEqulColor(img):
    ycrcb=cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
    channels=cv2.split(ycrcb)
    cv2.equalizeHist(channels[0],channels[0])
    cv2.merge(channels,ycrcb)
    cv2.cvtColor(ycrcb,cv2.COLOR_YCR_CB2BGR,img)
    return img

def return_diff_output_images(main_image, target_bbox, target_mask):
  output = {}

  mask = target_mask
  mask_stack = np.dstack([mask]*3)
  masked = mask_stack*main_image
  #Utilising the mask and extracting and keeping only unmasked pixels (only player with no background)

  top_left_x = int(np.floor(target_bbox[0]))
  top_left_y = int(np.floor(target_bbox[1]))
  bottom_right_x = int(np.ceil(target_bbox[2]))
  bottom_right_y = int(np.ceil(target_bbox[3]))


  masked_cropped_image = extract_cropped_image(masked, top_left_x, top_left_y, bottom_right_x, bottom_right_y)
  output['masked_cropped_image'] = masked_cropped_image
  # cv2_imshow(masked_cropped_image)

  masked_image_halved = removing_bottom_of_image(masked_cropped_image.copy())
  output['masked_image_halved'] = masked_image_halved
  # cv2_imshow(masked_image_halved)

  #Implementing histogram equalisation
  masked_image_halved_equalised = hisEqulColor(masked_image_halved.copy())
  output['masked_image_halved_hist_equalised'] = masked_image_halved_equalised

  normalised_histogram = create_normalised_color_histogram(masked_image_halved_equalised)
  output['normalised_histogram'] = normalised_histogram

  return output

def return_halved_equalised_color_hist(masked_cropped_image):
  output = {}

  masked_image_halved = removing_bottom_of_image(masked_cropped_image.copy())
  output['masked_image_halved'] = masked_image_halved

  #Implementing histogram equalisation
  masked_image_halved_equalised = hisEqulColor(masked_image_halved.copy())
  output['masked_image_halved_hist_equalised'] = masked_image_halved_equalised

  normalised_histogram = create_normalised_color_histogram(masked_image_halved_equalised)
  output['normalised_histogram'] = normalised_histogram

  return output