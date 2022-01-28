
def extract_cropped_image(image, top_left_x, top_left_y, bottom_right_x, bottom_right_y):
  return image[top_left_y:bottom_right_y, top_left_x:bottom_right_x].copy()

def removing_bottom_of_image(image):
  #Extracting top half of image for color histogram
  height = image.shape[0]
  width = image.shape[1]
  image_top_half = image
  image_top_half = image_top_half[0: int(height/2), 0:width]
  return image_top_half

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

  return output