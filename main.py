import os
import argparse
import random
from data import Data
import cv2
import blend_modes
import numpy as np
import time


parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", type=str, default="./task_pressmachineoperator")
parser.add_argument("--type", type=str, default="default", help="train|val|trainval|test")
parser.add_argument("--logo_dir", type=str, default="./stroma_logo.png")
parser.add_argument("--logo_loc", type=str, default="TOP_LEFT", help="TOP_LEFT|TOP_RIGHT|BOTTOM_LEFT|BOTTOM_RIGHT")
parser.add_argument("--random_seed", type=int, default=100)
parser.add_argument("--save_images", type=bool, default=True)
parser.add_argument("--save_dir", type=str, default="output")
parser.add_argument("--line_thickness", type=int, default=5)
args = parser.parse_args()
random.seed(args.random_seed)

img_dir = os.path.join(args.root_dir, 'JPEGImages')
ann_dir = os.path.join(args.root_dir, 'Annotations')
set_dir = os.path.join(args.root_dir, 'ImageSets', 'Main')

alpha = np.array([])


def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1,y1 = pt1
    x2,y2 = pt2

    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness,cv2.LINE_AA)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness,cv2.LINE_AA)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness,cv2.LINE_AA)

    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness,cv2.LINE_AA)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness,cv2.LINE_AA)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness,cv2.LINE_AA)

    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness,cv2.LINE_AA)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness,cv2.LINE_AA)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness,cv2.LINE_AA)

    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness,cv2.LINE_AA)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness,cv2.LINE_AA)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness,cv2.LINE_AA)

def get_image_list(dir, filename):
    image_list = open(os.path.join(dir, filename)).readlines()
    return [image_name.strip() for image_name in image_list]


def process_image(image_data):
    image = cv2.imread(image_data.image_path)
    # image = cv2.putText(image, image_data.image_name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    for ann in image_data.annotations:
        box_color = (0, 255, 0)  #Green
        if ann.difficult or ann.truncated:
            box_color = (0, 0, 255) #Red
        draw_border(image, (ann.xmin-5, ann.ymin-5), (ann.xmax+5, ann.ymax+5), (168, 250, 83), 2, 5, 5)
        # image = cv2.rectangle(image, (ann.xmin, ann.ymin), (ann.xmax, ann.ymax), box_color, args.line_thickness,cv2.LINE_AA)
        image = cv2.putText(image, ann.name, (ann.xmin, ann.ymin), cv2.FONT_HERSHEY_DUPLEX, 1, (191, 51, 4),2,cv2.LINE_AA)
    return image



def resize_logo(background_image, foreground_image):
    """
    Returns a matrix of the logo resized with respect to background image
    usage:
    python main.py --logo_loc "TOP_RIGHT"
    
    TODO: Add CENTER option with 0.5 opacity and big logo
    """
    # if location == CENTER:
    #     new_x, new_y = background_image.shape[1]//2,background_image.shape[0]//3
    #     foreground_image = cv2.resize(foreground_image, (new_x, new_y), interpolation = cv2.INTER_AREA)
    #     x, y, z = tuple(map(lambda x, y: x - y, background_image.shape, foreground_image.shape))
    #     foreground_image = cv2.copyMakeBorder(src = foreground_image, top=x-15, bottom=15, left=y-10, right=10, borderType=cv2.BORDER_CONSTANT,value=[0,0,0])

    # Resize logo
    new_x, new_y = background_image.shape[1]//10, background_image.shape[1]//10*foreground_image.shape[0]//foreground_image.shape[1]
    foreground_image = cv2.resize(foreground_image, (new_x, new_y), interpolation = cv2.INTER_AREA)
    return foreground_image

def blend(background_image, foreground_image, location = "TOP_LEFT"):
    """
    Overlays the given foreground image to the background image with 100% opacity. Only operates calculations on the specified location (Region of Interest).
    """
    # Adds 4.th Alpha layer to the background image if there is not.
    global alpha
    if alpha.size == 0:
        alpha = np.full((background_image.shape[0],background_image.shape[1]),255.0) # Fully opaque 4.th layer (alpha) for BGR frames --> BGRA
    if background_image.shape[2] != 4:
        background_image = np.dstack((background_image,alpha)) # Add the opacity layer to background image
                                                                   # background_image.shape --> (1080, 1920, 4)
    
    # Location indices of the logo
    x_loc, y_loc = foreground_image.shape[:2]
        
    if location == "TOP_LEFT":
        blended = blend_modes.normal(background_image[15:x_loc+15,15:y_loc+15,:], foreground_image, 1.0)
        background_image[15:x_loc+15,15:y_loc+15,:] = blended
    elif location == "TOP_RIGHT":
        blended = blend_modes.normal(background_image[15:x_loc+15,-(15+y_loc):-15,:], foreground_image, 1.0)
        background_image[15:x_loc+15,-(15+y_loc):-15,:] = blended
    elif location == "BOTTOM_LEFT":
        blended = blend_modes.normal(background_image[-(15+x_loc):-15,15:y_loc+15,:], foreground_image, 1.0)
        background_image[-(15+x_loc):-15,15:y_loc+15,:]= blended
    elif location == "BOTTOM_RIGHT":
        blended = blend_modes.normal(background_image[-(15+x_loc):-15,-(15+y_loc):-15,:], foreground_image, 1.0)
        background_image[-(15+x_loc):-15,-(15+y_loc):-15,:] = blended
    return background_image

def main(args):
    index = 0
    image_list = get_image_list(set_dir, args.type + ".txt")
    total_images = len(image_list)
    image_data = Data(args.root_dir, image_list[index])
    image = process_image(image_data)

    # Adjust MP4 codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter('top_left_logo_30fps.mp4',fourcc, 30, (image.shape[1],image.shape[0]))
    
    # Read specified logo
    foreground_img_float = cv2.imread(args.logo_dir,-1).astype(np.float32)


    # Resize logo with respect to background image
    resized_logo = resize_logo(image, foreground_img_float)


    while index != total_images:
        image_data = Data(args.root_dir, image_list[index])
        print(image_data.image_path)
        background_img_float = process_image(image_data)
        blended_img = blend(background_img_float, resized_logo, args.logo_loc)

        out.write(blended_img[:,:,:3].astype(np.uint8)) # Write frames with 3 layers [:,:,:3] --> (1080, 1920, 3) BGR format
                                                        # Convert frame to OpenCV native display format
        index = index + 1

    out.release()


if __name__ == '__main__':
    startTime = time.time()
    main(args)
    print ('The script took {0} second !'.format(time.time() - startTime))

