import os
import argparse
import random
from data import Data
import cv2
import blend_modes
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", type=str, default="./task_pressmachineoperator")
parser.add_argument("--type", type=str, default="default", help="train|val|trainval|test")
parser.add_argument("--logo_dir", type=str, default="./stroma_logo.png")
parser.add_argument("--random_seed", type=int, default=100)
parser.add_argument("--save_images", type=bool, default=True)
parser.add_argument("--save_dir", type=str, default="output")
parser.add_argument("--line_thickness", type=int, default=5)
args = parser.parse_args()
random.seed(args.random_seed)

img_dir = os.path.join(args.root_dir, 'JPEGImages')
ann_dir = os.path.join(args.root_dir, 'Annotations')
set_dir = os.path.join(args.root_dir, 'ImageSets', 'Main')


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



def logo_resize(background_image, foreground_image):
    """
    Returns a matrix of same size with background image with logo on the top left corner
    TODO: Add other location options
    """

    # Resize logo to 1/9th of height and 1/16th of length
    new_x, new_y = background_image.shape[1]//9,background_image.shape[0]//16
    foreground_image = cv2.resize(foreground_image, (new_x, new_y), interpolation = cv2.INTER_AREA)

    # Take differences of each indices
    x, y, z = tuple(map(lambda x, y: x - y, background_image.shape, foreground_image.shape))
    # Expand the logo matrix to match background resolution
    foreground_image = cv2.copyMakeBorder(src = foreground_image, top=15, bottom=x-15, left=10, right=y-10, borderType=cv2.BORDER_CONSTANT,value=[0,0,0])
    return foreground_image

def blend(background_image, foreground_image, add_alpha_layer = True):
    if add_alpha_layer:
        alpha = np.full((1080,1920),255.0) # Fully opaque 4.th layer (alpha) for BGR frames --> BGRA
        background_image = np.dstack((background_image,alpha)) # Add the opacity layer to background image
                                                                       # background_image.shape --> (1080, 1920, 4)
    
    # Blend images with 'normal' mode 
    # Opacity 1.0 and matrix dtype as uint8
    background_image = blend_modes.normal(background_image, foreground_image, 1.0).astype(np.uint8)

    # Return BGRA format (x,y,4)
    return background_image

def main(args):
    index = 0
    image_list = get_image_list(set_dir, args.type + ".txt")
    total_images = len(image_list)
    image_data = Data(args.root_dir, image_list[index])
    image = process_image(image_data)
    frame_width = image.shape[1]
    frame_height = image.shape[0]
    frame_layers = image.shape[2]  
    
    # if frame is not BGRA format add Alpha channel
    if frame_layers != 4:
        add_alpha_layer = True
    
    # Adjust MP4 codec
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    
    out = cv2.VideoWriter('top_left_logo_30fps.mp4',fourcc, 30, (frame_width,frame_height))
    
    # Read specified logo
    foreground_img_float = cv2.imread(args.logo_dir,-1).astype(float)
    # Create logo frame for blending process
    foreground_img_float = logo_resize(image, foreground_img_float)
 

    while index != total_images:
        image_data = Data(args.root_dir, image_list[index])
        print(image_data.image_path)
        background_img_float = process_image(image_data)
        blended_img_uint8 = blend(background_img_float, foreground_img_float, add_alpha_layer)

        out.write(blended_img_uint8[:,:,:3]) # Write frames with 3 layers [:,:,:3] --> (1080, 1920, 3) BGR format
        index = index + 1

    out.release()


if __name__ == '__main__':
    main(args)
