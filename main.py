import os
import argparse
import random
from data import Data
import cv2
import blend_modes
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", type=str, default="./task_pressmachineoperator")
parser.add_argument("--type", type=str, default="train", help="train|val|trainval|test")
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


def main(args):
    index = 0
    image_list = get_image_list(set_dir, args.type + ".txt")
    total_images = len(image_list)
    image_data = Data(args.root_dir, image_list[index])
    image = process_image(image_data)
    frame_width = image.shape[1]
    frame_height = image.shape[0]
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # mp4 encoding
    # index = random.randint(0, total_images)
    out = cv2.VideoWriter('outpy_30fps.mp4',fourcc, 30, (frame_width,frame_height))
    

    # Import foreground image
    # 1920x1080 transparent frame with logo on top-left corner
    foreground_img_float = cv2.imread('stroma_logo_expanded.png',-1).astype(float)

    # Foreground frame is generated with;
    # foreground_img = cv2.copyMakeBorder(src = resized, top=0, bottom=1005, left=0, right=1620, borderType=cv2.BORDER_CONSTANT,value=[0,0,0])



    # Blend images
    opacity = 1.0  # The opacity of the foreground that is blended onto the background is 70 %.
    alpha = np.full((1080,1920),255.0) # Fully opaque 4.th layer (alpha) for BGR frames --> BGRA

    while index != total_images:
        image_data = Data(args.root_dir, image_list[index])
        print(image_data.image_path)
        background_img_float = process_image(image_data)
        # if args.save_images:
        #     cv2.imwrite(os.path.join(args.save_dir, image_list[index] + ".jpg"), image)

        # b_channel, g_channel, r_channel = cv2.split(image)
        # alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 50
        # img_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
        # cv2.imwrite('temp.png',img_BGRA)
        #background_img_float = cv2.imread('temp.png',cv2.IMREAD_UNCHANGED).astype(float)
        background_img_float = np.dstack((background_img_float,alpha)) # Add the opacity layer to background image
                                                                       # background_img_float.shape --> (1080, 1920, 4)
        blended_img_float = blend_modes.normal(background_img_float, foreground_img_float, opacity) # Blend images with 'normal' mode with opacity 1.0
        #blended_img = np.uint8(blended_img_float)
        blended_img_uint8 = blended_img_float.astype(np.uint8)  # Convert image to OpenCV native display format

        out.write(blended_img_uint8[:,:,:3]) # Write frames with 3 layers [:,:,:3] --> (1080, 1920, 3) BGR format
        index = index + 1
        print(index)
        # cv2.imshow('image', image)
        # k = chr(cv2.waitKey())
        # if k == 'd':  # next
        #     index = index + 1 if index != total_images - 1 else 0
        # elif k == 'a':
        #     index = index - 1 if index != 0 else total_images - 1
        # elif k == 's':
        #     index = random.randint(0, total_images)
        # elif k == 'q':
        #     cv2.destroyAllWindows()
        #     cv2.waitKey(1)
        #     break
    out.release()


if __name__ == '__main__':
    main(args)
