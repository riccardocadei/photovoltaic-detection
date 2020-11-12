from crop import *
from imutils import paths
import argparse
import os

parser = argparse.ArgumentParser(description='Labeling tools')
parser.add_argument('--dir', help='Directory containing the images')
parser.add_argument('--batch_size', type=int, help='Number of images to be processed\
                    starting from 0')

args = vars(parser.parse_args())
imagePaths = sorted([os.path.join(args['dir'], f) for f in os.listdir(args['dir']) \
                     if os.path.isfile(os.path.join(args['dir'], f))])
#imagePaths = sorted(list(paths.list_images(args["dir"])))

imagedir = imagePaths[0].split(os.path.sep)
imagedir = os.sep.join(imagedir[:-1])
PV_dir = imagedir + '/PV'
PV_label_dir = imagedir + '/PV'+'/labels'
NOPV_dir = imagedir + '/noPV'

#Create the directories if they do not exist
if not os.path.exists(PV_dir):
    os.makedirs(PV_dir)
if not os.path.exists(PV_label_dir):
    os.makedirs(PV_label_dir)
if not os.path.exists(NOPV_dir):
    os.makedirs(NOPV_dir)


for image in imagePaths[0:args["batch_size"]]:
#    _ = input('Do you want to quit?')
#    if _ == 'y':
#        break
#    else:
    if(not image.endswith('.DS_Store')):
        print(image)
        crop = Crop(image)
        crop.do_crop() 
