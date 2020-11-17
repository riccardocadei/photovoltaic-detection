from crop import *
from move import *
from imutils import paths
import argparse
import os

parser = argparse.ArgumentParser(description='Labeling tools')
parser.add_argument('--dir', help='Directory containing the input images')
parser.add_argument('--outdir1', help='Directory 1 containing the selected images (must be almost PV)')
parser.add_argument('--outdir2', help='Directory 2 containing the selected images')
parser.add_argument('--outdir3', help='Directory 3 containing the selected images')
parser.add_argument('--outdir4', help='Directory 4 containing the selected images')
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

## outputdir1 
outputdir1 = args['outdir1']
almostPVout_dir = outputdir1 + '/PV'
almostPVout_label_dir = outputdir1 + '/PV'+'/labels'

## outputdir2   
outputdir2 = args['outdir2']
PVout_dir2 = outputdir2 + '/PV'
PVout_label_dir2 = outputdir2 + '/PV'+'/labels'

## outputdir3
outputdir3 = args['outdir3']
PVout_dir3 = outputdir3 + '/PV'
PVout_label_dir3 = outputdir3 + '/PV'+'/labels'

## outputdir4
outputdir4 = args['outdir4']
PVout_dir4 = outputdir3 + '/PV'
PVout_label_dir4 = outputdir3 + '/PV'+'/labels'

#Create the directories if they do not exist
#if not os.path.exists(PV_dir):
#    os.makedirs(PV_dir)
#if not os.path.exists(PV_label_dir):
#    os.makedirs(PV_label_dir)

if not os.path.exists(PVout_dir2):
    os.makedirs(PVout_dir2)
if not os.path.exists(PVout_label_dir2):
    os.makedirs(PVout_label_dir2)

if not os.path.exists(PVout_dir3):
    os.makedirs(PVout_dir3)
if not os.path.exists(PVout_label_dir3):
    os.makedirs(PVout_label_dir3)

if not os.path.exists(PVout_dir4):
    os.makedirs(PVout_dir4)
if not os.path.exists(PVout_label_dir4):
    os.makedirs(PVout_label_dir4)

if not os.path.exists(almostPVout_dir):
    os.makedirs(almostPVout_dir)
if not os.path.exists(almostPVout_label_dir):
    os.makedirs(almostPVout_label_dir)

for image in imagePaths[0:args["batch_size"]]:
#    _ = input('Do you want to quit?')
#    if _ == 'y':
#        break
#    else:
    if( not (image.endswith('.DS_Store') or image.startswith('._SI')) ):
        print(image)
        #print(PVout_dir)
        #print(PVout_label_dir)
        #print(almostPVout_dir)
        #print(almostPVout_label_dir)
        m = Move(image,outputdir1,outputdir2,outputdir3,outputdir4)
        m.do_move() 
