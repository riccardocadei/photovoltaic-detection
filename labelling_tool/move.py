import cv2
import numpy as np
import sys
import shutil
import os

class Move():

    def __init__(self, image_name, output_dir_name1,output_dir_name2,output_dir_name3,output_dir_name4):

        self.image_name = image_name
        self.output_dir_name1 = output_dir_name1
        self.output_dir_name2 = output_dir_name2
        self.output_dir_name3 = output_dir_name3
        self.output_dir_name4 = output_dir_name4

        self.image = cv2.imread(self.image_name,-1)
        self.copied_image = self.image.copy()

    def do_move(self):

        cv2.namedWindow("MOVE", cv2.WINDOW_NORMAL) ## Magnifying the window for more precise labelling 
        cv2.resizeWindow("MOVE", 1000, 1000) ## 1250
        self.move_im_1 = False
        self.move_im_2 = False
        self.move_im_3 = False
        self.move_im_4 = False

        while True:

            cv2.imshow("MOVE",self.image)
            #cv2.setMouseCallback("CROP",self.crop_for_mouse)
            keypress = cv2.waitKey(1)

            if keypress == ord('n'):
                self.move_im_1 = True
                break
            if keypress == ord('f'):
                self.move_im_2 = True
                break
            if keypress == ord('s'):
                self.move_im_3 = True
                break
            if keypress == ord('t'):
                self.move_im_4 = True
                break
        
        if self.move_im_1 is True:
          
            image_path = self.image_name.split(os.path.sep)
            image_dest_path = self.output_dir_name1 + '/PV/' + image_path[-1]
            print("Copying original image ", self.image_name, " to ", image_dest_path)
            shutil.copy(self.image_name, image_dest_path)

            label_path = os.sep.join(image_path[:-1]) + '/labels/'\
                         + image_path[-1][:-4] + '_label' + \
                         image_path[-1][-4:]
 
            label_dest_path = self.output_dir_name1 + '/PV/labels/'\
                         + image_path[-1][:-4] + '_label' + \
                         image_path[-1][-4:]
             
            print("Copying labelled image ", label_path, " to ", label_dest_path) 
            shutil.copy(label_path, label_dest_path)

        elif self.move_im_2 is True:

            image_path = self.image_name.split(os.path.sep)
            image_dest_path = self.output_dir_name2 + '/PV/' + image_path[-1]
            print("Copying original image ", self.image_name, " to ", image_dest_path)
            shutil.copy(self.image_name, image_dest_path)

            label_path = os.sep.join(image_path[:-1]) + '/labels/'\
                         + image_path[-1][:-4] + '_label' + \
                         image_path[-1][-4:]

            label_dest_path = self.output_dir_name2 + '/PV/labels/'\
                         + image_path[-1][:-4] + '_label' + \
                         image_path[-1][-4:]

            print("Copying labelled image ", label_path, " to ", label_dest_path)
            shutil.copy(label_path, label_dest_path)

        elif self.move_im_3 is True:

            image_path = self.image_name.split(os.path.sep)
            image_dest_path = self.output_dir_name3 + '/PV/' + image_path[-1]
            print("Copying original image ", self.image_name, " to ", image_dest_path)
            shutil.copy(self.image_name, image_dest_path)

            label_path = os.sep.join(image_path[:-1]) + '/labels/'\
                         + image_path[-1][:-4] + '_label' + \
                         image_path[-1][-4:]

            label_dest_path = self.output_dir_name3 + '/PV/labels/'\
                         + image_path[-1][:-4] + '_label' + \
                         image_path[-1][-4:]

            print("Copying labelled image ", label_path, " to ", label_dest_path)
            shutil.copy(label_path, label_dest_path)

        elif self.move_im_4 is True:

            image_path = self.image_name.split(os.path.sep)
            image_dest_path = self.output_dir_name4 + '/PV/' + image_path[-1]
            print("Copying original image ", self.image_name, " to ", image_dest_path)
            shutil.copy(self.image_name, image_dest_path)

            label_path = os.sep.join(image_path[:-1]) + '/labels/'\
                         + image_path[-1][:-4] + '_label' + \
                         image_path[-1][-4:]

            label_dest_path = self.output_dir_name4 + '/PV/labels/'\
                         + image_path[-1][:-4] + '_label' + \
                         image_path[-1][-4:]

            print("Copying labelled image ", label_path, " to ", label_dest_path)
            shutil.copy(label_path, label_dest_path)


        else:
            print("No folder assigned, exiting...")

