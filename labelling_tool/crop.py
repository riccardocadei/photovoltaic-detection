import cv2
import numpy as np
import sys
import shutil
import os

class Crop():

    def __init__(self, image_name):

        self.image_name = image_name
        
        #####Filled polygons
        #lpnts : polygons being built
        self.lpnts = np.empty((1,0,2), dtype=np.int32)
        #rpnts : ready polygons
        self.rpnts = []
        
        #####Extruded polygons
        #lextpnts: extruded polygons being built
        self.lextpnts = np.empty((1,0,2), dtype=np.int32)
        #rextpnts : ready extruded polygons
        self.rextpnts = []

        self.image = cv2.imread(self.image_name,-1)
        self.copied_image = self.image.copy()

    def crop_for_mouse(self,event,x,y,flags= None,parameters = None):

        self.event = event
        self.x = x
        self.y = y
        self.flags = flags
        self.parameters = parameters
        SHIFT_FLAG = 16 #this may vary on different OS/version of openCV
        
        #add point to polygon being built
        if self.event == cv2.EVENT_LBUTTONDOWN:
            if (SHIFT_FLAG == (flags & SHIFT_FLAG)):
                self.lextpnts = np.append(self.lextpnts, np.array([[[self.x, self.y]]]), axis=1)
                cv2.polylines(self.image, [self.lextpnts], False, (0, 0, 255))
            else:
                self.lpnts = np.append(self.lpnts, np.array([[[self.x, self.y]]]), axis=1)
                cv2.polylines(self.image, [self.lpnts], False, (0, 255, 0))
        
        #add last point to polygon being built (close the polygon)
        elif self.event == cv2.EVENT_RBUTTONDOWN:
            if (SHIFT_FLAG == (flags & SHIFT_FLAG)):
                self.lextpnts = np.append(self.lextpnts, np.array([[[self.x, self.y]]]), axis=1)
                cv2.polylines(self.image, [self.lextpnts], True, (0, 0, 255))
                self.rextpnts.append(self.lextpnts)
                self.lextpnts = np.empty((1,0,2), dtype=np.int32)
                self.copied_image = self.image.copy()
            else:
                self.lpnts = np.append(self.lpnts, np.array([[[self.x, self.y]]]), axis=1)
                cv2.polylines(self.image, [self.lpnts], True, (0, 255, 0))
                self.rpnts.append(self.lpnts)
                self.lpnts = np.empty((1,0,2), dtype=np.int32)
                self.copied_image = self.image.copy()
        
        #erase polygon being built
        elif self.event == cv2.EVENT_MBUTTONDOWN: 
            if (SHIFT_FLAG == (flags & SHIFT_FLAG)):
                self.lextpnts = np.empty((1,0,2), dtype=np.int32)
                self.image = self.copied_image.copy()
            else:
                self.lpnts = np.empty((1,0,2), dtype=np.int32)
                self.image = self.copied_image.copy()
            

    def do_crop(self):

        cv2.namedWindow("CROP", cv2.WINDOW_NORMAL) ## Magnifying the window for more precise labelling 
        cv2.resizeWindow("CROP", 800, 800) ## 1250
        cv2.setMouseCallback("CROP",self.crop_for_mouse)
        self.skip = False

        while True:

            cv2.imshow("CROP",self.image)
            cv2.setMouseCallback("CROP",self.crop_for_mouse)
            keypress = cv2.waitKey(1)
            
            #(e)rase last non closed polygon/extruded polygon
            if keypress == ord('e'):
                self.image = self.copied_image.copy() ## RC -- this command doesn't really work in practice
                self.lpnts = np.empty((1,0,2), dtype=np.int32)
                self.lextpnts = np.empty((1,0,2), dtype=np.int32)
            
            #(r)estore initial situation
            if keypress == ord('r'):
                self.image = cv2.imread(self.image_name,-1)
                self.copied_image = self.image.copy()
                self.lpnts = np.empty((1,0,2), dtype=np.int32)
                self.rpnts = []
                self.lextpnts = np.empty((1,0,2), dtype=np.int32)
                self.rextpnts = []

            #(n)ext file
            if keypress == ord('n'):
               self.skip = True
               break
           
            #(v)izualize result and go to next file
            if keypress == ord('v'):
               self.skip = False
               break
        
        if self.skip is False:
          
            image_path = self.image_name.split(os.path.sep)
            image_orig_path = os.sep.join(image_path[:-1]) + '/PV/' + image_path[-1]
            print("Saving original image to ", image_orig_path)
            shutil.move(self.image_name, image_orig_path) ## RC comment: once final it should become  --> shutil.move(self.image_name, image_path) ## shutil.copyfile

            mask  = np.zeros(self.image.shape, dtype=np.uint8)
            channel_count = self.image.shape[2]
            #white in output mask (i.e available for PV)
            ignore_mask_color = (255,)*channel_count
            #black in output mask (i.e not available for PV)-> used for extruded polygons
            extrude_mask_color = (0,)*channel_count
            
            for point in self.rpnts:
                cv2.fillPoly(mask, point, ignore_mask_color)
            
            #Extrude polygons
            for point in self.rextpnts:
                cv2.fillPoly(mask, point, extrude_mask_color)

            masked_image = cv2.bitwise_and(self.image,mask)

            image_path = os.sep.join(image_path[:-1]) + '/PV/labels/'\
                         + image_path[-1][:-4] + '_label' + \
                         image_path[-1][-4:] 
             
            print("Saving labelled image to ", image_path) 
            cv2.imshow("ROI", masked_image)
            cv2.imwrite(image_path, mask) ## Writing the B&W mask (instead of masked_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()	
        else:
            image_path = self.image_name.split(os.path.sep)
            image_path = os.sep.join(image_path[:-1]) + '/noPV/' + image_path[-1]
            print("Saving original image to ", image_path)
            shutil.move(self.image_name, image_path) ## RC comment: once final it should become  --> shutil.move(self.image_name, image_path)
