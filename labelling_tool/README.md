Launch the tool by specifying the number of images you want to analyze and the directory where images are located

```python label_images.py --dir path/to/images --batch_size 5```

# Procedure: 

* If the image **doesn’t contain area for PV** press **n** and skip it —> the image will be moved into the /noPV folder
* If the image does **contain area for PV**:
    * crop the area with the mouse functionalities (see below)
    * pressing **r** will **restore** the initial situation (in case of mistakes)
    * pressing **e** will **erase the last non-closed polygon** (extruded or not) (in case of mistakes)
    * once you are satisfied, press **v** to **visualise and finish** the cropped area which will be saved
    * press **any key** to go to the **next image** —> the image will be copied/ moved into the /PV folder and the labelled one into /label
* The scripts terminates automatically after the last image in the batch has been processed

# Mouse Functionalities
left —> draw lines  
right —> close the polygon  
central —> erase last non closed polygon  

SHIFT + left —> draw lines of polygon to be extruded  
SHIFT + right —> close the polygon to be extruded  
SHIFT + central —> erase last non closed polygon to be extruded  

## Normal polygons and extruded polygons
Draw normal polygons around areas where PV can be installed. If a part of the selected area is not suitable for installing PV (e.g. superstructure on a roof), you can simply extrude it by drawing an extruded polygon around it.

The script first will select all areas with normal polygons and fill them.  
Once this is done, the script extrudes all *extruded polygons*. So the order in which you draw polygons doesn't matter.
If you draw an extruded polygon partly or fully outside a normal polygon, the script ignores areas which can't be extruded.


# Others

To scan simply the previously labelled images to double check that the labelling has been done correctly:

python scan_images.py --dir /Users/robertocastello/deneb/labelling_tool/images_from_deneb/PV --outdir /Users/robertocastello/deneb/labelling_tool/images_from_deneb_final --batch_size 5

or to sort among different folders:

python scan_images.py --dir /Users/robertocastello/deneb/labelling_tool/SI_25_2013_1164-14/PV --outdir1 /Users/robertocastello/deneb/labelling_tool/almost_PV_deneb --outdir2 /Users/robertocastello/deneb/labelling_tool/PV_flat  --outdir3 /Users/robertocastello/deneb/labelling_tool/PV_slope --outdir4 /Users/robertocastello/deneb/labelling_tool/solar_thermal --batch_size 400
