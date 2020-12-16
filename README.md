# ML2020 Project 2 Detecting available rooftop area for PV installation with LESO-PB lab

The repository contains the code for Machine Learning course 2020 (CS-433) project 2 at EPFL. More information about this project can be found in the folder `documents`.
* * *
### General Information

### Team
The project is accomplished by team `OverfitTeam` with members:
- Riccardo Cadei: [@RiccardoCadei](https://github.com/RiccardoCadei)
- Raphael Attias: [@raphaelattias](https://github.com/raphaelattias)
- Shasha Jiang: [@dust629](https://github.com/dust629)

### Environment
The project has been developed and test with `python3.6`.
The required library are `numpy, Pytorch, sklearn`
The library for visualization is `matplotlib`.

* * *
## Project Information

### Topic: Detecting available rooftop area for PV installation

The project target is to segment in aerial images of Switzerland(Geneva) the area available for the installation of rooftop photovoltaics (PV) panels, namely the area we have on roofs after excluding chimneys, windows, existing PV installations and other so-called ‘superstructures’. The task is a pixel-wise binary-semantic segmentation problem. And we are interested in the class where pixels can be classified as ‘suitable area’ for PV installations.

![Screenshot from 2020-12-16 12-09-55](https://user-images.githubusercontent.com/32882147/102341568-4fb87680-3f98-11eb-9eba-ff2d7cfa2d7e.png)
![Screenshot from 2020-12-16 12-36-22](https://user-images.githubusercontent.com/32882147/102343832-5694b880-3f9b-11eb-8ba5-87c1ee43c33b.png)


### Data
- The input aerial images are RGB aerial images in PNG form and  each  image  has  size 250×250×3 with pixelsize 0.25×0.252. 
- The original input images are transformed with saturation and classic normalization before training. 
- A real-time data argumentation are applies only on training set by randomly flipping images horizontally or vertically or rotating in ninety degrees.
- We used the provided labelling tool to manually label all the data The labelled images are a binary mask with 1 for pixel in PV area , and 0 otherwise.
- The  output  of  our  model  isagain a binary image, where the pixel is one, if its probability ofbeing in the PV area is bigger than the threshold.
- Train/Validation/Test Ratio : 80/10/10

![intro](https://user-images.githubusercontent.com/32882147/102341360-0a944480-3f98-11eb-8970-9ddbd0277339.jpeg)

### Methods
- We used Conventional Neural Network(CNN) model base on U-net and adaptive learning to train our model. Iou and Acurrancy are computed to evaluat the performance.
- We trained our model firstly on the whole dataset, then we focused only on a specific class of images, residential area

### Results
In particular we are able to automatically detect in test images of residential areas the available rooftop area at pixel level with an accuracy of about 0.97 and an Intersection over Union index of up to 0.77 using only 244 images in the training. 

* * *
## Project structure

### process_data 
### model
### optimizer
### loss
### train
### hyperparameters
### Notebook



### Report

`documents/report.pdf`: a 2-pages report about how we finished the project in detail.

