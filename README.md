# ML2020 Project 2 Detecting available rooftop area for PV installation with LESO-PB lab

The repository contains the code for Machine Learning course 2020 (CS-433) project 2 at EPFL. More information about this project can be found in the folder `documents`.
* * *
### General Information

### Team
The project is accomplished by team `OverfitTeam` with members:
- Riccardo Cadei: [@riccardocadei](https://github.com/riccardocadei)
- Raphael Attias: [@raphaelattias](https://github.com/raphaelattias)
- Shasha Jiang: [@dust629](https://github.com/dust629)

### Environment
The project has been developed and test with `python3.6`.
The required library are `numpy, Pytorch, sklearn, openCV`
The library for visualization is `matplotlib`.

* * *
## Project Information

### Topic: Detecting available rooftop area for PV installation

The project target is to segment in aerial images of Switzerland(Geneva) the area available for the installation of rooftop photovoltaics (PV) panels, namely the area we have on roofs after excluding chimneys, windows, existing PV installations and other so-called ‘superstructures’. The task is a pixel-wise binary-semantic segmentation problem. And we are interested in the class where pixels can be classified as ‘suitable area’ for PV installations.

![Screenshot from 2020-12-16 13-11-43](https://user-images.githubusercontent.com/32882147/102347151-47643980-3fa0-11eb-83c7-354c90462914.png)

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
In particular we are able to automatically detect in test images of residential areas the available rooftop area at pixel level with an accuracy of about 0.97 and an Intersection over Union index of 0.77 using only 244 images in the training. 

![iou_batch5loss6](https://user-images.githubusercontent.com/32882147/102346625-7201c280-3f9f-11eb-9bb8-244ac7348d91.png)

* * *
## Project structure

├── labelling_tool
│   ├── crop.py
│   ├── data-verification.ipynb
│   ├── label_images_from_txt.py
│   ├── label_images.py
│   ├── move.py
│   ├── README.md
│   └── scan_images.py
├── loss
│   ├── 1.png
│   ├── 2.png
│   ├── loss.ipynb
│   ├── loss.py
│   └── __pycache__
│       └── loss.cpython-38.pyc
├── main.ipynb
├── model
│   ├── model.ipynb
│   ├── __pycache__
│   │   └── unet.cpython-38.pyc
│   └── unet.py
├── plots
│   ├── all
│   │   ├── batch5loss4
│   │   │   ├── history_train_ioubatch5loss4_1000.npy
│   │   │   ...
│   │   │   └── loss400_batch5loss4.png
│   │   └── batch5loss9
│   │       ├── history_train_ioubatch5loss9_1000.npy
│   │       ... 
│   │       └── loss1000_batch5loss9.png
│   ├── other
│   │   ├── b5w4_iou.pdf
│   │   ├── b5w4_loss.pdf
│   │   ├── history_train_iou09122020.npy
│   │   ...
│   │   ├── loss_batch5loss4.pdf
│   │   └── loss_batch5loss5.pdf
│   ├── plots.py
│   └── residencial
│       ├── history_train_ioubatch5loss6_1000.npy
│       ...
│       ├── loss1000_batch5loss6.png
│       └── loss_batch5loss6.png
├── process_data
│   ├── data_loader.py
│   ├── import_test.py
│   ├── normalize.py
│   └── __pycache__
│       ├── data_loader.cpython-38.pyc
│       ├── data_noara_loader.cpython-38.pyc
│       └── data_nopv_loader.cpython-38.pyc
├── README.md
├── reference
│   └── Literature
│       ├── Adam a method for stochastic optimization.pdf
│       ├── Deep learning in the built environment automatic detection of rooftop solar panels using Convolutional Neural Networks.pdf
│       ├── Dropout vs. batch normalization an empirical study.pdf
│       ├── Satellite Image Segmentation for Building Detection using U-Net.pdf
│       ├── Semantic Segmentation of Satellite Images using Deep Learning.pdf
│       └── U-Net, Convolutional Networks for Biomedical Image Segmentation.pdf
├── run.py
├── test.png
└── train
    ├── pred_residencial_3.png
    └── train.py



### Report

`documents/report.pdf`: a 4-pages report of this project

