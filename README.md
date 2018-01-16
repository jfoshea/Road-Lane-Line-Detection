# Finding Lane Lines on the Road 

## Overview
A self driving car needs to be able to navigate a road using computer vision.  This project documents a method for finding lanes on the road using OpenCV, python, numpy, and matplotlib. The pipeline code is executed within in a jupyter notebook environment. The project has two major sections; The first section is to develop a processing pipeline on a series of test images, and the pipeline will process each image and mark the lane lines on the road. The second second section is to take the pipeline algorithm and apply it to a set of videos.

## Installing and Running the Pipeline
The following steps are used to run the pipeline:
1. Install jupyter notebook environment and packages
    ```
    https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/doc/configure_via_anaconda.md
    ```
2. Clone the SDC-FindingLaneLines git repository
    ```  
    $  git clone https://github.com/jfoshea/SDC-FindingLaneLines.git
    ```

3. enable cardnd-term1 virtualenv
    ```
    $ source activate carnd-term1
    ```
4. Run the Pipeline 
    ```
    $ jupyter notebook FindingLaneLines.ipynb
    ```

The output images are located in `test_images_output` directory.
The output videos are located in `test_videos_output` directory.

## Writeup 
A detailed writeup of the pipeline construction and challenges are located here [writeup] (writeup.md)

