# laneTracker
This is a project I am using to learn some more about C++ and computer vision.

The code is built on openCV 3.1.0

Use as laneTracker.exe path_to_input_video

The parameters for where the regions are located and what settings are used in the processing are hard coded right now. It is done in a way that should make reading from a file easy.

## Overview of how it functions
-A video stream is opened

-8 'regions of interest' are located based on some parameters (based roughly on the infinity point of the image)

-Each region is skeletonized

-A probablistic hough transform is done on the skeleton

-The results are drawn.




