# Augmentation
This small script for rotate and change brightness on input images with bounding boxes in pickle file.
Typicle case of use is ML for CNN and computer vision image augmentation. 
Script uses multiprocessing and can adjust throughput.

Input params:
* aug_prob = 0.5 - probability of augmentation
* angle_clock = 359 - rotation angle clockwise
* angle_counter_clock = 100 - rotation angle counter clockwise
* bright = (0, 255) - brightness interval
* draw_bb = True - draw bounding box on output images
* max_img_per_sec = 10 - throughput of input images

\#augmentation \#bounding_box \#multiprocessing \#rated_semaphore \#rotation_matrix \#opencv \#rotation \#pickle \#python
