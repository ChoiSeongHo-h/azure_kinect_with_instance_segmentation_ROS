# azure_kinect_with_instance_segmentation_ROS
- Azure Kinect ROS publisher adding semantic information to point clouds
- Azure Kinect driver required
- Based on Azure Kinect ROS Driver (https://github.com/microsoft/Azure_Kinect_ROS_Driver)
- Tested on Ubuntu 20.04

## System
![image](https://github.com/ChoiSeongHo-h/azure_kinect_with_semantic_segmentation_ROS/assets/72921481/bfc549a9-c322-4961-a107-21246675379b)
1. subscribe a single segmented image with instance information and a single segmented image with class information.
2. publish a point cloud with instance information and class information.
- Two input segmented images are 8-bit single channels.
- Two input segmented image size: 640*480
- The output point cloud is an RGB point cloud.
- The G channel of an RGB point contains information about the instance.
- The B channel of an RGB point contains information about the class.

### Managing resources
- The system uses a queue to match past segmentation information with past depth information.
- The queue stores up to 10 depth information, and space for all 10 images is pre-allocated on the heap.
- The mapping from every pixel in the RGB image to every pixel in the depth image is cached.

## Usage
- Install this package instead of the Azure Kinect ROS Driver (https://github.com/microsoft/Azure_Kinect_ROS_Driver).
- Use the `seg_point_cloud = true` parameter to run.

# YOLOv8 as an instance segmentation model
- Set up the inputs for azure_kinect_with_instance_segmentation_ROS using YOLOv8.
## Merging segmentation information
- The number of instance segmentation output layers in YOLOv8 is equal to the number of inferred objects.
- Merge multiple output layers to create a single segmented instance image and a single segmented class image.
- The masks of the output layers overlap.
- Combining layers in an arbitrary order without handling overlaps creates class ambiguity in the overlaps.
- Suppressing low-probability inferences with high-probability inferences reduces ambiguity.
- The implementation sorts the inferred objects by probability and merges the output layers in the sorted order to produce the final output.
- The final result is a erosion algorithm applied to reduce false segmentation of the border.
## Usage
- Install YOLOv8 (https://github.com/ultralytics/ultralytics)
- `python3 seg_node.py`
