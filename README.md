# TensorFlow-based Video Augmentation

This script provides tensorflow-accelerated augmentations. It takes `cv2.VideoCapture supported` files as input 
and outputs a pkl file that contains a list of 4D numpy array(s) of shape (N, H, W, 3). 

It can be used to output videos that are augmented with different combinations of augmentation commands. For instance,
users can specify `rand_crop + rand_rototation` for first output and `rand_noise + rand_flip` for second output. 
It also supports visualizing the augmented output, saving it as mp4 file.

| Supported Augmentations | Customizable? |
| --- | --- |
| Random Crop | Yes - output crop size|
| Random Rotation | No |
| Random Gaussian Noise | Yes - gaussian mean |
| Random Left_Right Flipping | No |
| User-specified Resizing| Yes |

## How to run
  - **Environment Configuration**
  
| Packages |
| --- | 
| opencv |
| numpy | 
| tensorflow |
| imageio - if want to visualize |


  ```bash
  python tf_augmentation --video_file [path_to_video] --aug [aug commands] --crop_size[default: 160x160] --size[default: 16x16] --mean[default: 50] --demo[default:False] 
  ```

  Example (will produce 3 augmented videos):
  
  Note 1: if `rand_crop`, `rand_noise`, or `resize` is mentioned in --aug, a size must be given in `--crop_size`, `--mean`, `--resize` for customized values, otherwise default values will be used
  
  Note 2: if `--demo` is not set to True, only .pkl file will be outputted (no video file).
  
  ```bash
  python tf_augmentation --video_file [path] --aug rand_crop+rand_rot rand_flip+rand_noise rand_crop+resize --crop_size 160x160 --resize 16x16 --demo True
  ```

