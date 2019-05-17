import cv2
import numpy as np
import tensorflow as tf
import pickle


def main():
    import argparse
    # setting up arguments
    parser = argparse.ArgumentParser(
        description='Using tensorflow functions to augment video data with uniform treatments')
    parser.add_argument('--video_file', type=str, help='Your input video file', required=True)
    parser.add_argument('--aug', type=str, help='Your augmentation list', action='append', required=True)
    parser.add_argument('--crop_size', type=str, default='160x160')
    parser.add_argument('--resize', type=str, default='16x16')
    parser.add_argument('--demo', type=bool, default=False)
    parser.add_argument('--mean', type=float, default=50.0)
    args = parser.parse_args()
    # getting video in numpy and its shape
    np_video, shape = extract_frame(args)
    # converting np video into a list of augmented version, depending on the number of arguments given
    aug_list = augment(np_video, shape, args)

    # saving list of numpy array into pkl format file
    with open(f'augmented_video.pkl', 'wb') as f:
        pickle.dump(aug_list, f)


def extract_frame(args):
    # loading video using cv2
    video = cv2.VideoCapture(args.video_file)
    N = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    H = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

    # extracting frames from video as a 4-D np array [N, H, W, 3]
    np_video = np.empty((N, H, W, 3)).astype(np.uint8)
    fc = 0
    ret = True
    while fc < N and ret:
        ret, np_video[fc] = video.read()
        fc += 1
    video.release()

    shape = (N, H, W)
    return np_video, shape


def augment(np_video, this_shape, args):
    my_video = np_video
    shape = this_shape
    list_aug = args.aug
    counter = 0
    output_list = []

    '''
        list_aug ==> a list of string that represents augmentation arguments. For instance, ['rand_rot+rand_crop', 'rand_rot+rand_noise', 'rand_crop']
            
        output_list ==> a list of 4D numpy array that is then saved as .pkl file
    '''

    for this_arg in list_aug:
        # Here we split each argument, which could includes a combination of commands, into a list of commands
        # For instance, 'rand_crop+rand_rot' is converted to ['rand_crop', 'rand_rot']
        aug_commands = this_arg.split('+')
        for this_command in aug_commands:
            N, H, W = shape
            with tf.Session() as sess:
                inp_var = tf.placeholder(shape=[N, H, W, 3], dtype=tf.uint8)
                if this_command == 'rand_crop':
                    aug_video = crop(inp_var, shape, args)
                if this_command == 'rand_rot':
                    aug_video = rotate(inp_var, shape)
                if this_command == 'resize':
                    aug_video = resize(inp_var, args)
                if this_command == 'rand_noise':
                    aug_video = gaussian_noise(inp_var, shape, args)
                if this_command == 'rand_flip':
                    aug_video = flip(inp_var, shape)
                feed_dict = {inp_var: my_video}
                augmented_video = sess.run(aug_video, feed_dict=feed_dict)
                my_video = augmented_video
                shape = (augmented_video.shape[0], augmented_video.shape[1], augmented_video.shape[2])
        # Only if args.demo is True, we output the video file for checking purposes
        if args.demo:
            import imageio
            imageio.mimwrite(f'aug{counter}.mp4', augmented_video, fps=30, macro_block_size=1)
        # Otherwise, this 4D numpy array is appended to the list
        else:
            output_list.append(augmented_video)
        counter += 1
    return output_list


def crop(video, shape, args):
    N, H, W = shape
    crop_size = args.crop_size.split('x')
    target_h = int(crop_size[0])
    target_w = int(crop_size[1])
    # sampling crop size from uniform distribution
    off_h = tf.random.uniform([], minval=0, maxval=H - target_h, dtype=tf.int32)
    off_w = tf.random.uniform([], minval=0, maxval=W - target_w, dtype=tf.int32)
    # cropping video according to offset magnitudes
    crop_video = tf.image.crop_to_bounding_box(video, off_h, off_w, target_h, target_w)

    return crop_video


def rotate(video, shape):
    N, H, W = shape
    # sampling angles from normal distribution
    random_angles = tf.random.uniform([], minval=-np.pi / 4, maxval=np.pi / 4)
    # rotating video by using transformation
    rotated_video = tf.contrib.image.transform(
        video,
        tf.contrib.image.angles_to_projective_transforms(random_angles, tf.cast(H, tf.float32),
                                                         tf.cast(W, tf.float32)
                                                         ))
    return rotated_video


def gaussian_noise(video, shape, args):
    N, H, W = shape
    # sampling noise from normal distribution
    noise = tf.random_normal(shape=[N, H, W, 3], mean=args.mean, stddev=50 / 255, dtype=tf.float32)
    # adding noise to our video
    noise_video = tf.cast(video, dtype=tf.float32) + noise

    return tf.cast(noise_video, dtype=tf.uint8)


def resize(video, args):
    # converting string 'axb' to ['a','b']
    size = args.resize.split('x')
    # resizing video by user-specified ratio
    resized_video = tf.image.resize_images(video, tf.cast([int(size[0]), int(size[1])], tf.int32))
    return resized_video


def flip(video, shape):
    N, H, W = shape
    # slice each frame as a tensor of shape [1, H, W, 3], then squeeze it into [H, W, 3]
    my_video = tf.squeeze(tf.slice(video, [0, 0, 0, 0], [1, -1, -1, -1]))
    for n in range(N - 1):
        # slice and squeeze the next frame as above, then concat it to the previous frame
        # eventually, my_video will be of the shape [H, W, 3*N]
        my_video = tf.concat([my_video,
                              tf.squeeze(tf.slice(video, [n, 0, 0, 0], [1, -1, -1, -1]))]
                             , 2)
    # Flip operation is performed for H & W, since channels are not affected
    my_video = tf.image.random_flip_left_right(my_video)

    output_list = []
    for n in range(0, 3 * N, 3):
        # slicing previous tensor by each frame, then append it to the list
        output_list.append(tf.slice(my_video, [0, 0, n], [-1, -1, 3]))
    # stacking all tensors (frames) in the list to obtain the flipped video
    flipped_video = tf.stack(output_list)

    return flipped_video


if __name__ == '__main__':
    main()
