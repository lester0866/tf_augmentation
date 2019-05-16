import cv2
import numpy as np
import tensorflow as tf
import imageio
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
    aug_list = []
    for this_arg in list_aug:
        aug_commands = this_arg.split('+')
        for this_command in aug_commands:
            N, H, W = shape
            with tf.Session() as sess:
                inp_var = tf.placeholder(shape=[N, H, W, 3], dtype=tf.uint8)
                if this_command == 'rand_crop':
                    aug_video = crop(inp_var, shape, args)
                if this_command == 'rand_rot':
                    aug_video = rotate(inp_var, shape)
                if this_command == 'rand_resize':
                    aug_video = resize(inp_var, args)
                feed_dict = {inp_var: my_video}
                augmented_video = sess.run(aug_video, feed_dict=feed_dict)
                my_video = augmented_video
                shape = (augmented_video.shape[0], augmented_video.shape[1], augmented_video.shape[2])
        if args.demo:
            imageio.mimwrite(f'aug{counter}.mp4', augmented_video, fps=30, macro_block_size=1)
        else:
            aug_list.append(augmented_video)
        counter += 1
    return aug_list


def crop(video, shape, args):
    N, H, W = shape
    crop_size = args.crop_size.split('x')
    target_h = int(crop_size[0])
    target_w = int(crop_size[1])
    off_h = tf.random.uniform([], minval=0, maxval=H - target_h, dtype=tf.int32)
    off_w = tf.random.uniform([], minval=0, maxval=W - target_w, dtype=tf.int32)
    crop_video = tf.image.crop_to_bounding_box(video, off_h, off_w, target_h, target_w)

    return crop_video


def rotate(video, shape):
    N, H, W = shape
    random_angles = tf.random.uniform([], minval=-np.pi / 4, maxval=np.pi / 4)

    rotated_video = tf.contrib.image.transform(
        video,
        tf.contrib.image.angles_to_projective_transforms(random_angles, tf.cast(H, tf.float32),
                                                         tf.cast(W, tf.float32)
                                                         ))
    return rotated_video


def resize(video, args):
    size = args.resize.split('x')
    resized_video = tf.image.resize_images(video, tf.cast([int(size[0]), int(size[1])], tf.int32))
    return resized_video


if __name__ == '__main__':
    main()
