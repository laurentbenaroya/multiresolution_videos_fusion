import cv2
import numpy as np
from os.path import join
import os
from glob import glob
from tqdm import tqdm
import argparse

from utils import video2frames
from multiresolution import build_laplacian_pyramid, reconstruct_from_laplacian_pyramid

"""
multi-resolution fusion of two videos with different static weights at each level
"""


def get_args():
    parser = argparse.ArgumentParser(description='Fusion two sets of images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--video1', help='path to video 1', required=True, type=str)
    parser.add_argument('--video2', help='path to video 2', required=True, type=str)
    parser.add_argument('--output', help='path to output folder', required=True, type=str)
    parser.add_argument('--alphaR', nargs='+', help='RED, list multi-resolution alpha values for video 1',
                        required=True, type=float)
    parser.add_argument('--alphaG', nargs='+', help='GREEN, list multi-resolution alpha values for video 1',
                        required=True, type=float)
    parser.add_argument('--alphaB', nargs='+', help='BLUE, list multi-resolution alpha values for video 1',
                        required=True, type=float)
    args = parser.parse_args()
    for val in args.alphaR:
        assert 0 <= val <= 1., 'alpha values must be between 0 and 1'
    for val in args.alphaG:
        assert 0 <= val <= 1., 'alpha values must be between 0 and 1'
    for val in args.alphaB:
        assert 0 <= val <= 1., 'alpha values must be between 0 and 1'
    return args


def get_image2(image1_path, images2):
    """ get image2 with max correlation with image1"""
    image1 = cv2.imread(image1_path).flatten()
    N2 = len(images2)
    corr = []
    for image2 in images2:
        image2 = cv2.imread(image2).flatten()
        corr.append(float(np.corrcoef(image1, image2)[0, 1]))
    corr = np.array(corr)
    # print(corr)
    print(f'arg max correlation: {np.argmax(corr)}')
    return images2[np.argmax(corr)]


def main(args):
    """ multi-resolution fusion of two videos"""
    # create tmp folders
    os.makedirs('./tmp1', exist_ok=True)
    for img in os.listdir('./tmp1'):
        os.remove(f'./tmp1/{img}')
    os.makedirs('./tmp2', exist_ok=True)
    for img in os.listdir('./tmp2'):
        os.remove(f'./tmp2/{img}')
    # convert videos to images
    video2frames(args.video1, './tmp1')
    video2frames(args.video2, './tmp2')

    # read and sort image lists
    images1 = glob(join('./tmp1', '*.png'))
    images1 = sorted(images1, key=lambda x: int(x.split('/')[-1].split('.')[0]))
    images2 = glob(join('./tmp2', '*.png'))
    images2 = sorted(images2, key=lambda x: int(x.split('/')[-1].split('.')[0]))
    num_images = min(len(images1), len(images2))
    images1 = images1[:num_images]
    images2 = images2[:num_images]

    # iterate over images pairs
    for count, (image1_path, image2_path) in tqdm(enumerate(zip(images1, images2)), total=num_images):
        # Load an image
        image1 = cv2.imread(image1_path)
        assert image1 is not None, 'No image1 found'

        # Load an image
        image2 = cv2.imread(image2_path)
        assert image2 is not None, 'No image2 found'

        # Fusion
        [L, H, C] = image1.shape
        if count == 0:
            print(f'Image 1: {L}x{H}x{C}')
        [L2, H2, C2] = image2.shape
        if count == 0:
            print(f'Image 2: {L2}x{H2}x{C2}')
        assert L == L2 and H == H2 and C == C2, 'Images must have the same size'
        assert C == 3, 'Images must be RGB'
        # convert to float and normalize and reshape
        image1 = image1.astype(np.float32)
        image1 = image1 / 255.0

        image2 = image2.astype(np.float32)
        image2 = image2 / 255.0

        fused_image = np.zeros((L, H, C))
        # levels of the laplacian pyramid
        n_levels = len(args.alphaR)-1
        # loop over channels
        for c in range(C):
            beta = args.alphaR if c == 0 else args.alphaG if c == 1 else args.alphaB
            # build laplacian pyramids, encode
            lap_pyr_image1 = build_laplacian_pyramid(image1[:, :, c], levels=n_levels)
            lap_pyr_image2 = build_laplacian_pyramid(image2[:, :, c], levels=n_levels)
            # fusion
            fused_image_c = [(beta[level] * pyr_image1 + (1.-beta[level]) * pyr_image2)
                             for level, (pyr_image1, pyr_image2) in enumerate(zip(lap_pyr_image1, lap_pyr_image2))]
            # decode
            fused_image_c = reconstruct_from_laplacian_pyramid(fused_image_c)
            fused_image[:, :, c] = fused_image_c

        # save fused image
        fused_image = fused_image * 255.0
        fused_image = fused_image.astype(np.uint8)
        cv2.imwrite(join(args.output, f'fused_{count}.png'), fused_image)
    # convert images to video
    cmd = (f'/usr/bin/ffmpeg -i {args.output}/fused_%d.png -y -hide_banner -loglevel panic -c:v libx264 -r 25 '
           f'{args.output}/fused_video.mp4')
    print(cmd)
    os.system(cmd)
    # remove output png images
    for img in os.listdir(args.output):
        if img.endswith('.png'):
            os.remove(join(args.output, img))
    # remove tmp folders
    for img in os.listdir('./tmp1'):
        os.remove(f'./tmp1/{img}')
    os.rmdir('./tmp1')
    for img in os.listdir('./tmp2'):
        os.remove(f'./tmp2/{img}')
    os.rmdir('./tmp2')


if __name__ == '__main__':
    args = get_args()
    os.makedirs(args.output, exist_ok=True)
    # display_videos('video/Attal_wav2lip_gan.mp4', './video/fused_video.mp4', 'video/Attal.mp4')
    main(args)




