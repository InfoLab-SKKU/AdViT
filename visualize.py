import argparse
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Transformers')
    parser.add_argument('--img_path', default="", help='Image path (in npz format)')

    return parser.parse_args()

def resize(img, new_size=(224, 224)):
    img = np.uint8(255 * img.transpose([1, 2, 0]))
    img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)
    img = img.transpose([2, 0, 1])
    return np.float32(img / 255.)
def plot(img, heatmap):
#     m1 = np.uint8(255 * cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_LINEAR))
    m1 = np.uint8(heatmap)
    m1 = cv2.applyColorMap(m1, cv2.COLORMAP_JET)
    m1 = np.float32(m1 / 255.).transpose([2, 0, 1])[::-1]
    m1 = (img + m1)
    m1 = m1 / m1.max()
    return np.float32(m1).transpose((1,2,0))

def visualize():
    # setup run
    args = parse_args()

    if not args.img_path.endswith('.npz'):
      print('Invalid data format for visualization')
      return
    
    if not os.path.isfile(args.img_path):
      print(f'Image not found in the directory: {args.img_path}')
      return

    img = np.load(args.img_path)

    f, axarr = plt.subplots(2,2, figsize=(8, 8))
    axarr[0,0].imshow(img['img_x'].transpose((1,2,0)))
    axarr[0,0].set_title("Benign Image", fontdict=None, loc='center', color = "k")
    axarr[0,1].imshow(plot(img['img_x'], img['ben_cam'].squeeze()))
    axarr[0,1].set_title("Benign Interpretation", fontdict=None, loc='center', color = "k")
    axarr[1,0].imshow(img['adv_x'].transpose((1,2,0)))
    axarr[1,0].set_title("Adversarial Image", fontdict=None, loc='center', color = "k")
    axarr[1,1].imshow(plot(img['adv_x'], img['adv_cam'].squeeze()))
    axarr[1,1].set_title("Adversarial Interpretation", fontdict=None, loc='center', color = "k")


    name = args.img_path.split('/')[-1].split('.')[0]
    plt.savefig(f'visualization/{name}.png')
    



if __name__ == '__main__':
    visualize()
