import glob

import cv2

from gan_mc import pil_loader

if __name__ == '__main__':
    files = glob.glob("mc_skin_generated/generated-images-*.png")
    files.sort()

    frame = cv2.imread(files[0])
    height, width, layers = frame.shape

    frameSize = (width, height)
    out = cv2.VideoWriter('mc_gen.avi', 0, 2, frameSize)

    for filename in files:
        img = cv2.imread(filename)
        out.write(img)

    cv2.destroyAllWindows()
    out.release()
