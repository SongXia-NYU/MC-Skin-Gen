import os
import os.path as osp
from PIL import Image
from glob import glob
import shutil


def find_img_rec(folder_path, dst_path, tgt_size=(16, 16)):
    os.makedirs(dst_path, exist_ok=True)
    paths = glob(osp.join(folder_path, "*"))
    for p in paths:
        if osp.isdir(p):
            find_img_rec(p, dst_path, tgt_size)
        elif p.split(".")[-1] == "jpg":
            try:
                img = Image.open(p)
                if img.size == tgt_size:
                    shutil.copy(p, dst_path)
            except Exception:
                shutil.move(p, "mc_skin_problems")


if __name__ == '__main__':
    find_img_rec("mc_skins", "mc_skin_64", tgt_size=(64, 64))
