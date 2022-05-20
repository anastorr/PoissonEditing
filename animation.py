from PIL import Image
import os


def create_gif(path, out):
    imgs = [Image.open(path + file_name) for file_name in os.listdir(path)]
    img = imgs[0]
    img.save(fp=out, format='GIF', append_images=imgs,
             save_all=True, duration=200, loop=0)

create_gif('multiresolution/images/example4/progress/', 'multiresolution/images/example4/progress.gif')