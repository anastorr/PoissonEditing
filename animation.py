from PIL import Image
import os


def create_gif(path, out):
    imgs = [Image.open(path + file_name) for file_name in os.listdir(path)]
    img = imgs[0]
    img.save(fp=out, format='GIF', append_images=imgs,
             save_all=True, duration=200, loop=0)

create_gif('images/example5/progress/', 'images/example5/progress.gif')