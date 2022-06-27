from PIL import Image
import os


def create_gif(path, out):
    path_sorted = sorted(os.listdir(path))
    imgs = [Image.open(path + file_name) for file_name in path_sorted]
    img = imgs[0]
    img.save(fp=out, format='GIF', append_images=imgs,
             save_all=True, duration=150, loop=0)

create_gif('poisson/images/example12/progress/', 'poisson/images/example12/progress.gif')