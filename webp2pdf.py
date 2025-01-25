import os
from PIL import Image
from PyPDF2 import PdfMerger
import re
import subprocess
import numpy as np


def get_chapter_page(t):
    c_p = re.findall("\d+", t)

    return int(c_p[0]), int(c_p[1])


def debug(webp_images_dir):
    image_files = sorted(os.listdir(webp_images_dir), key=lambda x: get_chapter_page(x))
    for image_file in image_files:
        print(image_file)
        print(re.findall("\d+", image_file))


def im2pdf(image_file, pdf_file):
    image = Image.open(image_file)
    im = image.convert('RGB')
    im.save(pdf_file)


def webps2pdfs(images, pdfs):
    for image_file in os.listdir(images):
        name = image_file[:-4]
        im2pdf(images + image_file, pdfs + name + "pdf")


def merge_pfds(pdfs):
    merger = PdfMerger()

    for pdf in sorted(os.listdir(pdfs), key=lambda x: get_chapter_page(x)):
        merger.append(pdfs + pdf)

    merger.write("./data/result.pdf")
    merger.close()


def main():
    webp_images = "./data/qm_imgs/"
    qm_pdfs = "./data/qm_pdfs/"
    if not os.path.exists(qm_pdfs):
        os.mkdir(qm_pdfs)
    # debug(webp_images)
    # im2pdf("./data/qm_imgs/量子力学1-1经典力学.webp", "./data/qm_pdfs/量子力学1-1经典力学.pdf")
    webps2pdfs(webp_images, qm_pdfs)
    merge_pfds(qm_pdfs)
    subprocess.run(["rm", "-rf", qm_pdfs])
    x = np.arange(3 * 4 * 5).reshape((3, 4, 5))
    y = x.transpose((2, 0, 1))


if __name__ == '__main__':
    main()
