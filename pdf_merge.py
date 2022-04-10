# -*- coding: utf-8 -*-

import os
import PyPDF2


def merge_pdfs0(target_dir, name):
    pdf_lst = sorted([f for f in os.listdir(target_dir) if f.endswith('.pdf')])
    print(pdf_lst)
    pdf_lst = [os.path.join(target_dir, filename) for filename in pdf_lst]

    print("starting ......")
    pdf_writer = PyPDF2.PdfFileWriter()
    for file_name in pdf_lst:
        print(file_name)
        pdf_reader = PyPDF2.PdfFileReader(open(file_name, 'r', encoding="utf-8"), strict=False)
        for page_num in range(1, pdf_reader.numPages):
            page_obj = pdf_reader.getPage(page_num)
            pdf_writer.addPage(page_obj)

    pdf_writer.write(open(name, 'w'))
    print("done ......")


def merge_pdfs(target_dir, name):
    pdf_lst = sorted([f for f in os.listdir(target_dir) if f.endswith('.pdf')])
    print(pdf_lst)
    pdf_lst = [os.path.join(target_dir, filename) for filename in pdf_lst]

    print("starting ......")
    file_merger = PyPDF2.PdfFileMerger()
    for pdf in pdf_lst:
        print(pdf)
        file_merger.append(pdf)

    file_merger.write(name)
    print("done ......")


def main():
    merge_pdfs("./xxx", "ryy.pdf")


if __name__ == '__main__':
    main()
