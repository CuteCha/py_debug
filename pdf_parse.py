import fitz

pdf_path = "/Users/cxq/Downloads/LLM-TAP.pdf"
pdf_doc = fitz.open(pdf_path)
page_count = pdf_doc.page_count
print(f"页面总数：{page_count}")

# 读取第一页
page = pdf_doc[0]
print(page)
text = page.get_text()
print(text)

from PyPDF2 import PdfFileReader

pdf_document = "/Users/cxq/Downloads/LLM-TAP.pdf"
with open(pdf_document, "rb") as filehandle:
    pdf = PdfFileReader(filehandle)
    info = pdf.getDocumentInfo()
    pages = pdf.getNumPages()

    print(info)
    print("number of pages: %i" % pages)

    page1 = pdf.getPage(0)
    print(page1)
    print(page1.extractText())
