from PyPDF2 import PdfReader, PdfWriter

# Open the PDF file
input_pdf_path = "pneumatictire_hs-810-561.pdf"
output_pdf_path = "Chapter_1_An_Overview_of_Tire_Technology.pdf"

reader = PdfReader(input_pdf_path)
writer = PdfWriter()

# Define the range of pages for Chapter 1 based on the table of contents and actual content
start_page = 7  # Assuming the chapter starts on page 1
end_page = 33   # The end of Chapter 1 appears to be on page 27

# Add the specified pages to the writer object
for page_num in range(start_page, end_page):
    writer.add_page(reader.pages[page_num])

# Save the output PDF
with open(output_pdf_path, "wb") as output_pdf_file:
    writer.write(output_pdf_file)

output_pdf_path
