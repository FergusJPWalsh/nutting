# OCR Latin text in images
# Adapted from https://github.com/wjbmattingly/ocr_python_textbook

import pytesseract
import cv2
import numpy as np
import re

def normalise_latin(string):
    """Normalise Latin language character set."""
    norm_string = re.sub("ã|â|à|á", "ā", string)
    norm_string = re.sub("ẽ|ê|è|é", "ē", norm_string)
    norm_string = re.sub("ĩ|î|ì|í", "ī", norm_string)
    norm_string = re.sub("õ|ô|ò|ó", "ō", norm_string)
    norm_string = re.sub("ũ|û|ù|ú|ü", "ū", norm_string)
    return norm_string

def transcribe(page_number):
    img = cv2.imread(f"afirstlatinread01nuttgoog_tif/afirstlatinread01nuttgoog_{page_number}.tif")
    img_copy = img.copy()
    greyscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(greyscale, (15, 15), -1)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernal_e = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
    kernal_d = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
    dilate = cv2.dilate(thresh, kernal_d, iterations=8)
    # cv2.imwrite("dilate_test.png", dilate)
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    cnts = sorted(cnts, key=lambda x: cv2.boundingRect(x)[0] + cv2.boundingRect(x)[1] * img.shape[1])
    ocr_outputs = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if h > 300 and w > 1800:
            roi = img_copy[y:y+h, x:x+w]
            # cv2.rectangle(img, (x,y), (x+w, y+h), (36,255,12), 12)
            # cv2.imwrite(f"roi_test_{page_number}.png", img)
            ocr_result = pytesseract.image_to_string(roi, lang="lat")
            ocr_result = normalise_latin(ocr_result)
            ocr_outputs.append(ocr_result)
    return ocr_outputs

ocr_outputs = []


with open("nutting_ocr_transcription.txt", "w", encoding="utf-8") as text_file:
    for i in range(16,171):
        page_number = f"{i:04d}"
        ocr_outputs = transcribe(page_number)
        for output in ocr_outputs:
            text_file.write(f"{output}\n")
            print(output)
            print(f"Page number {page_number} transcribed")
print("Finished!")