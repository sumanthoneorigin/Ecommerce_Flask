
import warnings
warnings.filterwarnings("ignore")
import matplotlib
from PyPDF2 import PdfReader 
import pdf2jpg
from pdf2jpg import pdf2jpg
import os
import easyocr
import cv2
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from matplotlib import cm
import numpy as np
import mahotas
from pylab import imshow, show

reader=easyocr.Reader(['en'])
#pdf_path='/home/sumanth/Desktop/PDF_Legal/send3.pdf'#input
#inputpath = rf"{pdf_path}"
#outputpath = r"input_file"
#result1 = pdf2jpg.convert_pdf2jpg(inputpath,outputpath, pages="ALL")
# code to detect type of document

def detect_type(pdf_path):
    pdfreader = PdfReader(pdf_path)
    page = pdfreader.pages[0]
    text = page.extract_text() 
    text=text[110:144]
    text
    if 'ENROLLMENT AGREEMENT' in text:
        return 'type_3'
    elif 'STUDENT RECORDS RELEASE FORM' in text:
        return 'type_2'
    else:
        return 'type_1'

#code to extract selected choices

def selected_option(img):
  # Load the image
  image = cv2.imread(img)

  # Convert the image to grayscale
  gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

  # Apply Gaussian blur to reduce noise
  blurred = cv2.GaussianBlur(gray, (5, 5), 0)

  # Use Canny edge detection to find edges
  edges = cv2.Canny(blurred, 50, 150)

  # Find contours in the edge image
  contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # Initialize a list to store the detected square boxes
  square_boxes = []

  for contour in contours:
    # Approximate the contour to a polygon with less vertices
    epsilon = 0.04 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Check if the polygon has 4 vertices (a square)
    if len(approx) == 4:
      # Calculate the angles between lines connecting consecutive vertices
      angles = []
      for i in range(4):
        p1 = approx[i][0]
        p2 = approx[(i + 1)%4][0]
        p3 = approx[(i + 2)%4][0]
        v1 = p1 - p2
        v2 = p3 - p2
        dot_product = np.dot(v1, v2)
        angle = np.arccos(dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        angles.append(angle)

      # Check if all angles are close to 90 degrees (within a tolerance)
      if all(np.isclose(angles, np.pi / 2, rtol=0.1)):
      # Filter out small contours (you can adjust this threshold as needed)
        if cv2.contourArea(contour) > 101: #101
          square_boxes.append(approx)
  for i, box in enumerate(square_boxes):
    x, y, w, h = cv2.boundingRect(box)
    square= image[y:y+h,x:x+w]
    square_text= image[y:y+(h+40), (x+100):(x+100) + (w+2000)]
    cv2.imwrite(f'/home/sumanth/Desktop/PDF_Legal/checkbox_results/checkbox_squares/square_{i}.png', square)
    cv2.imwrite(f'/home/sumanth/Desktop/PDF_Legal/checkbox_results/checkbox_texts/square_{i}.png', square_text)
    # Draw rectangles around the detected square boxes on the original image
  for box in square_boxes:
    cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
    # Save or display the image with square boxes highlighted
  cv2.imwrite('output_image.png', image)
  #cv2_imshow(image)
  #cv2.waitKey(0)
  #cv2.destroyAllWindows()
  for i in os.listdir('/home/sumanth/Desktop/PDF_Legal/checkbox_results/checkbox_squares')[::-1]:
    if checked_box('/home/sumanth/Desktop/PDF_Legal/checkbox_results/checkbox_squares/'+i)=='checked_box':
      print(' '.join(reader.readtext('/home/sumanth/Desktop/PDF_Legal/checkbox_results/checkbox_texts/'+i,detail=0)))
    else:
      pass
  return

# code to check if the given box is selected or not
def checked_box(image_path):
  img = mahotas.imread(image_path)
  img = img[:, :, 0]
  mean = img.mean()
  #print("Mean Value for 0 channel : " + str(mean))
  if mean>165 and mean<180:
    return "checked_box"
  return "not_checked"

def selected_option_type2(img_path):
  # Read the main image
  img_rgb = cv2.imread(img_path)

  # Convert it to grayscale
  img_gray = cv2.cvtColor(img_rgb,cv2.COLOR_BGR2GRAY)

  # Read the template
  template = cv2.imread('/home/sumanth/Desktop/PDF_Legal/detected_square_13.jpg',0)

  # Store width and height of template in w and h
  w,h = template.shape[::-1]

  # Perform match operations.
  res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)

  # Specify a threshold
  threshold = 0.93

  # Store the coordinates of matched area in a numpy array
  loc = np.where(res >= threshold)

  detected_squares = set()

  # Draw a rectangle around the matched region and store the unique squares
  for pt in zip(*loc[::-1]):
    detected_square = img_rgb[pt[1]:pt[1] + h, pt[0]:pt[0] + w]
    square_text=img_rgb[(pt[1]+10):(pt[1]+10)+(h-20), (pt[0]+80):(pt[0]+80) + (w+30)]
    # Convert the square to a tuple of pixel values to make it hashable
    square_string = detected_square.tostring()
    # Add the square to the set if it's not already present
    detected_squares.add(square_string)
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)

  # Save the unique detected squares as separate images
  for i, square_hash in enumerate(detected_squares):
    square_array = np.frombuffer(square_string, dtype=np.uint8).reshape(h, w, 3)
    cv2.imwrite(f'/home/sumanth/Desktop/PDF_Legal/checkbox_results/checkbox_results_squares/detected_square_{i}.jpg', square_array)
    cv2.imwrite(f'/home/sumanth/Desktop/PDF_Legal/checkbox_results/checkbox_results_text/detected_square_{i}.jpg', square_text)
  print(reader.readtext('/home/sumanth/Desktop/PDF_Legal/checkbox_results/checkbox_results_text/detected_square_0.jpg',detail=0)[0])
  return 


# final code flow

##-- detect type of document type1,type2,type3

def pdf_data_extraction(pdf_path):
    inputpath = rf"{pdf_path}"
    outputpath = r"/home/sumanth/Desktop/PDF_Legal/saved_pdf"
    result = pdf2jpg.convert_pdf2jpg(inputpath,outputpath, pages="ALL")
    output_dir=outputpath+'/'+os.listdir(outputpath)[0]
    if detect_type(pdf_path)=='type_1':
        img_path_1=output_dir+f'/{os.listdir(output_dir)[0]}'
        img_path_2=output_dir+f'/{os.listdir(output_dir)[1]}'
        img=cv2.imread(img_path_2)
        output_image=cv2.rectangle(img, (100, 2200), (100 + 2300, 2200 + 600), (0, 255, 0), 2)
        x,y,w,h=100,2200,2300,600
        cropped_image = img[y:y+h, x:x+w]
        print("extracted text from pdf")
        print('\n')
        res=reader.readtext(cropped_image,detail=0)
        for i in [i for i in res if i[-1]in [':','_'] and i!='RMCAD PRESIDENT:']:
            print({i:res[res.index(i)+1]})
        print('\n')
        print('selected_options from page_1')
        selected_option(img_path_1)
        print('\n')
        print('selected_options from page_2')
        selected_option(img_path_2)
        s=os.listdir('/home/sumanth/Desktop/PDF_Legal/saved_pdf')[0] # delete the folder created
        d=f'/home/sumanth/Desktop/PDF_Legal/saved_pdf/{s}/'
        for i in os.listdir(d):
            os.remove(d+i)
        os.rmdir(d)
        os.rmdir('/home/sumanth/Desktop/PDF_Legal/saved_pdf')
    elif detect_type(pdf_path)=='type_2':
        img_path_1=output_dir+f'/{os.listdir(output_dir)[0]}'
        print("selected_choices")
        selected_option_type2(img_path_1)
        img_1=cv2.imread(img_path_1)
        output_image=cv2.rectangle(img_1, (100, 2830), (100 + 2400, 2830 + 280), (0, 255, 0), 2)
        x,y,w,h=100,2830,2400,280
        cropped_image = img_1[y:y+h, x:x+w]
        print("extracted text from pdf")
        print('\n')
        res=reader.readtext(cropped_image,detail=0)
        keys=[i for i in res if i[-1] in [':','_'] and i!='RMCAD PRESIDENT:']
        values=[i for i in res if i[-1] not in [':','_'] and i!='RMCAD PRESIDENT:']
        for i in list(zip(keys,values)):
           print(i)
        #######
        img_2=cv2.imread(img_path_1)
        output_image=cv2.rectangle(img_2, (140, 1250), (140 + 1140, 1250 + 1400), (0, 255, 0), 2)
        x,y,w,h=140,1250,1140,1400
        cropped_image = img_2[y:y+h, x:x+w]
        print('\n')
        res=reader.readtext(cropped_image,detail=0)
        if res[-1][-1]==':':
          res=res[0:-1]
        a=[i for i in res if i[-1]==':' and i!='RMCAD PRESIDENT:']
        result={i:res[res.index(i)+1] for i in a if i[0:2] not in ['di','Du']}
        for i in list(zip(result.keys(),result.values())):
           print(i)
        #######
        img_3=cv2.imread(img_path_1)
        output_image=cv2.rectangle(img_3, (1280, 1250), (1280 + 1140, 1250 + 1400), (0, 255, 0), 2)
        x,y,w,h=1280,1250,1140,1400
        cropped_image = img_3[y:y+h, x:x+w]
        print('\n')
        res=reader.readtext(cropped_image,detail=0)
        b=[i for i in res if i[-1]==':' and i!='RMCAD PRESIDENT:']
        result={i:res[res.index(i)+1] for i in b if i[0:2] not in ['di','Du']}
        for i in list(zip(result.keys(),result.values())):
           print(i)
        s=os.listdir('/home/sumanth/Desktop/PDF_Legal/saved_pdf')[0] # delete the folder created
        d=f'/home/sumanth/Desktop/PDF_Legal/saved_pdf/{s}/'
        for i in os.listdir(d):
            os.remove(d+i)
        os.rmdir(d)
        os.rmdir('/home/sumanth/Desktop/PDF_Legal/saved_pdf')
    elif detect_type(pdf_path)=='type_3':
        img_path_1=output_dir+f'/{os.listdir(output_dir)[-2]}'
        img_1=cv2.imread(img_path_1)
        output_image=cv2.rectangle(img_1, (100, 1830), (100 + 2300, 1830 + 300), (0, 255, 0), 2)
        x,y,w,h=100,1830,2300,300
        cropped_image = img_1[y:y+h, x:x+w]
        print("extracted text from pdf")
        print('\n')
        res=reader.readtext(cropped_image,detail=0)
        for i in [i for i in res if i[-1]==':' and i!='RMCAD PRESIDENT:']:
            print({i:res[res.index(i)+1]})
        img_2=cv2.imread(img_path_1)
        output_image=cv2.rectangle(img_2, (100, 2420), (100 + 2300, 2420 + 500), (0, 255, 0), 2)
        x,y,w,h=100,2420,2300,500
        cropped_image = img_2[y:y+h, x:x+w]
        print('\n')
        res=reader.readtext(cropped_image,detail=0)
        for i in [i for i in res if i[-1]==':' and i!='RMCAD PRESIDENT:']:
            print({i:res[res.index(i)+1]})
        s=os.listdir('/home/sumanth/Desktop/PDF_Legal/saved_pdf')[0] # delete the folder created
        d=f'/home/sumanth/Desktop/PDF_Legal/saved_pdf/{s}/'
        for i in os.listdir(d):
            os.remove(d+i)
        os.rmdir(d)
        os.rmdir('/home/sumanth/Desktop/PDF_Legal/saved_pdf')
    else:    
        pass
    print('\n')
    return "EXecuted Successfully"
pdf_path='/home/sumanth/Desktop/PDF_Legal/send1.pdf'
print(pdf_data_extraction(pdf_path))

