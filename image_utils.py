from PIL import Image
import numpy as np
from scipy.signal import convolve2d [cite: 15]

def load_image(image_path):
    """
    1. פונקציה המקבלת מיקום תמונה והופכת אותה למערך NumPy[cite: 2].
    """
    img = Image.open(image_path)
    img_array = np.array(img)
    return img_array

def edge_detection(img_array):
    """
    2. פונקציה המקבלת מערך תמונה ומחזירה מערך המדגיש את הקצוות[cite: 3].
    """
    # הפיכת המערך לתמונה אפורה על-ידי מיצוע שלושת ערוצי הצבע [cite: 6, 7]
    gray_img = np.mean(img_array, axis=2)
    
    # בניית פילטר לזיהוי שינויים בכיוון האנכי [cite: 8, 10]
    kernelY = np.array([[ 1,  2,  1],
                        [ 0,  0,  0],
                        [-1, -2, -1]])
    
    # בניית פילטר לזיהוי שינויים בכיוון האופקי [cite: 9, 10]
    kernelX = np.array([[ 1,  0, -1],
                        [ 2,  0, -2],
                        [ 1,  0, -1]])
    
    # ביצוע קונבולוציה בעזרת convolve2d מהספרייה scipy.signal 
    # mode='same' שומר על אורך ורוחב מקוריים [cite: 13]
    # boundary='fill' עם fillvalue=0 מבצע padding=0 [cite: 13]
    edgeY = convolve2d(gray_img, kernelY, mode='same', boundary='fill', fillvalue=0) [cite: 12, 13, 16]
    edgeX = convolve2d(gray_img, kernelX, mode='same', boundary='fill', fillvalue=0) [cite: 12, 13, 17]
    
    # חישוב עוצמת הקצוות לפי הנוסחה [cite: 18, 19]
    edgeMAG = np.sqrt(edgeX**2 + edgeY**2) [cite: 19]
    
    return edgeMAG
