from PIL import Image
import numpy as np
from scipy import ndimage

def load_image(image_path):
    """
    1. טעינת תמונת צבע והפיכתה למערך NumPy[cite: 2].
    """
    img = Image.open(image_path)
    return np.array(img)

def edge_detection(img_array):
    """
    2. זיהוי קצוות בתמונה לפי שלבי התרגיל[cite: 3].
    """
    # א. הפיכה לתמונה אפורה על-ידי מיצוע ערוצי הצבע [cite: 6, 7]
    gray_img = np.mean(img_array, axis=2)
    
    # ב. בניית הפילטרים האנכי והאופקי [cite: 8, 9, 10]
    kernelY = np.array([[ 1,  2,  1],
                        [ 0,  0,  0],
                        [-1, -2, -1]])
    
    kernelX = np.array([[ 1,  0, -1],
                        [ 2,  0, -2],
                        [ 1,  0, -1]])
    
    # ג. ביצוע קונבולוציה בעזרת scipy.ndimage.convolve 
    # הגדרת mode='constant' ו-cval=0 מבצעת padding=0 
    # הפעולה שומרת על אורך ורוחב מקוריים 
    edgeY = ndimage.convolve(gray_img, kernelY, mode='constant', cval=0.0)
    edgeX = ndimage.convolve(gray_img, kernelX, mode='same', boundary='fill', fillvalue=0) # השורה הזו שונתה למטה לתיקון
    
    # תיקון: שתי הפעולות חייבות להשתמש באותו סוג של קונבולוציה
    edgeY = ndimage.convolve(gray_img, kernelY, mode='constant', cval=0.0)
    edgeX = ndimage.convolve(gray_img, kernelX, mode='constant', cval=0.0)
    
    # ד. חישוב עוצמת הקצוות (Magnitude) [cite: 18, 19]
    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)
    
    return edgeMAG
