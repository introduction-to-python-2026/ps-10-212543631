from PIL import Image
import numpy as np
from scipy.signal import convolve2d

def load_image(image_path):
    """
    הפונקציה מקבלת נתיב לתמונה, טוענת אותה ומחזירה אותה כמערך NumPy.
    """
    # פתיחת קובץ התמונה
    img = Image.open(image_path)
    
    # המרת אובייקט התמונה למערך NumPy
    img_array = np.array(img)
    
    return img_array

# דוגמה לשימוש (בהתאם לנתיב שצירפת):
path = r"G:\My Drive\לימודים\רפואה 2025 - 2032\מבוא לפייתון\תרגיל 10\PXL_20240420_175844419.jpg"
image_data = load_image(path)

print(f"Shape of the image array: {image_data.shape}")

def edge_detection(img_array):
    """
    סעיף 2: זיהוי קצוות בתמונה באמצעות פילטרי Sobel.
    """
    # 1. הפיכת התמונה לאפורה על ידי מיצוע שלושת ערוצי הצבע [cite: 6, 7]
    # axis=2 מתייחס לערוצי ה-RGB
    gray_img = np.mean(img_array, axis=2)
    
    # 2. הגדרת הפילטרים (Kernels) לזיהוי שינויים באנכי ובאופקי [cite: 8, 9, 10]
    kernelY = np.array([[ 1,  2,  1],
                        [ 0,  0,  0],
                        [-1, -2, -1]])
    
    kernelX = np.array([[ 1,  0, -1],
                        [ 2,  0, -2],
                        [ 1,  0, -1]])
    
    # 3. ביצוע קונבולוציה בעזרת scipy.ndimage עם padding=0 [cite: 12, 13]
    # השימוש ב-mode='constant' עם cval=0 מבטיח ריפוד באפסים ושמירה על גודל התמונה המקורית
    edgeY = convolve(gray_img, kernelY, mode='constant', cval=0.0)
    edgeX = convolve(gray_img, kernelX, mode='constant', cval=0.0)
    
    # 4. חישוב עוצמת הקצוות (Magnitude) לפי הנוסחה שניתנה [cite: 18, 19]
    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)
    
    return edgeMAG
