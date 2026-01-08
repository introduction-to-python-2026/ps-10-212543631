from PIL import Image
import numpy as np
from scipy import ndimage # שימוש בספרייה המדויקת לפי ההנחיות 

def load_image(image_path):
    """
    1. כתיבת פונקציה המקבלת מיקום תמונה ומחזירה מערך NumPy[cite: 2].
    """
    img = Image.open(image_path)
    return np.array(img)

def edge_detection(img_array):
    """
    2. פונקציה המדגישה את הקצוות בתמונה לפי שלבי התרגיל[cite: 3].
    """
    # א. הפיכת המערך לתמונה אפורה על-ידי מיצוע ערכי הצבע [cite: 6, 7]
    gray_img = np.mean(img_array, axis=2)
    
    # ב. בניית פילטרים לזיהוי שינויים באנכי ובאופקי [cite: 8, 9, 10]
    kernelY = np.array([[ 1,  2,  1],
                        [ 0,  0,  0],
                        [-1, -2, -1]])
    
    kernelX = np.array([[ 1,  0, -1],
                        [ 2,  0, -2],
                        [ 1,  0, -1]])
    
    # ג. הפעלת הקונבולוציה בעזרת scipy.ndimage 
    # הגדרת mode='constant' ו-cval=0 מבצעת בדיוק padding=0 
    # פעולה זו שומרת על אורך ורוחב מקוריים 
    edgeY = ndimage.convolve(gray_img, kernelY, mode='constant', cval=0.0)
    edgeX = ndimage.convolve(gray_img, kernelX, mode='constant', cval=0.0)
    
    # ד. חישוב עוצמת הקצוות לפי הנוסחה המשולבת 
    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)
    
    return edgeMAG
