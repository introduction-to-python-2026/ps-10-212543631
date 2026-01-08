from PIL import Image
import numpy as np
from scipy.signal import convolve2d

def load_image(image_path):
    """
    1. פונקציה המקבלת מיקום תמונת צבע והופכת את התמונה ל-np.array[cite: 2].
    """
    # פתיחת התמונה
    img = Image.open(image_path)
    # המרה למערך נומפי והחזרתו [cite: 2]
    return np.array(img)

def edge_detection(img_array):
    """
    2. פונקציה המקבלת מערך המייצג תמונה ומחזירה מערך חדש המדגיש את הקצוות[cite: 3].
    """
    # הפיכת המערך לתמונה אפורה על-ידי מיצוע שלושת ערכי הצבע עבור כל פיקסל [cite: 6, 7]
    gray_img = np.mean(img_array, axis=2)
    
    # בניית פילטר לזיהוי שינויים בכיוון האנכי (kernelY) [cite: 8, 10]
    kernelY = np.array([[ 1,  2,  1],
                        [ 0,  0,  0],
                        [-1, -2, -1]])
    
    # בניית פילטר לזיהוי שינויים בכיוון האופקי (kernelX) [cite: 9, 10]
    kernelX = np.array([[ 1,  0, -1],
                        [ 2,  0, -2],
                        [ 1,  0, -1]])
    
    # הפעלת הקונבולוציה עבור כל אחד מהפילטרים תוך שימוש ב-convolve2d [cite: 12]
    # mode='same' מבטיח שהתוצאה תהיה באותו אורך ורוחב כמו התמונה המקורית 
    # boundary='fill' ו-fillvalue=0 מבצעים padding=0 כפי שנדרש 
    edgeY = convolve2d(gray_img, kernelY, mode='same', boundary='fill', fillvalue=0) [cite: 16]
    edgeX = convolve2d(gray_img, kernelX, mode='same', boundary='fill', fillvalue=0) [cite: 17]
    
    # חישוב עוצמת הקצוות (edgeMAG) לפי הנוסחה המשולבת [cite: 18, 19]
    edgeMAG = np.sqrt(edgeX**2 + edgeY**2) [cite: 19]
    
    return edgeMAG
