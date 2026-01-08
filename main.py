import matplotlib.pyplot as plt
from skimage.filters import median
from skimage.morphology import ball
from image_utils import load_image, edge_detection
import numpy as np

# 1. טעינת התמונה המקורית מהנתיב שלך
path = r"G:\My Drive\לימודים\רפואה 2025 - 2032\מבוא לפייתון\תרגיל 10\PXL_20240420_175844419.jpg"
image_data = load_image(path) [cite: 2, 21]

# 2. ניקוי רעשים בעזרת פילטר חציוני (Median Filter)
# השתמשנו ב-ball(3) כפי שהוגדר בהנחיות [cite: 24, 25, 26, 27]
clean_image = median(image_data, ball(3)) [cite: 26]

# 3. הרצת זיהוי הקצוות על התמונה הנקייה
edge_mag = edge_detection(clean_image) [cite: 28]

# 4. הפיכת המערך לבינארי (אפס ואחד) בעזרת ערך סף (Threshold)
# מומלץ לנסות ערכים שונים (למשל 50 או 100) כדי לראות מה עובד הכי טוב לתמונה שלך
threshold_value = 100 
binary_edges = edge_mag > threshold_value [cite: 28]

# 5. הצגת התוצאות והדפסה
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image_data)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Edge Detection (Binary)")
plt.imshow(binary_edges, cmap='gray')
plt.axis('off')

plt.show() [cite: 30]

# 6. שמירת התוצאה כקובץ png
plt.imsave("edge_result.png", binary_edges, cmap='gray') [cite: 30]

print("התהליך הסתיים בהצלחה! התמונה נשמרה כ-edge_result.png")
