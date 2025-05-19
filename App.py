import cv2
import pandas as pd

# Load color data
colors = pd.read_csv("colors.csv")

def get_color_name(R, G, B):
    min_dist = float("inf")
    cname = ""
    for i in range(len(colors)):
        d = abs(R - int(colors.loc[i, "R"])) + abs(G - int(colors.loc[i, "G"])) + abs(B - int(colors.loc[i, "B"]))
        if d < min_dist:
            min_dist = d
            cname = colors.loc[i, "color_name"]
    return cname

# Global variables for callback
scale_x = 1
scale_y = 1
display_img = None
orig_img = None

def draw_function(event, x, y, flags, param):
    global display_img, orig_img, scale_x, scale_y
    if event == cv2.EVENT_LBUTTONDOWN:
        # Map (x, y) on resized image back to original image coordinates
        orig_x = int(x * scale_x)
        orig_y = int(y * scale_y)

        # Get pixel color from original image
        b, g, r = orig_img[orig_y, orig_x]
        color_name = get_color_name(r, g, b)
        print(f"Clicked at resized ({x},{y}) -> original ({orig_x},{orig_y}) - Color: {color_name}")

        # Draw rectangle on displayed (resized) image
        cv2.rectangle(display_img, (20, 20), (400, 60), (int(b), int(g), int(r)), -1)
        cv2.putText(display_img, color_name, (30, 50), 2, 0.8, (255, 255, 255), 2)
        cv2.imshow('Image', display_img)

# Load image
img_path = "test_images/manu.jpg"  # Change if needed
orig_img = cv2.imread(img_path)
if orig_img is None:
    print("Error: Image not found. Check the path.")
    exit()

# Resize image to max width or height for display
max_width = 800
max_height = 600
h, w = orig_img.shape[:2]

scale_w = w / max_width if w > max_width else 1
scale_h = h / max_height if h > max_height else 1

scale = max(scale_w, scale_h)

new_w = int(w / scale)
new_h = int(h / scale)

display_img = cv2.resize(orig_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

# Save scaling factors for mapping clicks
scale_x = w / new_w
scale_y = h / new_h

cv2.namedWindow('Image')
cv2.setMouseCallback('Image', draw_function)

print("Click on the image to detect colors. Press ESC to exit.")
while True:
    cv2.imshow('Image', display_img)
    if cv2.waitKey(20) & 0xFF == 27:  # ESC key to exit
        break

cv2.destroyAllWindows()



