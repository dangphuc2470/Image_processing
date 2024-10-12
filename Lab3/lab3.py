import customtkinter as ctk
from tkinter import messagebox, filedialog, colorchooser
from PIL import Image, ImageOps, ImageTk, ImageFilter
from tkinter import ttk
from PIL import ExifTags
import cv2
import numpy as np

root = ctk.CTk()
root.geometry("900x460")
root.title("Image Drawing Tool")

pen_color = "black"
pen_size = 5
file_path = ""
canvas_width = 850
canvas_height = 600



def add_image():
    global file_path
    file_path = filedialog.askopenfilename()
    image = Image.open(file_path)
    try:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            exif = image._getexif()
            if exif is not None:
                orientation = exif.get(orientation, 1)
                if orientation == 3:
                    image = image.rotate(180, expand=True)
                elif orientation == 6:
                    image = image.rotate(270, expand=True)
                elif orientation == 8:
                    image = image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
            # Cases: image don't have getexif
            pass
    create_image(image)


def create_image(orgImage):
    orgImage.thumbnail((canvas_width, canvas_height), Image.LANCZOS)
    image = ImageTk.PhotoImage(orgImage)
    canvas.image = image
    x = (canvas_width - image.width()) // 2
    y = (canvas_height - image.height()) // 2
    canvas.create_image(x, y, image=image, anchor="nw")



def clear_canvas():
    canvas.delete("all")
    canvas.create_image(0, 0, image=canvas.image, anchor="nw")

def apply_filter(filter):
    global file_path
    image = Image.open(file_path)
    if filter == "Black and White":
        image = ImageOps.grayscale(image)
    elif filter == "Blur":
        image = image.filter(ImageFilter.BLUR)
    elif filter == "Sharpen":
        image = image.filter(ImageFilter.SHARPEN)
    elif filter == "Smooth":
        image = image.filter(ImageFilter.SMOOTH)
    elif filter == "Emboss":
        image = image.filter(ImageFilter.EMBOSS)
    elif filter == "Sobel":
        image = apply_sobel(image)
    elif filter == "Prewitt":
        image = apply_prewitt(image)
    elif filter == "Canny Edge":
        image = apply_canny(image)
    create_image(image)

def apply_sobel(image):
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(image_cv, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(image_cv, cv2.CV_64F, 0, 1, ksize=5)
    sobel = cv2.magnitude(sobelx, sobely)
    sobel = cv2.convertScaleAbs(sobel)
    return Image.fromarray(sobel)

def apply_prewitt(image):
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    prewittx = cv2.filter2D(image_cv, -1, kernelx)
    prewitty = cv2.filter2D(image_cv, -1, kernely)
    prewitt = cv2.magnitude(prewittx, prewitty)
    prewitt = cv2.convertScaleAbs(prewitt)
    return Image.fromarray(prewitt)

def apply_canny(image):
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    canny = cv2.Canny(image_cv, 100, 200)
    return Image.fromarray(canny)

root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=3)

left_frame = ctk.CTkFrame(root, width=500, height=600)
left_frame.grid(row=0, column=0, sticky="nswe")
canvas = ctk.CTkCanvas(root, width=750, height=600, bg="#24292A")
canvas.grid(row=0, column=1, sticky="nswe")

image_button = ctk.CTkButton(left_frame, text="Add Image", command=add_image)
image_button.pack(pady=15)


filter_label = ctk.CTkLabel(left_frame, text="Select Filter")
filter_label.pack()
filter_combobox = ttk.Combobox(left_frame, values=["Black and White", "Blur", "Emboss", "Sharpen", "Smooth", "Sobel", "Prewitt", "Canny Edge"])
filter_combobox.pack()

filter_combobox.bind("<<ComboboxSelected>>", lambda event: apply_filter(filter_combobox.get()))


root.mainloop()