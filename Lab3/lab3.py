import customtkinter as ctk
from tkinter import messagebox, filedialog, colorchooser
from PIL import Image, ImageOps, ImageTk, ImageFilter
from tkinter import ttk
from PIL import ExifTags

root = ctk.CTk()
root.geometry("900x460")
root.title("Image Drawing Tool")

pen_color = "black"
pen_size = 5
file_path = ""
required_height = 600



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

    scale = image.width / image.height
    width = image.width if image.height < required_height else int(required_height * scale)
    height = required_height
    image = image.resize((width, height), Image.LANCZOS)
    canvas.config(width=image.width, height=image.height)
    image = ImageTk.PhotoImage(image)
    canvas.image = image
    canvas.create_image(0, 0, image=image, anchor="nw")

def clear_canvas():
    canvas.delete("all")
    canvas.create_image(0, 0, image=canvas.image, anchor="nw")

def apply_filter(filter):
    image = Image.open(file_path)
    width, height = int(image.width / 2), int(image.height / 2)
    image = image.resize((width, height), Image.LANCZOS)
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
    image = ImageTk.PhotoImage(image)
    canvas.image = image
    canvas.create_image(0, 0, image=image, anchor="nw")

root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=3)

left_frame = ctk.CTkFrame(root, width=500, height=600)
left_frame.grid(row=0, column=0, sticky="nswe")

canvas = ctk.CTkCanvas(root, width=750, height=600)
canvas.grid(row=0, column=1, sticky="nswe")

image_button = ctk.CTkButton(left_frame, text="Add Image", command=add_image)
image_button.pack(pady=15)


filter_label = ctk.CTkLabel(left_frame, text="Select Filter")
filter_label.pack()
filter_combobox = ttk.Combobox(left_frame, values=["Black and White", "Blur", "Emboss", "Sharpen", "Smooth"])
filter_combobox.pack()

filter_combobox.bind("<<ComboboxSelected>>", lambda event: apply_filter(filter_combobox.get()))


root.mainloop()