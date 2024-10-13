import customtkinter as ctk
from tkinter import messagebox, filedialog, colorchooser, Toplevel
from PIL import Image, ImageOps, ImageTk, ImageFilter
from tkinter import ttk
import matplotlib.pyplot as plt
from PIL import ExifTags
import cv2
import numpy as np
import threading
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.ndimage import gaussian_filter1d

root = ctk.CTk()
root.geometry("900x650")
root.title("Image Editing")

file_path = ""
canvas_width = 897
canvas_height = 746
original_image = None
edited_image = None
filtered_image = None
scale_factor = 1.0
denoise_timer = None
denoised_image = None
denoised_value = 0

def add_image():
    global file_path, original_image, edited_image, scale_factor
    file_path = filedialog.askopenfilename()
    if file_path:
        image = Image.open(file_path)
        image = correct_orientation(image)
        original_image = image.copy()
        original_image.thumbnail((canvas_width, canvas_height))
        scale_factor = 1.0
        edited_image = None
        filtered_image = None
        apply_adjustments()
        display_image(image)

def display_histogram(image):
    fig, ax = plt.subplots(figsize=(2, 1)) 
    total_pixels = image.shape[0] * image.shape[1]

    if len(image.shape) == 3 and image.shape[2] == 3:
        channels = ('b', 'g', 'r')
        colors = {'b': 'blue', 'g': 'green', 'r': 'red'}
        hist_data = np.zeros((256, 3))
        
        for i, col in enumerate(channels):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            hist_data[:, i] = hist[:, 0]
        
        hist_data = gaussian_filter1d(hist_data, sigma=2, axis=0)
        
        for i, col in enumerate(channels):
            ax.fill_between(range(256), 0, hist_data[:, i], alpha=0.5, color=colors[col], label=f'{col.upper()} channel')
        
        ax.set_xlabel('Pixel value')
        ax.set_yticks([])  
        ax.set_ylim(0, total_pixels /100)  
    else:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
        hist = gaussian_filter1d(hist, sigma=2)  
        ax.plot(hist, color='black')
        ax.set_xlim([0, 256])
        ax.set_ylim(0, total_pixels /100)  

    
    histogram_frame.configure(width=200, height=100)
    
    for widget in histogram_frame.winfo_children():
        widget.destroy() 
    canvas = FigureCanvasTkAgg(fig, master=histogram_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    
    plt.close(fig) 

def correct_orientation(image):
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
        pass
    return image

def display_image(image):
    final_img = np.array(image)
    image = correct_orientation(image)
    global edited_image
    image.thumbnail((canvas_width, canvas_height))
    # Get image to compare (normal size)
    resized_image_normal = image.resize(image.size, Image.LANCZOS)
    edited_image = resized_image_normal

    # Display image (zoomed)
    width, height = image.size
    new_size_zoom = (int(width * scale_factor), int(height * scale_factor))
    resized_image = image.resize(new_size_zoom, Image.LANCZOS)
   
    image_tk = ImageTk.PhotoImage(resized_image)
    canvas.delete("all")

    x = (canvas_width - new_size_zoom[0]) // 2
    y = (canvas_height - new_size_zoom[1]) // 2
    canvas.create_image(x, y, image=image_tk, anchor="nw")
    canvas.image = image_tk
    root.after(0, lambda: display_histogram(final_img))


def clear_canvas():
    canvas.delete("all")
    if canvas.image:
        canvas.create_image(0, 0, image=canvas.image, anchor="nw")

def apply_filter(filter):
    global original_image
    if original_image:
        image = original_image.copy() 
        image = correct_orientation(image)
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
        elif filter == "Gaussian Blur":
            image = apply_gaussian(image)
        elif filter == "Noise":
            image = apply_noise(image)
        global filtered_image, edited_image
        filtered_image = image.copy()
        edited_image = image.copy()
        apply_adjustments()

def apply_sobel(image):
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(image_cv, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(image_cv, cv2.CV_64F, 0, 1, ksize=5)
    sobel = cv2.magnitude(sobelx, sobely)
    sobel = cv2.convertScaleAbs(sobel)
    return Image.fromarray(sobel)

def apply_prewitt(image):
    image_np = np.array(image)
    
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_np
    
    prewittx = cv2.filter2D(gray, cv2.CV_32F, np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]))
    prewitty = cv2.filter2D(gray, cv2.CV_32F, np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]))
    prewittx = prewittx.astype(np.float32)
    prewitty = prewitty.astype(np.float32)
    prewitt = cv2.magnitude(prewittx, prewitty)
    prewitt = cv2.normalize(prewitt, None, 0, 255, cv2.NORM_MINMAX)
    prewitt = prewitt.astype(np.uint8)
    return Image.fromarray(prewitt)

def apply_canny(image):
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    canny = cv2.Canny(image_cv, 100, 200)
    return Image.fromarray(canny)

def apply_gaussian(image):
    image_cv = np.array(image)
    kernel_size = 15
    sigma = 5.0      
    kernel = np.fromfunction(
        lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-(kernel_size-1)/2)**2 + (y-(kernel_size-1)/2)**2) / (2*sigma**2)),
        (kernel_size, kernel_size)
    )
    kernel /= np.sum(kernel)
    
    channels = cv2.split(image_cv)
    blurred_channels = [cv2.filter2D(channel, -1, kernel) for channel in channels]
    blurred_image = cv2.merge(blurred_channels)
    
    return Image.fromarray(blurred_image)

def apply_noise(image):
    image_np = np.array(image)
    row, col, ch = image_np.shape
    mean = 0
    var = 0.01
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image_np + gauss * 255
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)

def apply_adjustments(brightness=None, contrast=None, sharpen=None, denoise=None):
    if brightness is None:
        brightness = brightness_slider.get()
    if contrast is None:
        contrast = contrast_slider.get()
    if sharpen is None:
        sharpen = sharpen_slider.get()
    if denoise is None:
        denoise = denoise_slider.get()
    global filtered_image, original_image
    image_cv = np.array(original_image.copy()) if filtered_image is None else np.array(filtered_image.copy())
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
    final_img = image_cv.copy()

    
    
    if denoise > 0:
        def update_image(denoised_img):
            apply_adjustments_continued(denoised_img, brightness, contrast, sharpen)
        
        threading.Thread(target=denoise_image, args=(final_img, denoise, update_image)).start()
    else:
        apply_adjustments_continued(final_img, brightness, contrast, sharpen)

def denoise_image(final_img, denoise_level, callback):
    global denoised_image, denoised_value
    if denoised_image is None:
        denoised_image = cv2.fastNlMeansDenoisingColored(final_img, None, denoise_level, denoise_level, 7, 21)
        denoised_value = denoise_slider.get()
    elif denoised_value != denoise_slider.get():
        denoised_image = cv2.fastNlMeansDenoisingColored(final_img, None, denoise_level, denoise_level, 7, 21)
        denoised_value = denoise_slider.get()
    callback(denoised_image)

def apply_adjustments_continued(final_img, brightness, contrast, sharpen):
    kernels = [
            np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),  
            np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),  
            np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]),  
            np.array([[-1, -2, -1], [-2, 13, -2], [-1, -2, -1]]), 
            np.array([[-2, -2, -2], [-2, 17, -2], [-2, -2, -2]]),  
            np.array([[-2, -3, -2], [-3, 21, -3], [-2, -3, -2]]), 
            np.array([[-3, -3, -3], [-3, 25, -3], [-3, -3, -3]]),  
            np.array([[-4, -4, -4], [-4, 33, -4], [-4, -4, -4]]), 
            np.array([[-4, -5, -4], [-5, 37, -5], [-4, -5, -4]])  
        ]
    index = int(sharpen)
    final_img = cv2.filter2D(final_img, -1, kernels[index]) 

    brightness = int((brightness - 50) * 2.55) 
    contrast = int((contrast - 50) * 2.55)  
    
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow
        final_img = cv2.addWeighted(final_img, alpha_b, final_img, 0, gamma_b)
    
    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)
        final_img = cv2.addWeighted(final_img, alpha_c, final_img, 0, gamma_c)
    


    adjusted_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
    adjusted_img = Image.fromarray(adjusted_img)
    display_image(adjusted_img)


def compare_images():
    compare_window = Toplevel(root)
    compare_window.title("Compare Images")
    
    original_frame = ctk.CTkFrame(compare_window)
    original_frame.pack(side="left", padx=10, pady=10)
    
    original_image_tk = ImageTk.PhotoImage(original_image)
    original_label = ctk.CTkLabel(original_frame, image=original_image_tk)
    original_label.image = original_image_tk
    original_label.pack(pady=10)
    
    original_text_label = ctk.CTkLabel(original_frame, text="Original Image")
    original_text_label.pack()
    
    edited_frame = ctk.CTkFrame(compare_window)
    edited_frame.pack(side="right", padx=10, pady=10)
    
    width, height = edited_image.size
    resized_edited_image = edited_image.resize((width, height), Image.LANCZOS)
    edited_image_tk = ImageTk.PhotoImage(resized_edited_image)
    edited_label = ctk.CTkLabel(edited_frame, image=edited_image_tk)
    edited_label.image = edited_image_tk
    edited_label.pack(pady=10)
    
    edited_text_label = ctk.CTkLabel(edited_frame, text="Edited Image")
    edited_text_label.pack()
    
def save_image():
    global edited_image, file_path
    if edited_image is not None:
        file_path_new = filedialog.asksaveasfilename(initialfile=file_path.split("/")[-1].split(".")[0] + "_edited.jpg", defaultextension=".jpg", filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
        if file_path_new:
            edited_image.save(file_path_new)
            messagebox.showinfo("Image Saved", "The image has been saved successfully.")
    else:
        messagebox.showerror("Error", "No image to save.")

def zoom(event):
    global scale_factor
    if event.delta > 0:
        scale_factor *= 1.1
    else:
        scale_factor /= 1.1
    apply_adjustments()

def pan_start(event):
    canvas.scan_mark(event.x, event.y)

def pan_move(event):
    canvas.scan_dragto(event.x, event.y, gain=1)

root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=3)

left_frame = ctk.CTkFrame(root, width=500, height=600)
left_frame.grid(row=0, column=0, sticky="nswe")

histogram_frame = ctk.CTkFrame(left_frame)
histogram_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
histogram_frame.configure(width=200, height=80)

canvas = ctk.CTkCanvas(root, width=750, height=600, bg="#24292A")
canvas.grid(row=0, column=1, sticky="nswe")

image_button = ctk.CTkButton(left_frame, text="Open Image", command=add_image)
image_button.pack(pady=15)

brightness_label = ctk.CTkLabel(left_frame, text="Brightness")
brightness_label.pack()
brightness_value_label = ctk.CTkLabel(left_frame, text="50")
brightness_value_label.pack()
def update_brightness_value(value):
    brightness_value_label.configure(text=f"{int(float(value))}")
    apply_adjustments()


brightness_slider = ctk.CTkSlider(left_frame, from_=0, to=100, number_of_steps=100, command=update_brightness_value)
brightness_slider.set(50)
brightness_slider.pack(pady=15)


contrast_label = ctk.CTkLabel(left_frame, text="Contrast")
contrast_label.pack()
contrast_value_label = ctk.CTkLabel(left_frame, text="50")
contrast_value_label.pack()
def update_contrast_value(value):
    contrast_value_label.configure(text=f"{int(float(value))}")
    apply_adjustments()

contrast_slider = ctk.CTkSlider(left_frame, from_=0, to=100, number_of_steps=100, command=update_contrast_value)
contrast_slider.set(50)
contrast_slider.pack(pady=15)

sharpen_label = ctk.CTkLabel(left_frame, text="Sharpen")
sharpen_label.pack()
sharpen_value_label = ctk.CTkLabel(left_frame, text="0")
sharpen_value_label.pack()
def update_sharpen_value(value):
    sharpen_value_label.configure(text=f"{int(float(value))}")
    apply_adjustments()

sharpen_slider = ctk.CTkSlider(left_frame, from_=0, to=8, number_of_steps=9, command=update_sharpen_value)
sharpen_slider.set(0) 
sharpen_slider.pack(pady=15)

denoise_label = ctk.CTkLabel(left_frame, text="Denoising/Smoothing")
denoise_label.pack()
denoise_value_label = ctk.CTkLabel(left_frame, text="0")
denoise_value_label.pack()
def update_denoise_value(value):
    global denoise_timer
    denoise_value_label.configure(text=f"{int(float(value))}")

    if denoise_timer is not None:
        denoise_timer.cancel()

    denoise_timer = threading.Timer(0.1, lambda: apply_adjustments())
    denoise_timer.start()

denoise_slider = ctk.CTkSlider(left_frame, from_=0, to=20, number_of_steps=18, command=update_denoise_value)
denoise_slider.set(0) 
denoise_slider.pack(pady=15)




filter_label = ctk.CTkLabel(left_frame, text="Select Filter")
filter_label.pack()
filter_combobox = ttk.Combobox(left_frame, values=["None", "Black and White", "Sharpen", "Smooth","Blur", "Gaussian Blur", "Emboss", "Sobel", "Prewitt", "Canny Edge", "Noise"])
filter_combobox.current(0)
filter_combobox.pack()

filter_combobox.bind("<<ComboboxSelected>>", lambda event: apply_filter(filter_combobox.get()))

button_frame = ctk.CTkFrame(left_frame)
button_frame.pack(pady=15)

compare_button = ctk.CTkButton(button_frame, text="Compare", command=compare_images)
compare_button.pack(side=ctk.LEFT, padx=5)

save_button = ctk.CTkButton(button_frame, text="Save", command=save_image)
save_button.pack(side=ctk.LEFT, padx=5)

canvas.bind("<MouseWheel>", zoom)
canvas.bind("<Button-1>", pan_start)
canvas.bind("<B1-Motion>", pan_move)

root.mainloop()