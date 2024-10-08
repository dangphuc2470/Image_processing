import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

def find_similar_images(test_image_data, dataset, num_results=10):
    # Tính histogram của ảnh test
    test_image = test_image_data.get_image()
    test_hist = cv2.calcHist([test_image], [0], None, [256], [0, 256])
    
    # Tính histogram và so sánh với các ảnh trong dataset
    distances = []
    for data in dataset:
        img = data.get_image()
        img_hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        distance = compare_histograms(test_hist, img_hist)
        distances.append((distance, data))
    
    # Sắp xếp và chọn ảnh có khoảng cách nhỏ nhất
    distances.sort(key=lambda x: x[0])
    return [data for _, data in distances[:num_results]]

def compare_histograms(hist1, hist2):
    # return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return np.linalg.norm(hist1 - hist2)

class ImageData:
    def __init__(self, path):
        self.path = path
        self.image = cv2.imread(path)

    def get_path(self):
        return self.path

    def get_image(self):
        width, height = self.image.shape[1], self.image.shape[0]
        scale = width/height
        resized_image = cv2.resize(self.image, (int(100*scale), 100), interpolation=cv2.INTER_AREA)
        # return resized_image
        return self.image

class ImageHistogramNavigator:
    def __init__(self, default_image_data, image_data_list):
        self.default_image_data = default_image_data
        self.image_data_list = image_data_list
        self.index = 0
        self.fig = plt.figure(figsize=(5, 5))
        self.gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[3, 1])
        self.ax1 = plt.subplot(self.gs[0, 0])
        self.ax2 = plt.subplot(self.gs[0, 1])
        self.ax3 = plt.subplot(self.gs[1, 0])
        self.ax4 = plt.subplot(self.gs[1, 1])
        self.text_obj = None
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.update_plot()

    def update_plot(self):
        if not self.image_data_list:
            print("No images in the list.")
            return

        self.index = self.index % len(self.image_data_list)  # Ensure index is within bounds

        default_image = self.default_image_data.get_image()
        default_image_filename = os.path.basename(self.default_image_data.get_path())
        current_image_data = self.image_data_list[self.index]
        current_image = current_image_data.get_image()
        current_image_path = current_image_data.get_path()
        current_image_filename = os.path.basename(current_image_path)

        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        
        # Plot the default image on the top-left subplot
        self.ax1.imshow(cv2.cvtColor(default_image, cv2.COLOR_BGR2RGB))
        self.ax1.set_title(default_image_filename)
        self.ax1.axis('off')  # Hide the axis

        # Plot the current image on the top-right subplot
        self.ax2.imshow(cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB))
        self.ax2.set_title(current_image_filename)
        self.ax2.axis('off')  # Hide the axis

        # Plot the histogram for the default image on the bottom-left subplot
        self.plot_histogram(default_image, self.ax3, 'Histogram')

        # Plot the histogram for the current image on the bottom-right subplot
        self.plot_histogram(current_image, self.ax4, 'Histogram')
        
        # Update the count at the center top of the plot
        if self.text_obj:
            self.text_obj.remove()
        self.text_obj = self.fig.text(0.5, 0.95, f'{self.index + 1} / {len(self.image_data_list)}', ha='center', va='center', fontsize=12, color='black')

        plt.tight_layout()
        self.fig.canvas.draw()

    def plot_histogram(self, image, ax, title):
        if len(image.shape) == 3 and image.shape[2] == 3:
            channels = ('b', 'g', 'r')
            colors = {'b': 'blue', 'g': 'green', 'r': 'red'}
            hist_data = np.zeros((256, 3))
            
            for i, col in enumerate(channels):
                hist = cv2.calcHist([image], [i], None, [256], [0, 256])
                hist_data[:, i] = hist[:, 0]
            
            for i, col in enumerate(channels):
                ax.fill_between(range(256), 0, hist_data[:, i], alpha=0.5, color=colors[col], label=f'{col.upper()} channel')
            
            ax.set_title(title)
            ax.set_xlabel('Pixel value')
            ax.set_yticks([])  # Hide the y-axis ticks (frequency numbers)
            # ax.legend()  # Hide the legend if needed
        else:
            print("The image is not in RGB format.")

    def save_combined_image(self, filename):
        current_image = self.image_data_list[self.index].get_image()
        combined_image = self.combine_images(self.default_image_data.get_image(), current_image)
        cv2.imwrite(filename, combined_image)
        print(f"Combined image saved as {filename}")

    def combine_images(self, img1, img2):
        # Ensure both images have the same height
        height = max(img1.shape[0], img2.shape[0])
        width1 = img1.shape[1]
        width2 = img2.shape[1]
        
        combined_image = np.zeros((height, width1 + width2, 3), dtype=np.uint8)
        combined_image[:img1.shape[0], :width1, :] = img1
        combined_image[:img2.shape[0], width1:width1 + width2, :] = img2
        
        return combined_image

    def on_key(self, event):
        if event.key == 'right':
            self.index = (self.index + 1) % len(self.image_data_list)
            self.update_plot()
        elif event.key == 'left':
            self.index = (self.index - 1) % len(self.image_data_list)
            self.update_plot()
        elif event.key == 's':
            self.save_combined_image(f'combined_image_{self.index + 1}.jpg')

def select_default_image():
    global default_image_path
    default_image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*JPEG;*JPG")])
    default_image_label.config(text=default_image_path)

def select_dataset_path():
    global dataset_path
    dataset_path = filedialog.askdirectory()
    dataset_label.config(text=dataset_path)

def start_processing():
    if not default_image_path or not dataset_path:
        messagebox.showerror("Error", "Please select both default image and dataset path.")
        return

    try:
        num_images = int(num_images_entry.get())
    except ValueError:
        messagebox.showerror("Error", "Please enter a valid number of images.")
        return

    default_image_data = ImageData(default_image_path)
    image_data_list = [ImageData(os.path.join(dataset_path, f)) for f in os.listdir(dataset_path) if f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG'))]


    if not image_data_list:
        messagebox.showerror("Error", "No images found in the dataset path.")
        return
    
    compared_image_data = find_similar_images(default_image_data, image_data_list, num_images)


    navigator = ImageHistogramNavigator(default_image_data, compared_image_data)
    plt.show()


def set_dpi_awareness():
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception as e:
        print(f"Failed to set DPI awareness: {e}")

set_dpi_awareness()

root = tk.Tk()
root.title("Image Histogram Navigator")
root.geometry("500x300") 

default_image_path = ""
dataset_path = ""

style = ttk.Style()
style.theme_use('clam')

default_image_button = tk.Button(root, text="Select Default Image", command=select_default_image)
default_image_button.pack(pady=10)

default_image_label = tk.Label(root, text="No file selected")
default_image_label.pack(pady=5)

dataset_button = tk.Button(root, text="Select Dataset Path", command=select_dataset_path)
dataset_button.pack(pady=10)

num_images_label = ttk.Label(root, text="Number of images to show:")
num_images_label.pack(pady=5)
num_images_entry = ttk.Entry(root)

num_images_entry = ttk.Entry(root)
num_images_entry.pack(pady=5)

dataset_label = tk.Label(root, text="No directory selected")
dataset_label.pack(pady=5)

start_button = tk.Button(root, text="Start Processing", command=start_processing)
start_button.pack(pady=20)

root.mainloop()