# Manual labeling of points on TV images and saves to CSV File

from tkinter import *
from PIL import Image, ImageTk
import json
import numpy as np
import numpy as np
from PIL import Image
from pathlib import Path
from scipy.io import readsav
import csv
from utils.get_file import GetTV
#test

def next_photo(event):
    global current_photo_index
    current_photo_index = (current_photo_index + 1) % len(photos)
    if current_photo_index == len(photos) - 1:
        print("Done!")
    photo = ImageTk.PhotoImage(image=Image.fromarray(photos[current_photo_index]))
    photo = photo._PhotoImage__photo.zoom(4)
    w.configure(image=photo)
    w.image = photo


def previous_photo(event):
    global current_photo_index
    current_photo_index = (current_photo_index - 1) % len(photos)
    photo = ImageTk.PhotoImage(image=Image.fromarray(photos[current_photo_index]))
    photo = photo._PhotoImage__photo.zoom(4)
    w.configure(image=photo)
    w.image = photo


def callback(event):
    print("clicked at", event.x, event.y)
    coordinates.append((event.x, event.y))

    # Dump coordinates and photo index to file
    with open(out_file, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([current_photo_index, event.x, event.y])


if __name__ == "__main__":

    file_idx = 2

    root = Tk()
    tv_image_path = Path("tv_images/l-mode")
    output_path = Path("../outputs/manual_labeled_points")

    tv = GetTV(tv_image_path)
    files = tv.list_files()
    file = files[file_idx]
    inverted = tv.load(file, "inverted")
    out_file = output_path / Path(file.stem).with_suffix(".csv")
    photos = (
        255 * (inverted - np.min(inverted)) / (np.max(inverted) - np.min(inverted))
    ).astype("uint8")

    current_photo_index = 0
    photo = ImageTk.PhotoImage(image=Image.fromarray(photos[current_photo_index]))
    photo = photo._PhotoImage__photo.zoom(4)
    w = Label(root, image=photo)
    w.pack()

    coordinates = []

    root.bind("<Right>", next_photo)
    root.bind("<Left>", previous_photo)
    w.bind("<Button-1>", callback)

    root.mainloop()
