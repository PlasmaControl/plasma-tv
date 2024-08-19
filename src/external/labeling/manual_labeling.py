import json
import numpy as np
from PIL import Image, ImageTk
from pathlib import Path
from scipy.io import readsav
import csv
from src.data.file_utils import GetTV
from tkinter import Tk, Label

def next_photo(event):
    """
    Advance to the next photo in the list, wrap around to the first photo if at the end.
    Zooms in on the image for display.

    Parameters:
    event : Event
        The event object from tkinter representing user actions (e.g., key press).
    """
    global current_photo_index, w, photos
    current_photo_index = (current_photo_index + 1) % len(photos)
    if current_photo_index == len(photos) - 1:
        print("Done!")
    photo = ImageTk.PhotoImage(image=Image.fromarray(photos[current_photo_index]))
    photo = photo._PhotoImage__photo.zoom(4)
    w.configure(image=photo)
    w.image = photo

def previous_photo(event):
    """
    Move to the previous photo in the list, wrap around to the last photo if at the beginning.
    Zooms in on the image for display.

    Parameters:
    event : Event
        The event object from tkinter representing user actions (e.g., key press).
    """
    global current_photo_index, w, photos
    current_photo_index = (current_photo_index - 1) % len(photos)
    photo = ImageTk.PhotoImage(image=Image.fromarray(photos[current_photo_index]))
    photo = photo._PhotoImage__photo.zoom(4)
    w.configure(image=photo)
    w.image = photo

def callback(event):
    """
    Handles mouse click events, saves the clicked coordinates into a CSV file along with the current photo index.

    Parameters:
    event : Event
        The event object from tkinter representing mouse actions.
    """
    global current_photo_index, coordinates, out_file
    print("clicked at", event.x, event.y)
    coordinates.append((event.x, event.y))

    with open(out_file, "a", newline='') as file:
        writer = csv.writer(file)
        writer.writerow([current_photo_index, event.x, event.y])

if __name__ == "__main__":
    """
    Main execution block initializing the GUI, loading images, and binding event handlers.
    """
    file_idx = 0  # Example index of the file to load
    root = Tk()
    tv_image_path = Path("data/raw/tv_images/campaign_24")
    output_path = Path("data/labels/xr_points")

    tv = GetTV(tv_image_path)
    files = tv.list_files()
    file = files[file_idx]
    inverted = tv.load(file, "inverted")
    out_file = output_path / Path(file.stem).with_suffix(".csv")
    photos = (255 * (inverted - np.min(inverted)) / (np.max(inverted) - np.min(inverted))).astype("uint8")

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