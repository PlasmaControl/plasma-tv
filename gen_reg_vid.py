import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from scipy.io import readsav
from tqdm import tqdm

tv_image_path = Path('tv_images')
out_path = Path('comparison_vids')
files = sorted(tv_image_path.glob('*.sav'))
file_lengths = [len(readsav(str(file))['emission_structure'][0][3]) for file in files]
tv_dim = readsav(str(files[0]))['emission_structure'][0][7][0].shape
inversion_dim = readsav(str(files[0]))['emission_structure'][0][0][0].shape

# Create the subfolder if it doesn't exist
out_path.mkdir(parents=True, exist_ok=True)

width = 5
w, h = width, tv_dim[0] / tv_dim[1] * width
for file in files:
    
    frames = readsav(file)['emission_structure'][0][3].astype(int)
    tv_image = readsav(file)['emission_structure'][0][7][frames]
    num_frames = len(frames)
    

    # Create the figure and axes
    fig, axes = plt.subplots(frameon=False)

    # Initialize the image
    image1 = axes.imshow(tv_image[0], cmap='plasma')
    axes.axis('off')
    fig.set_size_inches(w,h)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    
    # Define the update function for the animation
    def update(frame):
        image1.set_array(tv_image[frame])
        return image1,

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=tqdm(range(num_frames)), interval=25, blit=True)

    # Save the animation to a file
    output_file = out_path / Path(file.stem).with_suffix('.mp4')
    ani.save(output_file, writer='ffmpeg')

    print(f"Animation saved to: {output_file}")
