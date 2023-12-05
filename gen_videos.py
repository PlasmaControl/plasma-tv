import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from scipy.io import readsav
from tqdm import tqdm

tv_image_path = Path('tv_images/issues')
out_path = Path('comparison_vids')
files = sorted(tv_image_path.glob('*.sav'))
file_lengths = [len(readsav(str(file))['emission_structure'][0][3]) for file in files]
tv_dim = readsav(str(files[0]))['emission_structure'][0][7][0].shape
inversion_dim = readsav(str(files[0]))['emission_structure'][0][0][0].shape

# Create the subfolder if it doesn't exist
out_path.mkdir(parents=True, exist_ok=True)

for file in tqdm(files):
    
    frames = readsav(file)['emission_structure'][0][3].astype(int)
    tv_image = readsav(file)['emission_structure'][0][7][frames]
    inversion_image = readsav(file)['emission_structure'][0][0]
    num_frames = len(frames)

    # Create the figure and axes
    fig, axes = plt.subplots(2, 1, figsize=(5, 10), gridspec_kw={'width_ratios': [1]})
    fig.tight_layout()

    # Initialize the images
    image1 = axes[0].imshow(tv_image[0], cmap='plasma')
    image2 = axes[1].imshow(inversion_image[0], cmap='viridis', origin='lower')

    # Define the update function for the animation
    def update(frame):
        image1.set_array(tv_image[frame])
        image2.set_array(inversion_image[frame])
        return image1, image2

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=tqdm(range(num_frames)), interval=25, blit=True)

    # Save the animation to a file
    output_file = out_path / Path(file.stem).with_suffix('.mp4')
    ani.save(output_file, writer='ffmpeg')

    print(f"Animation saved to: {output_file}")
