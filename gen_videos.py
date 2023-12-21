import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from scipy.io import readsav
from tqdm import tqdm

def gen_reg_vids():
    tv_image_path = Path('tv_images')
    out_path = Path('outputs/regular_vids')
    files = sorted(tv_image_path.glob('*.sav'))
    tv_dim = readsav(str(files[0]))['emission_structure'][0][7][0].shape

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
        
def gen_comparison_vids():
    tv_image_path = Path('tv_images')
    out_path = Path('outputs/comparison_vids')
    files = sorted(tv_image_path.glob('*.sav'))

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

if __name__ == '__main__':
    
    prompt = input('Generate comparison videos? (y/n):')
    if prompt == 'y':
        gen_comparison_vids()
    else:
        gen_reg_vids()