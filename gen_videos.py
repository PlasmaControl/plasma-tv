import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from scipy.io import readsav
import matplotlib.gridspec as gridspec
from tqdm import tqdm
import h5py
import cv2

def format_axes(fig):
    for i, ax in enumerate(fig.axes):
        ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
        ax.tick_params(labelbottom=False, labelleft=False)

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
        
        tv_image = readsav(file)['emission_structure'][0][7]
        tv_times = readsav(file)['emission_structure'][0][6]
   
        # Create the figure and axes
        fig, axes = plt.subplots(1, 1)

        # Initialize the image
        image1 = axes.imshow(tv_image[0], cmap='plasma', vmin=0, vmax=255)
        axes.axis('off')
        plt.suptitle(f'{file.stem}')
        fig.set_size_inches(w,h * 1.7)
        fig.subplots_adjust(left=0, bottom=0.05, right=1, top=0.95, wspace=None, hspace=None)
        fig.colorbar(image1, ax=axes, orientation='horizontal', pad=0.01, shrink=0.8)
        # Define the update function for the animation
        def update(frame):
            image1.set_array(np.flip(tv_image[frame],1))
            axes.set_title(f'Time: {tv_times[frame]:.2e} ms, Frame: {frame}')
            return image1,

        # Create the animation
        ani = animation.FuncAnimation(fig, update, frames=tqdm(range(len(tv_image))), interval=30, blit=True)

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
        
        _frames = readsav(file)['emission_structure'][0][3].astype(int)
        tv_image = readsav(file)['emission_structure'][0][7][_frames] # get only images corresponding to inverted
        tv_times = readsav(file)['emission_structure'][0][6][_frames]
        inversion_image = readsav(file)['emission_structure'][0][0]
        num_frames = len(_frames)

        # Create the figure and axes
        fig = plt.figure()
        
        gs0 = gridspec.GridSpec(2, 1, figure=fig)

        ax1 = fig.add_subplot(gs0[0])
        ax2 = fig.add_subplot(gs0[1])

        # Initialize the images
        image1 = ax1.imshow(tv_image[0], cmap='plasma')
        image2 = ax2.imshow(inversion_image[0], cmap='viridis', origin='lower', aspect='auto')
        title = ax1.set_title(f'{file.stem}')
        plt.suptitle(f'{file.stem}')
        # Define the update function for the animation
        def update(frame):
            image1.set_array(np.flip(tv_image[frame],1))
            image2.set_array(inversion_image[frame])
            title.set_text(f'Time: {tv_times[frame]:.2e} ms, Frame: {_frames[frame]}')
            return image1, image2, title

        # Create the animation
        ani = animation.FuncAnimation(fig, update, frames=tqdm(range(num_frames)), interval=25, blit=True)

        # Save the animation to a file
        output_file = out_path / Path(file.stem).with_suffix('.mp4')
        ani.save(output_file, writer='ffmpeg')

        print(f"Animation saved to: {output_file}")
        
def gen_synthetic_comparison():
   
    file_name = 'outputs/hdf5/s_outs_v3_limited.h5'
    with h5py.File(file_name, 'r') as f:
        synthetic_images = f['image'][:]
        
    file_name_2 = 'outputs/hdf5/x_outer_radiation.hdf5'
    with h5py.File(file_name_2, 'r') as f:
        tv_images = f['tv_images'][:]
        
    num_frames = synthetic_images.shape[0]

    # Create the figure and axes
    fig, axes = plt.subplots(2, 1)
    fig.tight_layout()

    # Initialize the images
    image1 = axes[0].imshow(cv2.flip(tv_images[0],0), cmap='plasma', origin='lower')
    image2 = axes[1].imshow(synthetic_images[0], cmap='viridis', origin='lower')

    # Define the update function for the animation
    def update(frame):
        image1.set_array(cv2.flip(tv_images[frame],0))
        image2.set_array(synthetic_images[frame])
        return image1, image2

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=tqdm(range(num_frames)), interval=25, blit=True)

    # Save the animation to a file
    output_file = 'synthetic_comparison.mp4'
    ani.save(output_file, writer='ffmpeg')

    print(f"Animation saved to: {output_file}")

if __name__ == '__main__':
    
    prompt = input('1: Regular, 2: Inversion Comparison, 3: Synthetic Comparison -- ')
    
    match prompt:
        case '1':
            gen_reg_vids()
        case '2':
            gen_comparison_vids()
        case '3':
            gen_synthetic_comparison()
        case _:
            print('Invalid input')