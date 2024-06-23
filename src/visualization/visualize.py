import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import imageio

def plot_single(data, with_axis=True, labels=None, filename=None):
    fig, ax = plt.subplots()
    if not with_axis:
        ax.axis('off')
    im = ax.imshow(data, cmap='plasma')
    if labels:
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
    if filename:
        plt.savefig(filename)
    plt.show()

def plot_stacked(data1, data2, with_axis=True, labels=None, filename=None):
    fig, axs = plt.subplots(2, 1)
    for idx, (ax, data) in enumerate(zip(axs, [data1, data2])):
        if not with_axis:
            ax.axis('off')
        im = ax.imshow(data, cmap='plasma')
        if labels:
            ax.set_xlabel(labels[idx][0])
            ax.set_ylabel(labels[idx][1])
    if filename:
        plt.savefig(filename)
    plt.show()

def animate_images(data_sequence, mode='gif', filename='animation.gif'):
    fig, ax = plt.subplots()
    ax.axis('off')
    frames = []  # List to hold the generated images

    for data in data_sequence:
        frame = ax.imshow(data, cmap='plasma', animated=True)
        if mode == 'gif':
            ax.figure.canvas.draw()
            image = np.frombuffer(ax.figure.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(ax.figure.canvas.get_width_height()[::-1] + (3,))
            frames.append(image)
        elif mode == 'video':
            frames.append([frame])

    if mode == 'gif':
        imageio.mimsave(filename, frames, fps=2)
    elif mode == 'video':
        ani = FuncAnimation(fig, lambda i: frames[i], frames=len(frames), interval=500, blit=True)
        ani.save(filename, writer='ffmpeg', fps=2)
    plt.close()

# Example usage
if __name__ == "__main__":
    # Generate sample data
    data = np.random.rand(10, 10)
    data_stack = [np.random.rand(10, 10) for _ in range(10)]

    # Static image
    plot_single_image(data, with_axis=True, with_labels=True)

    # Stacked images
    plot_stacked_images(data, data, with_axis=False, with_labels=False)

    # Animate a sequence of images
    animate_images(data_stack, mode='gif', filename='animated.gif')
    animate_images(data_stack, mode='video', filename='animation.mp4')