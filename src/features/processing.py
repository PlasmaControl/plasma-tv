import numpy as np
import cv2
import diplib as dip
import matplotlib.pyplot as plt


class CustomProcessing:
    def __init__(self, kernel_size=5, sigma=1, threshold=0.5, erode_kernel=3):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.threshold = threshold
        self.erode_kernel = erode_kernel

    def norm(self, data):
        mn = data.mean()
        std = data.std()
        return (data - mn) / std

    def rescale(self, data):
        return (data - data.min()) / (data.max() - data.min())

    def quantfilt(self, src, thr=0.9):
        filt = np.quantile(src, thr, axis=0)
        out = np.where(src < filt, 0, src)
        return out

    # gaussian filtering
    def gaussblr(self, src, filt=(31, 3)):
        src = (self.rescale(src) * 255).astype("uint8")
        out = cv2.GaussianBlur(src, filt, 0)
        return self.rescale(out)

    # mean filtering
    def meansub(self, src):
        mn = np.mean(src, axis=1)[:, np.newaxis]
        out = np.absolute(src - mn)
        return self.rescale(out)

    # morphological filtering
    def morph(self, src):
        src = (self.rescale(src) * 255).astype("uint8")
        se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
        se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        mask = cv2.morphologyEx(src, cv2.MORPH_CLOSE, se1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)
        return self.rescale(mask)


class AdvancedProcessing(CustomProcessing):
    def __init__(self, kernel_size=5, sigma=1, threshold=0.5, erode_kernel=3):
        super().__init__(kernel_size, sigma, threshold, erode_kernel)

    def brightness_reconstruction(self, img):  # doi: 10.1109/TPS.2018.2828863.
        im_norm = img / 255
        img = np.average(im_norm, axis=None)
        img = np.log(im_norm + 1) * (im_norm - img)
        img = img / np.max(img)
        img = np.where(img < 0, 0, img)
        return img * 255

    def fourier_shifting(self, img):
        dft = np.fft.fft2(img, axes=(0, 1))
        dft_shift = np.fft.fftshift(dft)
        radius = 1
        mask = np.zeros_like(img)
        cy = mask.shape[0] // 2
        cx = mask.shape[1] // 2
        cv2.circle(mask, (cx, cy), radius, (255, 255, 255), -1)[0]
        mask = 255 - mask
        dft_shift_masked = np.multiply(dft_shift, mask) / 255
        back_ishift = np.fft.ifftshift(dft_shift)
        back_ishift_masked = np.fft.ifftshift(dft_shift_masked)
        img_back = np.fft.ifft2(back_ishift, axes=(0, 1))
        img_filtered = np.fft.ifft2(back_ishift_masked, axes=(0, 1))
        img_back = np.abs(img_back).clip(0, 255).astype(np.uint8)
        img_filtered = np.abs(3 * img_filtered).clip(0, 255).astype(np.uint8)
        return img_filtered

    def prob_to_edge(self, image, threshold):
        ratio = np.amax(image) / 255
        img8 = (image / ratio).astype("uint8")
        edge_ = cv2.Canny(img8, threshold[0], threshold[1])
        return edge_

    def dark_filter(self, img):
        img = np.where(img < 5, 0, img)
        return img

    def canny(self, img):
        # gray=(255-255*(img-np.min(img))/(np.max(img)-np.min(img))).astype('uint8')

        # reduce the noise using Gaussian filters
        kernel_size = 11
        blur_gray = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

        # Apply Canny edge detctor
        low_threshold = 10
        high_threshold = 20
        edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

        return edges

    def process_image(self, img):
        img = cv2.GaussianBlur(img, (self.kernel_size, self.kernel_size), 0)
        img = self.brightness_reconstruction(img)
        img = dip.MatchedFiltersLineDetector2D(img, sigma=self.sigma)
        img = np.array(img)  # 10.1109/42.34715
        img *= 255.0 / img.max()
        img = self.brightness_reconstruction(img)
        img = np.where(img < self.threshold, 0, 1).astype("uint8")
        img = cv2.erode(
            img, np.ones((self.erode_kernel, self.erode_kernel), np.uint8), iterations=1
        )
        return img


class HomomorphicFilter:
    """Homomorphic filter implemented with diferents filters and an option to an external filter.

    High-frequency filters implemented:
        butterworth
        gaussian

    Attributes:
        a, b: Floats used on emphasis filter:
            H = a + b*H

        .
    """

    def __init__(self, a=0.5, b=1.5):
        self.a = float(a)
        self.b = float(b)

    # Filters
    def __butterworth_filter(self, I_shape, filter_params):
        P = I_shape[0] / 2
        Q = I_shape[1] / 2
        U, V = np.meshgrid(
            range(I_shape[0]), range(I_shape[1]), sparse=False, indexing="ij"
        )
        Duv = (((U - P) ** 2 + (V - Q) ** 2)).astype(float)
        H = 1 / (1 + (Duv / filter_params[0] ** 2) ** filter_params[1])
        return 1 - H

    def __gaussian_filter(self, I_shape, filter_params):
        P = I_shape[0] / 2
        Q = I_shape[1] / 2
        H = np.zeros(I_shape)
        U, V = np.meshgrid(
            range(I_shape[0]), range(I_shape[1]), sparse=False, indexing="ij"
        )
        Duv = (((U - P) ** 2 + (V - Q) ** 2)).astype(float)
        H = np.exp((-Duv / (2 * (filter_params[0]) ** 2)))
        return 1 - H

    # Methods
    def __apply_filter(self, I, H):
        H = np.fft.fftshift(H)
        I_filtered = (self.a + self.b * H) * I
        return I_filtered

    def filter(self, I, filter_params, filter="butterworth", H=None):
        """
        Method to apply homormophic filter on an image

        Attributes:
            I: Single channel image
            filter_params: Parameters to be used on filters:
                butterworth:
                    filter_params[0]: Cutoff frequency
                    filter_params[1]: Order of filter
                gaussian:
                    filter_params[0]: Cutoff frequency
            filter: Choose of the filter, options:
                butterworth
                gaussian
                external
            H: Used to pass external filter
        """

        #  Validating image
        if len(I.shape) != 2:
            raise Exception("Improper image")

        # Take the image to log domain and then to frequency domain
        I_log = np.log1p(np.array(I, dtype="float"))
        I_fft = np.fft.fft2(I_log)

        # Filters
        if filter == "butterworth":
            H = self.__butterworth_filter(
                I_shape=I_fft.shape, filter_params=filter_params
            )
        elif filter == "gaussian":
            H = self.__gaussian_filter(I_shape=I_fft.shape, filter_params=filter_params)
        elif filter == "external":
            print("external")
            if len(H.shape) != 2:
                raise Exception("Invalid external filter")
        else:
            raise Exception("Selected filter not implemented")

        # Apply filter on frequency domain then take the image back to spatial domain
        I_fft_filt = self.__apply_filter(I=I_fft, H=H)
        I_filt = np.fft.ifft2(I_fft_filt)
        I = np.exp(np.real(I_filt)) - 1
        return np.uint8(I)
