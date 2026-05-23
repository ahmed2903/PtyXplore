import numpy as np
from skimage.registration import phase_cross_correlation
import scipy 

fft_images = lambda imgs: np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(imgs, axes=(-2, -1)), axes=(-2, -1)), axes = (-2,-1))
ifft_images = lambda imgs: np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(imgs, axes=(-2, -1)), axes=(-2, -1)), axes = (-2,-1))

def align_images(image_list, ref_img = None, ref_idx = None):
    
    aligned_images = []
    shifts = []
    
    if ref_img is None:
        try: 
            ref_img = image_list[ref_idx]
        except:
            N = len(image_list)
            ref_img = image_list[N//2]
    
    for img in image_list:
        # avg = np.mean(np.array(aligned_images), axis = 0)    
        shift, _, _ = phase_cross_correlation(ref_img, img, upsample_factor=10)  
        shifts.append(shift)
        
        aligned_img = scipy.ndimage.shift(img, shift)  # Apply shift
        aligned_images.append(aligned_img)

    aligned_images = np.array(aligned_images)
    shifts = np.array(shifts)
    
    return aligned_images, shifts

def align_images_from_shifts(images, shifts):
    aligned_images = []
    for img, shift in zip(images, shifts):
        aligned_img = scipy.ndimage.shift(img, shift)  # Apply shift
        aligned_images.append(aligned_img)

    return np.array(aligned_images)
    
def tilt_correction(images, tc_image = None, iters = 10):

    Ncoh = images.shape[0]    

    if tc_image is None:
        tc_image = images[Ncoh//2+1] 
    
    for i in range(iters):
        
        aligned_images, shifts = align_images(images, tc_image)
        tc_image = np.sum(aligned_images, axis = 0)

    return tc_image, shifts

