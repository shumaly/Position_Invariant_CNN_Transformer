import cv2
import numpy as np
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    '''
    Custom Dataset for loading images along with their corresponding numerical inputs.
    
    Attributes:
        params (np.array): Array containing image file references and positional data.
        labels (np.array): Array containing labels for each sample.
        img_size (tuple): Target image size (width, height).
        sliding_window (int): Number of consecutive frames to process.
        max_verpix (int): Maximum vertical pixels for image framing.
        max_horpix (int): Maximum horizontal pixels for image framing.
        aug (bool): Flag indicating whether to apply augmentation.
        aug_intensity (float): Intensity factor for augmentation (default 1.0 reproduces current behavior).
        images_dict (dict): Pre-loaded dictionary of images.
    '''
    def __init__(self, params, labels, img_size, sliding_window, max_verpix, max_horpix, aug, aug_intensity, images_dict):
        self.params = params
        self.labels = labels
        self.img_size = img_size
        self.sliding_window = sliding_window
        self.max_verpix = max_verpix
        self.max_horpix = max_horpix
        self.aug = aug
        self.aug_intensity = aug_intensity
        self.images_dict = images_dict

    def __len__(self):
        '''Return the total number of samples.'''
        return len(self.params)
    
    def img_aug(self, img, seed_num):
        '''
        Apply a sequence of augmentations to the input image.
        Uses functions from Models.augmentation and adjusts parameters based on aug_intensity.
        
        Args:
            img (np.array): Input image.
            seed_num (int): Random seed.
            
        Returns:
            np.array: Augmented image.
        '''
        # Import augmentation functions.
        from Models.augmentation import (glow_spot, make_stain, generate_circular_gradient,
                                         locate_stain)
        # Set random seed for reproducibility.
        np.random.seed(seed_num)
        # Apply glow spot augmentation.
        img = glow_spot(img, max_radius=int(2 * self.aug_intensity))
        # Create and apply stain augmentation.
        stain_size = int(4 * self.aug_intensity)
        stain = make_stain(image_size=stain_size)
        filtered_gradient = generate_circular_gradient(size=stain_size) * (1 - stain)
        filtered_gradient[filtered_gradient == 0] = 1
        img = locate_stain(img, filtered_gradient, image_size=stain_size)
        # Apply Gaussian blur with random kernel size.
        blur_amount = int(np.random.choice([1, 3, 5], size=1)[0])
        img = cv2.GaussianBlur(img, (blur_amount, blur_amount), 0)
        # Adjust brightness randomly.
        brightness_factor = np.random.uniform(0.85, 1.15)
        img = cv2.convertScaleAbs(img, alpha=brightness_factor, beta=0)
        return img

    def load_image(self, path, aug, seed_num):
        '''
        Load an image from the images dictionary, optionally apply augmentation,
        and preprocess it to fit into a fixed frame.
        
        Args:
            path (str): Key to retrieve the image.
            aug (bool): Whether to apply augmentation.
            seed_num (int): Random seed.
            
        Returns:
            np.array: Preprocessed image.
        '''
        img = self.images_dict[path]
        if aug:
            img = self.img_aug(img, seed_num)
        # Normalize and place the image in a fixed frame.
        img = img[:, :, 0] / 255.0
        img_frame = np.zeros([self.max_verpix + 4, self.max_horpix + 10])
        img_frame[-img.shape[0]-2:-2, 2:img.shape[1]+2] = img
        img_frame = cv2.resize(img_frame, (self.img_size[0], self.img_size[1]))
        return img_frame

    def __getitem__(self, idx):
        '''
        Retrieve a sample (images and corresponding numerical inputs) and its label.
        
        Args:
            idx (int): Index of the sample.
            
        Returns:
            tuple: ({'image_input': images, 'number_input': numerical data}, label)
        '''
        seed_num = int(np.random.uniform(1, 20000))
        images = [self.load_image(self.params[idx, j, 0], self.aug, seed_num)
                  for j in range(self.sliding_window)]
        images = np.array(images).reshape((self.sliding_window, self.img_size[1], self.img_size[0], 1))
        images = np.transpose(images, (0, 3, 1, 2))
        poss = self.params[idx, :, 1]
        label = self.labels[idx]
        return {'image_input': images, 'number_input': poss.astype(float)}, label.astype(float)

