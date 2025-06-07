import cv2
import numpy as np
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    '''
    Custom Dataset for loading image data and corresponding numerical inputs.
    
    Attributes:
        train_params (np.array): Array containing image file references and positional data.
        train_label (np.array): Array containing labels for each sample.
        img_size (tuple): Target image size (width, height).
        sliding_window (int): Number of consecutive frames to process.
        add_imgs (int): Additional images flag (unused in current implementation).
        max_verpix (int): Maximum vertical pixels for image framing.
        max_horpix (int): Maximum horizontal pixels for image framing.
        aug (int): Augmentation flag (0 indicates no augmentation).
        images_dict (dict): Pre-loaded dictionary of images.
    '''
    def __init__(self, train_params, train_label, img_size, sliding_window,
                 add_imgs, max_verpix, max_horpix, aug, images_dict):
        self.train_params = train_params
        self.train_label = train_label
        self.img_size = img_size
        self.sliding_window = sliding_window
        self.add_imgs = add_imgs
        self.max_verpix = max_verpix
        self.max_horpix = max_horpix
        self.aug = aug
        self.images_dict = images_dict

    def __len__(self):
        '''Return the total number of samples.'''
        return len(self.train_params)
    
    def load_image(self, path, aug, seed_num):
        ''' 
        Load an image from the pre-loaded dictionary and apply preprocessing.
        
        Args:
            path (str): Key to retrieve the image from the images dictionary.
            aug (int): Augmentation flag.
            seed_num (int): Random seed (not used in current implementation).
            
        Returns:
            np.array: Processed image frame.
        '''
        img = self.images_dict[path]
        # No augmentation is applied when aug == 0.
        img = img[:, :, 0] / 255.0
        img_frame = np.zeros([self.max_verpix + 4, self.max_horpix + 10])
        img_frame[-img.shape[0]-2:-2, 2:img.shape[1]+2] = img
        img_frame = cv2.resize(img_frame, (self.img_size[0], self.img_size[1]))
        return img_frame

    def __getitem__(self, idx):
        ''' 
        Retrieve a sample (images and numerical data) and its corresponding label.
        
        Args:
            idx (int): Index of the sample.
            
        Returns:
            tuple: (inputs, label) where inputs is a dictionary with image and numerical data.
        '''
        import numpy as np
        seed_num = int(np.random.uniform(1, 20000))
        images = [self.load_image(self.train_params[idx, j, 0], aug=self.aug, seed_num=seed_num)
                  for j in range(self.sliding_window)]
        images = np.array(images)
        images = images.reshape((self.sliding_window, self.img_size[1], self.img_size[0], 1))
        images = np.transpose(images, (0, 3, 1, 2))
        poss = self.train_params[idx, :, 1]
        label = self.train_label[idx]
        return {'image_input': images, 'number_input': poss.astype(float)}, label.astype(float)

