import os
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd


class PreprocessUtilities:
    @staticmethod
    def crop_volume(volume, min_bounds, max_bounds):
        cropped_volume = volume[min_bounds[0]:max_bounds[0]+1,
                                min_bounds[1]:max_bounds[1]+1,
                                min_bounds[2]:max_bounds[2]+1]
        return cropped_volume

    @staticmethod
    def reshape_input_torch(volume):
        reshaped = volume.permute(3, 0, 1, 2)
        reshaped = reshaped.unsqueeze(-1)
        return reshaped

    @staticmethod
    def find_bounds(volume):
        nonzero_indices = torch.nonzero(volume > 0)
        if nonzero_indices.nelement() == 0:
            return None, None
        min_bounds = torch.min(nonzero_indices, dim=0).values
        max_bounds = torch.max(nonzero_indices, dim=0).values
        return min_bounds, max_bounds

    @staticmethod
    def update_global_bounds(global_min, global_max, min_bounds, max_bounds):
        if global_min is None or global_max is None:
            return min_bounds, max_bounds
        new_global_min = torch.min(global_min, min_bounds)
        new_global_max = torch.max(global_max, max_bounds)
        return new_global_min, new_global_max

    @staticmethod
    def load_nii_image(folder, file_name):
        file_path = os.path.join(folder, file_name)
        image_nifti = nib.load(file_path)
        return image_nifti.get_fdata()

    @staticmethod
    def calculate_bounds(mask_folder):
        global_max, global_min = None, None
        for i, file_name in enumerate(os.listdir(mask_folder)):
            if file_name.endswith('.nii.gz'):
                mask_image = torch.from_numpy(
                    PreprocessUtilities.load_nii_image(
                        mask_folder, 
                        file_name
                    )
                )
                # Find volumes
                min_bounds, max_bounds = PreprocessUtilities.find_bounds(mask_image)
                # Calculate bounds globally
                if min_bounds is not None and max_bounds is not None:
                    global_min, global_max = PreprocessUtilities.update_global_bounds(
                        global_min, 
                        global_max, 
                        min_bounds, 
                        max_bounds
                    )
        return global_min, global_max

    @staticmethod
    def repeat_permute_images(image):
        image = image.repeat(3, 1, 1, 1) 
        image = image.permute(0, 3, 1, 2)
        return image

    @staticmethod
    def check_shapes(v1, v2):
        if len(v1) != len(v2):
            return False
        for i in range(len(v1)):
            if v1[i] != v2[i]:
                return False
        return True

    @staticmethod
    def preprocess_nii_images(folder:str , output_folder: str = "aocr2024/preprocessed_images/", bounds: list = None):
        min_shape_image = [
            int(bounds[1][0] - bounds[0][0]) + 1, 
            int(bounds[1][1] - bounds[0][1]) + 1, 
            int(bounds[1][2] - bounds[0][2]) + 1
        ]
        reductions_stored = []
        removed_images = 0
        for i, file_name in enumerate(os.listdir(folder)):
            if file_name.endswith('.nii.gz'):            
                image_array = PreprocessUtilities.load_nii_image(folder, file_name)
                image_array_torch = torch.from_numpy(image_array)
                shape_original = image_array_torch.shape
                size_original = (
                    shape_original[0] *
                    shape_original[1] *
                    shape_original[2]
                )
                preprocessed_image = PreprocessUtilities.crop_volume(
                    image_array_torch,
                    bounds[0],
                    bounds[1]
                )
                if not PreprocessUtilities.check_shapes(preprocessed_image.shape, min_shape_image):
                    removed_images += 1
                    continue
                preprocessed_image = PreprocessUtilities.repeat_permute_images(
                    preprocessed_image
                )
                shape_new = preprocessed_image.shape
                size_new = (
                    shape_new[0] *
                    shape_new[1] *
                    shape_new[2]
                )
                
                # Calculate ratio between original and new size
                ratio = size_new / size_original * 100
                reductions_stored.append(
                    ratio
                )
                numpy_array = preprocessed_image.cpu().numpy()
                nifti_image = nib.Nifti1Image(numpy_array, affine=np.eye(4))
                nib.save(nifti_image, f"{output_folder}/{file_name}")
                
        print(f"Preprocessed and stored {i} images")
        print(f"Percentage removed: {removed_images / i * 100:.2f}%")
        return reductions_stored
    
    @staticmethod
    def crop_images_with_calculated_bounds(folder_masks = "aocr2024/2_Train,Valid_Mask/"):
        # Get the global bounds
        global_min, global_max = PreprocessUtilities.calculate_bounds(folder_masks)
        # Crop the images with the new bounds
        print(f"Thresholds - min {global_min}, max {global_max}")
        ratios_training = PreprocessUtilities.preprocess_nii_images(
            folder = "aocr2024/1_Train,Valid_Image/", 
            bounds = [global_min,  global_max], 
            output_folder = "aocr2024/preprocessed_images/"
        )
        ratios_test = PreprocessUtilities.preprocess_nii_images(
            folder = "aocr2024/3_Test1_Image/", 
            bounds = [global_min,  global_max], 
            output_folder = "aocr2024/preprocessed_images_test/"
        )
        # TODO: store and report some metrics for ratio reductions.
            