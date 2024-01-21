import os
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import time

from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from pydantic import BaseModel

# TODO: move to another file
class ClassificationMetrics():
    def __init__(self, tp: int, tn: int, fp: int, fn: int):
        self.true_positives = tp
        self.true_negatives = tn
        self.false_positives = fp
        self.false_negatives = fn

    @property
    def precision(self) -> float:
        try:
            return self.true_positives / (self.true_positives + self.false_positives)
        except ZeroDivisionError:
            return 1.0

    @property
    def recall(self) -> float:
        try:
            return self.true_positives / (self.true_positives + self.false_negatives)
        except ZeroDivisionError:
            return 1.0

    @property
    def accuracy(self) -> float:
        try:
            return (self.true_positives + self.true_negatives) / \
                   (self.true_positives + self.true_negatives + self.false_positives + self.false_negatives)
        except ZeroDivisionError:
            return 0.0

    @property
    def f1_score(self) -> float:
        try:
            precision = self.precision
            recall = self.recall
            return 2 * (precision * recall) / (precision + recall)
        except ZeroDivisionError:
            return 0.0
        
    def __add__(self, other):
        if not isinstance(other, ClassificationMetrics):
            return NotImplemented
        return ClassificationMetrics(
            self.true_positives + other.true_positives,
            self.true_negatives + other.true_negatives,
            self.false_positives + other.false_positives,
            self.false_negatives + other.false_negatives,
        )

    def __str__(self):
        return f"tp: {self.true_positives} fp: {self.false_positives} tn: {self.true_negatives} fn: {self.false_negatives}"

        
class MetricUtils:
    @staticmethod
    def calculate_metrics(pred: list[int], real: list[int]) -> ClassificationMetrics:
        pred_real = zip(pred, real)
        tp, tn, fp, fn = 0, 0, 0, 0
        for p, r in pred_real:
            p = int(p)
            r = int(r)
            tp += (p == r == 1)
            tn += (p == r == 0)
            fp += ((p == 1) and (r == 0))
            fn += ((p == 0) and (r == 1))
        return ClassificationMetrics(tp, tn, fp, fn)

class PreprocessUtilities:
    @staticmethod
    def crop_volume(volume, min_bounds, max_bounds, filter_depth = True):
        if filter_depth:
            cropped_volume = volume[min_bounds[0]:max_bounds[0]+1,
                                    min_bounds[1]:max_bounds[1]+1,
                                    min_bounds[2]:max_bounds[2]+1]
        else:
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
    def preprocess_nii_images(
        folder:str, 
        output_folder: str = "aocr2024/preprocessed_images/", 
        bounds: list = None, 
        filter_depth: bool = True
    ):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"Created: {output_folder}")
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
                    bounds[1], 
                    filter_depth
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
    
    def convert_3d_to_2d(input_folder, output_folder):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        files = os.listdir(input_folder)
        total_files = len(files)
        progress_bar = tqdm(files, total=total_files)
        for file_name in progress_bar:
            if file_name.endswith('.nii.gz'):
                file_path = os.path.join(input_folder, file_name)

                img_3d = nib.load(file_path)
                data_3d = img_3d.get_fdata()
                file_name_splitted = file_name.split(".")
                original_image_id = file_name_splitted[0][:-6]
                for i in range(data_3d.shape[2]):
                    slice_2d = data_3d[:, :, i]
                    slice_img = nib.Nifti1Image(slice_2d, affine=img_3d.affine)
                    file_name_splitted[0] = f"{original_image_id}_{i}"
                    slice_file_name = ".".join(file_name_splitted)
                    nib.save(slice_img, os.path.join(output_folder, slice_file_name))
            progress_bar.set_postfix(
                {
                    'lastest_file_processed': file_name
                }
            )
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
    
    @staticmethod
    def has_ones(
        volume
    ):
        nonzero_indices = torch.nonzero(volume > 0)
        return 0 if nonzero_indices.nelement() == 0 else 0
    
    @staticmethod
    def calculate_labels_based_on_masks(
        output_file = "aocr2024/TrainValid_ground_truth_slices.csv", 
        output_file_valid_split = "aocr2024/TrainValid_split_slices.csv",
        valid_split_file = "aocr2024/TrainValid_split.csv",
        folder_masks = "aocr2024/2_Train,Valid_Mask/"
    ):
        df_valid_split = pd.read_csv(valid_split_file)
        images = []
        classes = []
        group = []
        files = os.listdir(folder_masks)
        total_files = len(files)
        progress_bar = tqdm(files, total=total_files)
        for file_name in progress_bar:
            if file_name.endswith('.nii.gz'):
                file_path = os.path.join(folder_masks, file_name)
                img_3d = nib.load(file_path)
                data_3d = torch.from_numpy(img_3d.get_fdata())
                file_name_splitted = file_name.split(".")
                original_image_id = file_name_splitted[0][:-6]
                group_image = str(
                    df_valid_split.loc[
                        df_valid_split.id == original_image_id,
                        "group"
                    ].iloc[0]
                )
                for i in range(data_3d.shape[2]):
                    _class = torch.any(data_3d[:, :, i])
                    image_filename = f"{original_image_id}_{i}"
                    images.append(image_filename)
                    classes.append(int(_class))
                    group.append(group_image)
            progress_bar.set_postfix({'lastest_file_processed': file_name})
        # Ground truth
        pd.DataFrame(zip(images,classes),
            columns = [
                "id",
                "label"
            ]
        ).to_csv(output_file, index = False)
        # Valid split
        pd.DataFrame(zip(images,classes, group),
            columns = [
                "id",
                "slice-level-label",
                "group"
            ]
        ).to_csv(output_file_valid_split, index = False)

    # For slice prediction
    @staticmethod
    def images_from_3d_to_2d(
        folder_images_train = "aocr2024/1_Train,Valid_Image/", 
        folder_images_test = "aocr2024/3_Test1_Image/",
        folder_masks = "aocr2024/2_Train,Valid_Mask/"
    ):
        # Calculate the volume, but in this case we only filter the w and h
        start_time = time.time()
        global_min, global_max = PreprocessUtilities.calculate_bounds(folder_masks)
        print(f"Thresholds - min {global_min}, max {global_max}")
        end_time = time.time()
        print(f"Time calculate bounds: {end_time - start_time:.2f}s")
        # Crop the images with the new bounds
        start_time = time.time()
        ratios_training = PreprocessUtilities.preprocess_nii_images(
            folder = folder_images_train, 
            bounds = [global_min,  global_max], 
            output_folder = "aocr2024/preprocessed_images_2d_tmp/",
            filter_depth = False
        )
        end_time = time.time()
        print(f"Time preprocess training images (volumes): {end_time - start_time:.2f}s")
        start_time = time.time()
        ratios_test = PreprocessUtilities.preprocess_nii_images(
            folder = folder_images_test, 
            bounds = [global_min,  global_max], 
            output_folder = "aocr2024/preprocessed_images_test_2d_tmp/",
            filter_depth = False
        )
        end_time = time.time()
        print(f"Time preprocess tests images (volumes): {end_time - start_time:.2f}s")
        # From 3d to 2d
        print(f"From 3d to 2d the training images")
        start_time = time.time()
        PreprocessUtilities.convert_3d_to_2d(
            input_folder = "aocr2024/preprocessed_images_2d_tmp/",
            output_folder = "aocr2024/preprocessed_images_2d/"
        )
        end_time = time.time()
        print(f"Time convert training images from 3d to 2d: {end_time - start_time:.2f}s")
        print(f"From 3d to 2d the test images")
        start_time = time.time()
        PreprocessUtilities.convert_3d_to_2d(
            input_folder = "aocr2024/preprocessed_images_test_2d_tmp/",
            output_folder = "aocr2024/preprocessed_images_test_2d/"
        )
        end_time = time.time()
        print(f"Time convert testing images from 3d to 2d: {end_time - start_time:.2f}s")
        # Adding the labels:
        start_time = time.time()
        PreprocessUtilities.calculate_labels_based_on_masks(
            folder_masks = folder_masks,
            output_file = "aocr2024/TrainValid_ground_truth_slices.csv"
        )
        end_time = time.time()
        print(f"Time calculate the labels by using the masks: {end_time - start_time:.2f}s")
