import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import os

class SimpleCTInference:
    def __init__(self, checkpoint_path):
        """
        Initialize the inference model for a single CT image.
        
        Args:
            checkpoint_path: Path to the model checkpoint
        """
        # Load the SegClsModel from the original code
        from inference import SegClsModel
        self.model = SegClsModel(segcls_checkpoint=checkpoint_path)
        self.model.segcls_module.eval()
        
    def process_image(self, ct_image, patient_id="unknown", slice_id="0001", 
                      save_result=False, output_dir=None):
        """
        Process a single CT image.
        
        Args:
            ct_image: Numpy array of CT image data (512x512)
            patient_id: Identifier for the patient/case
            slice_id: Identifier for this specific slice
            save_result: Whether to save the visualization
            output_dir: Directory to save results if save_result is True
            
        Returns:
            Dictionary with inference results:
            - seg_nodule: Segmentation mask for nodules
            - visualization: Visualization of the results
            - cls_lung_pos: Lung position classification results (if enabled)
            - seg_lung_loc: Lung location segmentation (if enabled)
            - cls_nodule: Nodule classification results (if enabled)
        """
        # Prepare datapoint according to what infer_slice expects
        datapoint = {
            "slice": ct_image.astype(np.float32),
            "bnid": patient_id,
            "raw_id": slice_id,
            "seg_nodule": None,
            "nodule_info": None
        }
        
        # Create output directory if needed
        if save_result and output_dir:
            des_path = Path(output_dir)
            des_path.mkdir(parents=True, exist_ok=True)
        else:
            des_path = Path("./output")
        
        # Run inference using the existing function
        from inference import infer_slice
        results = infer_slice(self.model, datapoint, des_path, 
                            show_result=False, save_result=save_result)
        
        # Normalize image for visualization
        norm_image = self.model.ct_normalize(ct_image)
        visualization = np.stack([norm_image, norm_image, norm_image], axis=-1)
        
        # Draw segmentation mask on visualization
        if results["seg_nodule"].sum() > 0:
            contours, _ = self.find_bbox(results["seg_nodule"])
            cv2.drawContours(visualization, contours, -1, (1., 1., 0.), 1)
        
        # Add visualization to results
        results["visualization"] = visualization
        
        return results
    
    def find_bbox(self, mask):
        """Utility function to find bounding boxes from mask"""
        mask = (mask > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours, [cv2.boundingRect(contour) for contour in contours]
    
    def display_results(self, results):
        """
        Display visualization of results.
        
        Args:
            results: Results dictionary from process_image
        """
        plt.figure(figsize=(10, 10))
        plt.imshow(results["visualization"])
        plt.axis('off')
        plt.title(f"Nodule Detection Results")
        plt.show()


# Example usage
if __name__ == "__main__":
    # Path to your checkpoint
    checkpoint_path = "weights/4.4/epoch_004.ckpt"
    
    # Initialize inference model
    inference = SimpleCTInference(checkpoint_path)
    
    # Load a sample CT image (replace with your actual loading code)
    # This could be from a DICOM file or any other format
    sample_npy_path = "data/kc_cancer_4.4/nhom_benh/LungDamage/KC_CT_0148/KC_CT_0148_LUNG_DAMAGE_0001_DAMAGE_0.npy"
    ct_image = np.load(sample_npy_path)  # Assumes shape (512, 512)
    
    # Process the image
    results = inference.process_image(
        ct_image, 
        patient_id="patient123",
        slice_id="series01_0001",
        save_result=True, 
        output_dir="./output"
    )
    
    # Display results
    inference.display_results(results)
    
    # Access specific results
    nodule_mask = results["seg_nodule"]
    print(f"Detected nodule pixels: {nodule_mask.sum()}")
    
    if "cls_lung_pos" in results:
        lung_pos_probs, _ = results["cls_lung_pos"]
        right_lung_prob = lung_pos_probs["right_lung"][0, 1].item()
        left_lung_prob = lung_pos_probs["left_lung"][0, 1].item()
        print(f"Right lung probability: {right_lung_prob:.4f}")
        print(f"Left lung probability: {left_lung_prob:.4f}")