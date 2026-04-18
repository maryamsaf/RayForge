"""Lung segmentation functions"""
import numpy as np
import pandas as pd
import SimpleITK as sitk


def load_bounding_box_from_csv(csv_path):
    """Load bounding box data from CSV/TSV file and compute 3D bbox.

    Args:
        csv_path: Path to CSV file with columns: inst, x_min, x_max, y_min, y_max

    Returns:
        Dictionary with 3D bounding box coordinates or None if loading fails
    """
    try:
        df = pd.read_csv(csv_path, sep='\t')
        if len(df.columns) == 1:
            df = pd.read_csv(csv_path)

        print(f"[BBOX CSV] Loaded {len(df)} slice annotations")
        print(f"[BBOX CSV] Columns: {df.columns.tolist()}")

        if len(df) == 0:
            print("[BBOX CSV] No annotations found")
            return None

        # Compute 3D bounding box from all 2D slices
        x_min_global = int(df['x_min'].min())
        x_max_global = int(df['x_max'].max())
        y_min_global = int(df['y_min'].min())
        y_max_global = int(df['y_max'].max())

        # Z coordinates from instance numbers
        z_min_global = int(df['inst'].min())
        z_max_global = int(df['inst'].max())

        bbox_3d = {
            'x_min': x_min_global,
            'x_max': x_max_global,
            'y_min': y_min_global,
            'y_max': y_max_global,
            'z_min': z_min_global,
            'z_max': z_max_global
        }

        print(f"\n[BBOX CSV] Computed 3D bounding box:")
        print(f"  X: [{x_min_global}, {x_max_global}]")
        print(f"  Y: [{y_min_global}, {y_max_global}]")
        print(f"  Z: [{z_min_global}, {z_max_global}]")

        return bbox_3d

    except Exception as e:
        print(f"[BBOX CSV] Error loading CSV: {e}")
        return None


def dilate_bbox_region_in_hu(hu_image, bbox_3d, dilation_value=250, margin=3):
    """Dilate (brighten) the region within bounding box in HU image.

    This function increases HU values in the tumor region to make it more prominent
    in subsequent processing and visualization.

    Args:
        hu_image: Original HU image (SimpleITK)
        bbox_3d: Dictionary with keys: x_min, x_max, y_min, y_max, z_min, z_max
        dilation_value: HU value to add to the region (default: 200)
        margin: Safety margin in pixels to add around bbox (default: 5)

    Returns:
        Modified HU image with dilated tumor region
    """
    if bbox_3d is None:
        print("[HU DILATION] No bounding box provided, skipping dilation")
        return hu_image

    # Convert to numpy array for manipulation
    hu_array = sitk.GetArrayFromImage(hu_image)

    # Extract bounding box coordinates with safety margins
    x_min = max(0, bbox_3d['x_min'] - margin)
    x_max = min(hu_array.shape[2], bbox_3d['x_max'] + margin)
    y_min = max(0, bbox_3d['y_min'] - margin)
    y_max = min(hu_array.shape[1], bbox_3d['y_max'] + margin)
    z_min = max(0, bbox_3d['z_min'] - margin)
    z_max = min(hu_array.shape[0], bbox_3d['z_max'] + margin)

    print(f"\n[HU DILATION] Dilating region:")
    print(f"  X: [{x_min}, {x_max}]")
    print(f"  Y: [{y_min}, {y_max}]")
    print(f"  Z: [{z_min}, {z_max}]")
    print(f"  Adding {dilation_value} HU to region")

    # Get original statistics in the region
    roi_original = hu_array[z_min:z_max, y_min:y_max, x_min:x_max]
    original_mean = roi_original.mean()
    original_max = roi_original.max()

    # Add dilation value to the region
    hu_array[z_min:z_max, y_min:y_max, x_min:x_max] += dilation_value

    # Get new statistics
    roi_modified = hu_array[z_min:z_max, y_min:y_max, x_min:x_max]
    new_mean = roi_modified.mean()
    new_max = roi_modified.max()

    print(f"[HU DILATION] Original ROI - Mean: {original_mean:.1f} HU, Max: {original_max:.1f} HU")
    print(f"[HU DILATION] Modified ROI - Mean: {new_mean:.1f} HU, Max: {new_max:.1f} HU")

    # Convert back to SimpleITK image
    dilated_hu = sitk.GetImageFromArray(hu_array)
    dilated_hu.CopyInformation(hu_image)

    return dilated_hu


def dilate_bbox_region_morphological(hu_image, bbox_3d, kernel_radius=[3, 3, 1], margin=5):
    """Apply morphological dilation only within bounding box region.

    This uses actual morphological dilation on the HU values in the tumor region.

    Args:
        hu_image: Original HU image (SimpleITK)
        bbox_3d: Dictionary with keys: x_min, x_max, y_min, y_max, z_min, z_max
        kernel_radius: Dilation kernel size [x, y, z] (default: [3, 3, 1])
        margin: Safety margin in pixels to add around bbox (default: 5)

    Returns:
        HU image with morphologically dilated tumor region
    """
    if bbox_3d is None:
        print("[MORPH DILATION] No bounding box provided, skipping dilation")
        return hu_image

    hu_image = sitk.Cast(hu_image, sitk.sitkFloat32)

    # Extract bounding box coordinates with margins
    x_min = max(0, bbox_3d['x_min'] - margin)
    x_max = min(hu_image.GetSize()[0], bbox_3d['x_max'] + margin)
    y_min = max(0, bbox_3d['y_min'] - margin)
    y_max = min(hu_image.GetSize()[1], bbox_3d['y_max'] + margin)
    z_min = max(0, bbox_3d['z_min'] - margin)
    z_max = min(hu_image.GetSize()[2], bbox_3d['z_max'] + margin)

    # Calculate ROI size
    roi_size = [x_max - x_min, y_max - y_min, z_max - z_min]
    roi_index = [x_min, y_min, z_min]

    print(f"\n[MORPH DILATION] Extracting ROI:")
    print(f"  Position: x={x_min}, y={y_min}, z={z_min}")
    print(f"  Size: {roi_size}")
    print(f"  Kernel radius: {kernel_radius}")

    # Extract region of interest
    roi = sitk.RegionOfInterest(hu_image, roi_size, roi_index)

    # Apply grayscale dilation to ROI
    roi_dilated = sitk.GrayscaleDilate(roi, kernelRadius=kernel_radius)

    # Optional: smooth the dilated region
    roi_dilated = sitk.SmoothingRecursiveGaussian(roi_dilated, sigma=0.5)

    # Create output image (copy of original)
    output = sitk.Image(hu_image)

    # Paste dilated ROI back
    output = sitk.Paste(output, roi_dilated, roi_size, roi_index, roi_index)

    print(f"[MORPH DILATION] Morphological dilation completed")

    return output


def build_lung_mask_seeded_itk(iso_img):
    """Build lung mask using seeded region growing"""
    smooth = sitk.CurvatureFlow(iso_img, timeStep=0.04, numberOfIterations=10)

    arr = sitk.GetArrayFromImage(iso_img)
    z, y, x = arr.shape
    mz, my = z // 2, y // 2

    lung = np.logical_and(arr >= -1024, arr <= -350)

    def scan(xs, xe):
        L = -1
        R = -1
        step = 1 if xe >= xs else -1
        i = xs
        while i != xe:
            if lung[mz, my, i]:
                if L < 0:
                    L = i
                R = i
            elif L >= 0:
                break
            i += step
        return (L + R) // 2 if L >= 0 else -1

    lx = scan(int(x * 0.10), int(x * 0.45))
    rx = scan(int(x * 0.90), int(x * 0.55))

    seeds = []
    if lx >= 0:
        seeds.append((lx, my, mz))
    if rx >= 0:
        seeds.append((rx, my, mz))

    if not seeds:
        return None

    seg = sitk.ConnectedThreshold(
        smooth, seedList=seeds, lower=-990, upper=-250, replaceValue=1
    )

    seg = sitk.BinaryMorphologicalClosing(seg, [2, 2, 2])
    seg = sitk.BinaryFillhole(seg)
    return seg


def build_lung_mask_top2_itk(iso_img):
    """Build lung mask using top 2 connected components"""
    cand = sitk.BinaryThreshold(iso_img, -1024, -350, 1, 0)
    cand = sitk.BinaryMorphologicalClosing(cand, [2, 2, 2])
    cand = sitk.BinaryFillhole(cand)

    cc = sitk.ConnectedComponent(cand)
    rel = sitk.RelabelComponent(cc, sortByObjectSize=True)
    lungs = sitk.BinaryThreshold(rel, 1, 2, 1, 0)
    return lungs