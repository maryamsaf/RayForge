"""
Main script - Simple modular version
"""
import os
import argparse
import SimpleITK as sitk
from config import Config

from config import Config
from data_loader import (
    load_subject_by_index, construct_full_paths,
    load_dicom_itk, get_instance_number_mapping, resample_isotropic
)
from segmentation import build_lung_mask_seeded_itk, build_lung_mask_top2_itk
from enhancement import (
    compute_vessel_blob, compute_tumor_probability,
    build_shell_volume, build_inner_volume
)
from visualization import (
    load_bounding_boxes, sitk_to_vtk, build_renderer
)


def process_subject_by_index(index, config):
    """Process a subject by index number."""
    print("=" * 80)
    print(f"PROCESSING SUBJECT BY INDEX: {index}")
    print("=" * 80)

    # Load subject info from master CSV
    subject_info = load_subject_by_index(index, config.MASTER_CSV_PATH)


    # Construct full paths
    paths = construct_full_paths(
        subject_info, config.DICOM_BASE_PATH, config.ANNOTATION_BASE_PATH
    )

    # Check if paths exist
    if not os.path.exists(paths['dicom_path']):
        print(f"\n[ERROR] DICOM path does not exist: {paths['dicom_path']}")
        return

    if not os.path.exists(paths['annotation_csv_path']):
        print(f"\n[WARNING] Annotation CSV not found: {paths['annotation_csv_path']}")
        print("Continuing without bounding boxes...")
        bbox_csv = None
    else:
        bbox_csv = paths['annotation_csv_path']

    # Load DICOM
    print("\n" + "=" * 80)
    print("LOADING DICOM DATA")
    print("=" * 80)
    img, files = load_dicom_itk(paths['dicom_path'])
    instance_map = get_instance_number_mapping(files)
    original_spacing = img.GetSpacing()

    # Resample
    iso_spacing = 0.9
    img_iso = resample_isotropic(img, iso_spacing)

    # Lung segmentation
    print("\n" + "=" * 80)
    print("LUNG SEGMENTATION")
    print("=" * 80)
    lung_mask = build_lung_mask_seeded_itk(img_iso)
    if lung_mask is None:
        lung_mask = build_lung_mask_top2_itk(img_iso)
    lung_mask = sitk.Cast(lung_mask > 0, sitk.sitkUInt8)

    # Enhancement
    print("\n" + "=" * 80)
    print("VESSEL AND TUMOR ENHANCEMENT")
    print("=" * 80)
    vessel, blob = compute_vessel_blob(img_iso, lung_mask)
    tumor = compute_tumor_probability(img_iso, vessel, blob, lung_mask)

    # Build volumes
    print("\n" + "=" * 80)
    print("BUILDING VOLUMES")
    print("=" * 80)
    inner = build_inner_volume(img_iso, vessel, tumor, lung_mask)
    shell = build_shell_volume(lung_mask)

    # Load bounding boxes
    bbox_df = None
    if bbox_csv:
        bbox_df = load_bounding_boxes(bbox_csv)

    # ITK → VTK
    print("\n" + "=" * 80)
    print("PREPARING VISUALIZATION")
    print("=" * 80)
    vtk_inner = sitk_to_vtk(inner)
    vtk_shell = sitk_to_vtk(shell)

    # Render
    print("\n" + "=" * 80)
    print("LAUNCHING 3D VIEWER")
    print("=" * 80)
    win, iren = build_renderer(vtk_inner, vtk_shell, bbox_df,
                               instance_map, original_spacing)
    win.Render()
    iren.Start()


def main():
    parser = argparse.ArgumentParser(
        description='Process CT scan by index number from master CSV'
    )

    parser.add_argument(
        'index',
        type=int,
        nargs='?',
        help='Index number (1-376) from the master CSV'
    )

    parser.add_argument(
        '--dicom-base',
        default=Config.DICOM_BASE_PATH,
        help='Base directory for DICOM files'
    )

    parser.add_argument(
        '--annotation-base',
        default=Config.ANNOTATION_BASE_PATH,
        help='Base directory for annotation CSV files'
    )

    parser.add_argument(
        '--csv',
        default=Config.MASTER_CSV_PATH,
        help='Path to master CSV file'
    )

    args = parser.parse_args()

    # Apply CLI overrides globally
    Config.set_paths(
        dicom_base=args.dicom_base,
        annotation_base=args.annotation_base,
        master_csv=args.csv
    )

    # From here on, ALWAYS use:
    # Config.DICOM_BASE_PATH
    # Config.ANNOTATION_BASE_PATH
    # Config.MASTER_CSV_PATH

    args = parser.parse_args()

    # Set configuration
    Config.set_paths(args.dicom_base, args.annotation_base, args.csv)

    # Get index number
    if args.index is None:
        try:
            index = int(input("Enter subject index number (1-376): "))
        except ValueError:
            print("Invalid index number!")
            return
    else:
        index = args.index

    # Validate index
    if index < 1 or index > 376:
        print(f"Error: Index must be between 1 and 376, got {index}")
        return

    # Process the subject
    try:
        process_subject_by_index(index, Config)
    except Exception as e:
        print(f"\n[ERROR] Failed to process subject: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()