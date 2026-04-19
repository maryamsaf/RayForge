"""Data loading functions"""
import os
import pandas as pd
import SimpleITK as sitk


def load_subject_by_index(index, master_csv_path):
    """Load subject information from master CSV by index."""
    df = pd.read_csv(master_csv_path, encoding='latin-1', encoding_errors='ignore')
    df.columns = df.columns.str.replace('ï»¿', '').str.strip()
    row = df[df['Index'] == index]

    if len(row) == 0:
        raise ValueError(f"Index {index} not found in master CSV")

    row = row.iloc[0]
    subject_info = {
        'index': int(row['Index']),
        'subject_id': str(row['Subject_ID']).strip(),
        'dicom_path': str(row['DICOM_Path']).replace('Â', '').strip(),
        'annotation_dir': str(row['annotation_dir']).replace('Â', '').strip(),
    }

    print(f"\n[LOOKUP] Index {index}:")
    print(f"  Subject ID: {subject_info['subject_id']}")
    print(f"  DICOM Path: {subject_info['dicom_path']}")
    print(f"  Annotation: {subject_info['annotation_dir']}")

    return subject_info


def construct_full_paths(subject_info, dicom_base, annotation_base):
    """Construct full filesystem paths from subject info."""
    subject_id = subject_info['subject_id']
    dicom_path = os.path.join(dicom_base, subject_info['dicom_path'])
    annotation_filename = subject_info['annotation_dir'] + '.csv'
    annotation_csv_path = os.path.join(annotation_base, subject_id, annotation_filename)

    print(f"\n[PATHS] Constructed paths:")
    print(f"  DICOM: {dicom_path}")
    print(f"  Annotation CSV: {annotation_csv_path}")

    if not os.path.exists(dicom_path):
        print(f"  WARNING: DICOM path does not exist!")
    if not os.path.exists(annotation_csv_path):
        print(f"  WARNING: Annotation CSV does not exist!")

    return {
        'dicom_path': dicom_path,
        'annotation_csv_path': annotation_csv_path
    }


def load_dicom_itk(dicom_path):
    """Load DICOM series using SimpleITK"""
    print("[ITK] Reading DICOM series from:", dicom_path)
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dicom_path)
    if not series_ids:
        raise RuntimeError("No DICOM series found")

    files = reader.GetGDCMSeriesFileNames(dicom_path, series_ids[0])
    reader.SetFileNames(files)
    reader.LoadPrivateTagsOn()
    reader.MetaDataDictionaryArrayUpdateOn()
    img = reader.Execute()
    print("[ITK] Size:", img.GetSize(), "Spacing:", img.GetSpacing())
    print("[ITK] Origin:", img.GetOrigin())
    return img, files


def get_instance_number_mapping(files):
    """Create mapping from instance number to slice index."""
    reader = sitk.ImageFileReader()
    instance_map = {}

    for idx, filename in enumerate(files):
        reader.SetFileName(filename)
        reader.LoadPrivateTagsOn()
        reader.ReadImageInformation()
        try:
            instance_num = int(reader.GetMetaData("0020|0013"))
            instance_map[instance_num] = idx
        except:
            pass

    print(f"[DICOM] Instance mapping: {instance_map}")
    return instance_map


def resample_isotropic(image, iso_spacing=0.9):
    """Resample image to isotropic spacing"""
    orig_spacing = image.GetSpacing()
    orig_size = image.GetSize()

    new_size = [
        int(round(osz * osp / iso_spacing))
        for osz, osp in zip(orig_size, orig_spacing)
    ]

    res = sitk.Resample(
        image, new_size, sitk.Transform(), sitk.sitkLinear,
        image.GetOrigin(), (iso_spacing, iso_spacing, iso_spacing),
        image.GetDirection(), -1024, image.GetPixelID()
    )
    print(f"[RESAMPLE] Original size: {orig_size}, spacing: {orig_spacing}")
    print(f"[RESAMPLE] New size: {new_size}, spacing: {(iso_spacing, iso_spacing, iso_spacing)}")
    return res