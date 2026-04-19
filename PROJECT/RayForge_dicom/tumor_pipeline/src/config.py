

class Config:
    """Configuration class to store base paths"""
    DICOM_BASE_PATH = r"C:\Users\Maryam\Desktop\manifest-1759611643417\Lung-PET-CT-Dx"
    ANNOTATION_BASE_PATH = r"C:\Users\Maryam\Videos\FYP_MARYAM\Annotation subject wise directory"
    MASTER_CSV_PATH = r"C:\Users\Maryam\Videos\FYP_MARYAM\subject metadata.csv"

    @classmethod
    def set_paths(cls, dicom_base, annotation_base, master_csv):
        cls.DICOM_BASE_PATH = dicom_base
        cls.ANNOTATION_BASE_PATH = annotation_base
        cls.MASTER_CSV_PATH = master_csv