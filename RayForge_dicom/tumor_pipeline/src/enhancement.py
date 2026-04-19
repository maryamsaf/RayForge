"""Enhancement and volume building functions"""
import SimpleITK as sitk


def compute_vessel_blob(hu, lung_mask):
    """Compute vessel and blob features using multi-scale filtering"""
    hu = sitk.Cast(hu, sitk.sitkFloat32)
    lung_mask = sitk.Cast(lung_mask, sitk.sitkFloat32)
    hu = hu * lung_mask + (-1024.0) * (1.0 - lung_mask)

    hu = sitk.CurvatureAnisotropicDiffusion(
        hu, timeStep=0.04, conductanceParameter=2.5, numberOfIterations=8
    )

    vessel_acc = sitk.Image(hu.GetSize(), sitk.sitkFloat32)
    blob_acc = sitk.Image(hu.GetSize(), sitk.sitkFloat32)
    vessel_acc.CopyInformation(hu)
    blob_acc.CopyInformation(hu)

    sigmas = [0.8, 1.2, 1.8, 2.5]

    for s in sigmas:
        grad = sitk.GradientMagnitudeRecursiveGaussian(hu, s)
        grad = sitk.RescaleIntensity(grad, 0.0, 1.0)

        log = sitk.LaplacianRecursiveGaussian(hu, s)
        log = sitk.Abs(log)
        log = sitk.RescaleIntensity(log, 0.0, 1.0)

        vessel_acc = sitk.Maximum(vessel_acc, grad)
        blob_acc = sitk.Maximum(blob_acc, log)

    return vessel_acc, blob_acc


def compute_tumor_probability(hu, vessel, blob, lung_mask):
    """Compute tumor probability map"""
    hu = sitk.Cast(hu, sitk.sitkFloat32)
    vessel = sitk.Cast(vessel, sitk.sitkFloat32)
    blob = sitk.Cast(blob, sitk.sitkFloat32)

    hu_n = sitk.RescaleIntensity(hu, 0.0, 1.0)
    not_vessel = sitk.InvertIntensity(vessel, 1.0)

    tumor = (blob * 0.55 + hu_n * 0.30 + not_vessel * 0.15)
    tumor = sitk.Mask(tumor, lung_mask, outsideValue=0.0)
    tumor = sitk.RescaleIntensity(tumor, 0.0, 1.0)
    return tumor


def build_shell_volume(lung_mask):
    """Build shell volume for visualization"""
    lung_mask_u8 = sitk.Cast(lung_mask > 0, sitk.sitkUInt8)

    dist = sitk.SignedMaurerDistanceMap(
        lung_mask_u8, insideIsPositive=False, squaredDistance=False, useImageSpacing=True
    )

    dist = sitk.Cast(dist, sitk.sitkFloat32)
    dist = sitk.SmoothingRecursiveGaussian(dist, sigma=1.6)

    shell = sitk.Clamp(dist, lowerBound=-4.0, upperBound=0.0)
    shell = sitk.RescaleIntensity(shell, 0.0, 255.0)
    shell = sitk.Cast(shell, sitk.sitkFloat32)

    return shell


def build_inner_volume(hu, vessel, tumor, lung_mask):
    """Build inner volume for visualization

    Note: Tumor dilation should be done in preprocessing (see segmentation.py)
    before calling this function for best results.

    Args:
        hu: HU intensity image (should be pre-dilated if using bbox dilation)
        vessel: Vessel probability map
        tumor: Tumor probability map
        lung_mask: Lung segmentation mask
    """
    hu = sitk.Cast(hu, sitk.sitkFloat32)
    vessel = sitk.Cast(vessel, sitk.sitkFloat32)
    tumor = sitk.Cast(tumor, sitk.sitkFloat32)
    lung_mask = sitk.Cast(lung_mask, sitk.sitkFloat32)

    hu_lung = hu * lung_mask
    hu_r = sitk.RescaleIntensity(hu_lung, 0.0, 40.0)

    inner = (hu_r * 0.08 + vessel * 90.0 + tumor * 260.0)
    inner = inner * lung_mask
    return inner