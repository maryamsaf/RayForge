#!/usr/bin/env python3
"""
Headless lung surface export for the RayForge web pipeline.

Reads a DICOM series folder, runs the same lung segmentation as main.py,
then writes a decimated triangle mesh as Wavefront OBJ (no VTK window).
"""
from __future__ import annotations

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import numpy as np
import SimpleITK as sitk
import vtk
from vtkmodules.util import numpy_support

from data_loader import load_dicom_itk, resample_isotropic
from segmentation import build_lung_mask_seeded_itk, build_lung_mask_top2_itk


def find_dicom_series_dir(root: str) -> str:
    """Pick a directory under ``root`` that GDCM recognizes as a DICOM series."""
    reader = sitk.ImageSeriesReader()
    if reader.GetGDCMSeriesIDs(root):
        return root

    best_dir = root
    best_dcm = 0
    for dirpath, _, filenames in os.walk(root):
        n = sum(1 for f in filenames if f.lower().endswith(".dcm"))
        if n == 0:
            continue
        if reader.GetGDCMSeriesIDs(dirpath) and n > best_dcm:
            best_dir = dirpath
            best_dcm = n
    return best_dir


def sitk_mask_to_vtk(mask_img: sitk.Image) -> vtk.vtkImageData:
    """Binary / label mask as vtkImageData (matches tumor_pipeline visualization layout)."""
    arr = sitk.GetArrayFromImage(mask_img).astype(np.float32)
    vtk_arr = numpy_support.numpy_to_vtk(
        arr.ravel(order="C"), deep=True, array_type=vtk.VTK_FLOAT
    )
    vtk_img = vtk.vtkImageData()
    vtk_img.SetDimensions(mask_img.GetSize())
    vtk_img.SetSpacing(mask_img.GetSpacing())
    vtk_img.SetOrigin(mask_img.GetOrigin())
    vtk_img.GetPointData().SetScalars(vtk_arr)
    return vtk_img


def vtk_polydata_to_obj(poly: vtk.vtkPolyData, filepath: str) -> None:
    """Minimal OBJ writer (faces are triangles, 1-based indices)."""
    poly.BuildCells()
    pts = poly.GetPoints()
    if pts is None or pts.GetNumberOfPoints() == 0:
        raise RuntimeError("Mesh has no points")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("# RayForge lung mesh\n")
        for i in range(pts.GetNumberOfPoints()):
            x, y, z = pts.GetPoint(i)
            f.write(f"v {x} {y} {z}\n")

        polys = poly.GetPolys()
        polys.InitTraversal()
        id_list = vtk.vtkIdList()
        while polys.GetNextCell(id_list):
            if id_list.GetNumberOfIds() != 3:
                continue
            a = id_list.GetId(0) + 1
            b = id_list.GetId(1) + 1
            c = id_list.GetId(2) + 1
            f.write(f"f {a} {b} {c}\n")


def lung_mask_to_obj_mesh(
    mask_img: sitk.Image,
    output_path: str,
    iso: float = 0.5,
    target_reduction: float = 0.93,
) -> None:
    vtk_img = sitk_mask_to_vtk(mask_img)

    if hasattr(vtk, "vtkFlyingEdges3D"):
        surf = vtk.vtkFlyingEdges3D()
    else:
        surf = vtk.vtkMarchingCubes()
    surf.SetInputData(vtk_img)
    surf.SetValue(0, float(iso))
    surf.ComputeNormalsOn()

    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(surf.GetOutputPort())

    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(tri.GetOutputPort())
    clean.SetTolerance(0.001)

    dec = vtk.vtkDecimatePro()
    dec.SetInputConnection(clean.GetOutputPort())
    dec.SetTargetReduction(float(target_reduction))
    dec.PreserveTopologyOn()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(dec.GetOutputPort())
    normals.SplittingOff()
    normals.ConsistencyOn()
    normals.Update()

    out = normals.GetOutput()
    vtk_polydata_to_obj(out, output_path)


def run_pipeline(
    dicom_dir: str,
    output_obj: str,
    iso_spacing: float = 1.0,
    target_reduction: float = 0.93,
) -> None:
    dicom_dir = find_dicom_series_dir(os.path.abspath(dicom_dir))
    print(f"[export_lung_mesh] Using DICOM directory: {dicom_dir}")

    img, _files = load_dicom_itk(dicom_dir)
    img_iso = resample_isotropic(img, iso_spacing)

    print("[export_lung_mesh] Lung segmentation…")
    lung_mask = build_lung_mask_seeded_itk(img_iso)
    if lung_mask is None:
        lung_mask = build_lung_mask_top2_itk(img_iso)
    if lung_mask is None:
        raise RuntimeError("Lung segmentation failed")
    lung_mask = sitk.Cast(lung_mask > 0, sitk.sitkUInt8)

    out_abs = os.path.abspath(output_obj)
    parent = os.path.dirname(out_abs)
    if parent:
        os.makedirs(parent, exist_ok=True)
    print(f"[export_lung_mesh] Writing mesh: {output_obj}")
    lung_mask_to_obj_mesh(lung_mask, output_obj, target_reduction=target_reduction)
    print("[export_lung_mesh] Done.")


def main() -> int:
    p = argparse.ArgumentParser(description="Export lung surface mesh from DICOM folder")
    p.add_argument("--dicom-dir", required=True, help="Folder containing DICOM series")
    p.add_argument("--output", required=True, help="Output .obj path")
    p.add_argument(
        "--iso-spacing",
        type=float,
        default=1.0,
        help="Isotropic resampling spacing in mm (smaller = more detail, slower)",
    )
    p.add_argument(
        "--target-reduction",
        type=float,
        default=0.93,
        help="VTK decimation 0–0.95; higher = fewer triangles (faster in browser)",
    )
    args = p.parse_args()

    try:
        run_pipeline(
            args.dicom_dir,
            args.output,
            iso_spacing=args.iso_spacing,
            target_reduction=args.target_reduction,
        )
    except Exception as e:
        print(f"[export_lung_mesh] ERROR: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
