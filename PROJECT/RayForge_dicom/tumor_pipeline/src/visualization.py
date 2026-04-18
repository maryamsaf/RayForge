"""Visualization functions"""
import numpy as np
import pandas as pd
import SimpleITK as sitk
import vtk
from vtkmodules.util import numpy_support


def sitk_to_vtk(image):
    """Convert SimpleITK image to VTK image"""
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    vtk_arr = numpy_support.numpy_to_vtk(
        arr.ravel(order="C"), deep=True, array_type=vtk.VTK_FLOAT
    )

    vtk_img = vtk.vtkImageData()
    vtk_img.SetDimensions(image.GetSize())
    vtk_img.SetSpacing(image.GetSpacing())
    vtk_img.SetOrigin(image.GetOrigin())
    vtk_img.GetPointData().SetScalars(vtk_arr)
    return vtk_img


def load_bounding_boxes(csv_path):
    """Load bounding box data from CSV/TSV file."""
    try:
        df = pd.read_csv(csv_path, sep='\t')
        if len(df.columns) == 1:
            df = pd.read_csv(csv_path)

        print(f"[BBOX] Loaded {len(df)} slice annotations")
        print(f"[BBOX] Columns: {df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"[BBOX] Error loading CSV: {e}")
        return None


def compute_3d_bounding_box(bbox_df, instance_map):
    """Compute a single 3D bounding box from multiple 2D slice annotations."""
    if bbox_df is None or len(bbox_df) == 0:
        return None

    x_min_global = bbox_df['x_min'].min()
    x_max_global = bbox_df['x_max'].max()
    y_min_global = bbox_df['y_min'].min()
    y_max_global = bbox_df['y_max'].max()

    inst_nums = bbox_df['inst'].values
    z_indices = []
    for inst in inst_nums:
        if instance_map and inst in instance_map:
            z_indices.append(instance_map[inst])
        else:
            z_indices.append(inst)

    z_min_global = min(z_indices)
    z_max_global = max(z_indices)

    print(f"\n[3D BBOX] Computed from {len(bbox_df)} slices:")
    print(f"  X: [{x_min_global}, {x_max_global}]")
    print(f"  Y: [{y_min_global}, {y_max_global}]")
    print(f"  Z: [{z_min_global}, {z_max_global}] (slices {inst_nums.min()} to {inst_nums.max()})")

    return {
        'x_min': x_min_global, 'x_max': x_max_global,
        'y_min': y_min_global, 'y_max': y_max_global,
        'z_min': z_min_global, 'z_max': z_max_global
    }


def create_bounding_box_actor(x_min, y_min, z_min, x_max, y_max, z_max,
                              spacing, origin, color=(1, 0, 0), label=""):
    """Create a THICK wireframe bounding box actor in physical coordinates."""
    x_min_phys = origin[0] + x_min * spacing[0]
    x_max_phys = origin[0] + x_max * spacing[0]
    y_min_phys = origin[1] + y_min * spacing[1]
    y_max_phys = origin[1] + y_max * spacing[1]
    z_min_phys = origin[2] + z_min * spacing[2]
    z_max_phys = origin[2] + z_max * spacing[2]

    print(f"[BBOX {label}] Voxel: x=[{x_min},{x_max}], y=[{y_min},{y_max}], z=[{z_min},{z_max}]")
    print(f"[BBOX {label}] Physical: x=[{x_min_phys:.1f},{x_max_phys:.1f}], "
          f"y=[{y_min_phys:.1f},{y_max_phys:.1f}], z=[{z_min_phys:.1f},{z_max_phys:.1f}]")

    points = vtk.vtkPoints()
    points.InsertNextPoint(x_min_phys, y_min_phys, z_min_phys)
    points.InsertNextPoint(x_max_phys, y_min_phys, z_min_phys)
    points.InsertNextPoint(x_max_phys, y_max_phys, z_min_phys)
    points.InsertNextPoint(x_min_phys, y_max_phys, z_min_phys)
    points.InsertNextPoint(x_min_phys, y_min_phys, z_max_phys)
    points.InsertNextPoint(x_max_phys, y_min_phys, z_max_phys)
    points.InsertNextPoint(x_max_phys, y_max_phys, z_max_phys)
    points.InsertNextPoint(x_min_phys, y_max_phys, z_max_phys)

    lines = vtk.vtkCellArray()
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]

    for edge in edges:
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, edge[0])
        line.GetPointIds().SetId(1, edge[1])
        lines.InsertNextCell(line)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(lines)

    tube = vtk.vtkTubeFilter()
    tube.SetInputData(polydata)
    tube.SetRadius(2.0)
    tube.SetNumberOfSides(8)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(tube.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)
    actor.GetProperty().SetOpacity(1.0)

    return actor


def build_renderer(vtk_inner, vtk_shell, bbox_df=None, instance_map=None, original_spacing=None):
    """Build VTK renderer with volume rendering and optional bounding boxes"""
    mapper_inner = vtk.vtkGPUVolumeRayCastMapper()
    mapper_inner.SetInputData(vtk_inner)
    mapper_inner.SetSampleDistance(0.15 * min(vtk_inner.GetSpacing()))

    ctf_inner = vtk.vtkColorTransferFunction()
    ctf_inner.AddRGBPoint(0, 0.0, 0.0, 0.0)
    ctf_inner.AddRGBPoint(20, 0.7, 0.6, 0.4)
    ctf_inner.AddRGBPoint(60, 0.9, 0.8, 0.5)
    ctf_inner.AddRGBPoint(140, 1.0, 0.25, 0.25)

    otf_inner = vtk.vtkPiecewiseFunction()
    otf_inner.AddPoint(0, 0.0)
    otf_inner.AddPoint(30, 0.0008)
    otf_inner.AddPoint(60, 0.004)
    otf_inner.AddPoint(100, 0.015)
    otf_inner.AddPoint(180, 0.10)

    gotf_inner = vtk.vtkPiecewiseFunction()
    gotf_inner.AddPoint(0, 0.0)
    gotf_inner.AddPoint(2, 0.7)
    gotf_inner.AddPoint(6, 1.0)

    prop_inner = vtk.vtkVolumeProperty()
    prop_inner.SetColor(ctf_inner)
    prop_inner.SetScalarOpacity(otf_inner)
    prop_inner.SetGradientOpacity(gotf_inner)
    prop_inner.ShadeOn()
    prop_inner.SetAmbient(0.15)
    prop_inner.SetDiffuse(0.85)
    prop_inner.SetSpecular(0.05)

    vol_inner = vtk.vtkVolume()
    vol_inner.SetMapper(mapper_inner)
    vol_inner.SetProperty(prop_inner)

    mapper_shell = vtk.vtkGPUVolumeRayCastMapper()
    mapper_shell.SetInputData(vtk_shell)
    mapper_shell.SetSampleDistance(0.45 * min(vtk_shell.GetSpacing()))

    ctf_shell = vtk.vtkColorTransferFunction()
    ctf_shell.AddRGBPoint(0, 0.0, 0.0, 0.0)
    ctf_shell.AddRGBPoint(255, 1.0, 0.95, 0.85)

    otf_shell = vtk.vtkPiecewiseFunction()
    otf_shell.AddPoint(0, 0.0)
    otf_shell.AddPoint(20, 0.0)
    otf_shell.AddPoint(60, 0.015)
    otf_shell.AddPoint(255, 0.03)

    gotf_shell = vtk.vtkPiecewiseFunction()
    gotf_shell.AddPoint(0, 0.0)
    gotf_shell.AddPoint(3, 0.7)
    gotf_shell.AddPoint(8, 1.0)

    prop_shell = vtk.vtkVolumeProperty()
    prop_shell.SetColor(ctf_shell)
    prop_shell.SetScalarOpacity(otf_shell)
    prop_shell.SetGradientOpacity(gotf_shell)
    prop_shell.ShadeOff()

    vol_shell = vtk.vtkVolume()
    vol_shell.SetMapper(mapper_shell)
    vol_shell.SetProperty(prop_shell)

    ren = vtk.vtkRenderer()
    ren.AddVolume(vol_shell)
    ren.AddVolume(vol_inner)

    if bbox_df is not None:
        spacing = vtk_inner.GetSpacing()
        origin = vtk_inner.GetOrigin()

        if original_spacing is not None:
            scale_xy = original_spacing[0] / spacing[0]
            scale_z = original_spacing[2] / spacing[2]
        else:
            scale_xy = 1.0
            scale_z = 1.0

        bbox_3d = compute_3d_bounding_box(bbox_df, instance_map)

        if bbox_3d:
            x_min = int(bbox_3d['x_min'] * scale_xy)
            y_min = int(bbox_3d['y_min'] * scale_xy)
            x_max = int(bbox_3d['x_max'] * scale_xy)
            y_max = int(bbox_3d['y_max'] * scale_xy)
            z_min = int(bbox_3d['z_min'] * scale_z)
            z_max = int(bbox_3d['z_max'] * scale_z)

            bbox_actor = create_bounding_box_actor(
                x_min, y_min, z_min, x_max, y_max, z_max,
                spacing, origin, color=(1.0, 0.0, 0.0), label="TUMOR"
            )
            ren.AddActor(bbox_actor)

    ren.SetBackground(0, 0, 0)

    win = vtk.vtkRenderWindow()
    win.AddRenderer(ren)
    win.SetSize(1100, 1100)

    iren = vtk.vtkRenderWindowInteractor()
    style = vtk.vtkInteractorStyleTrackballCamera()
    style.SetMotionFactor(2.3)
    iren.SetInteractorStyle(style)
    iren.SetRenderWindow(win)

    ren.ResetCamera()
    return win, iren