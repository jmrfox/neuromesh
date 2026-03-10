"""
Script to subtract dendritic spine meshes from a neuron cell mesh and repair holes.
"""

import logging
from pathlib import Path

import numpy as np
import trimesh
from trimesh import grouping

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def repair_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Repair mesh by filling holes and fixing non-manifold edges.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh to repair

    Returns
    -------
    trimesh.Trimesh
        Repaired mesh
    """
    logger.info("Repairing mesh...")

    mesh.fill_holes()

    unique_faces, inverse = grouping.unique_rows(np.sort(mesh.faces, axis=1))
    if len(unique_faces) < len(mesh.faces):
        logger.info(f"Removing {len(mesh.faces) - len(unique_faces)} duplicate faces")
        mesh.update_faces(unique_faces)

    mesh.remove_unreferenced_vertices()

    if not mesh.is_watertight:
        logger.warning("Mesh is not watertight after repair")

    return mesh


def subtract_spine(
    cell_mesh: trimesh.Trimesh,
    spine_mesh: trimesh.Trimesh,
    spine_name: str,
    engine: str = "manifold",
) -> trimesh.Trimesh:
    """
    Subtract a spine mesh from the cell mesh.

    Parameters
    ----------
    cell_mesh : trimesh.Trimesh
        Main neuron cell mesh
    spine_mesh : trimesh.Trimesh
        Spine mesh to subtract
    spine_name : str
        Name of spine for logging
    engine : str
        Boolean operation engine ('manifold', 'blender', or 'scad')

    Returns
    -------
    trimesh.Trimesh
        Cell mesh with spine subtracted
    """
    logger.info(f"Subtracting {spine_name} from cell mesh...")

    if not cell_mesh.is_watertight:
        logger.warning("Cell mesh is not watertight, attempting repair...")
        cell_mesh.fill_holes()

    if not spine_mesh.is_watertight:
        logger.warning("Spine mesh is not watertight, attempting repair...")
        spine_mesh.fill_holes()

    try:
        result = trimesh.boolean.difference([cell_mesh, spine_mesh], engine=engine)
        logger.info(f"Successfully subtracted {spine_name}")
        return result
    except Exception as e:
        logger.error(f"Failed to subtract {spine_name}: {e}")
        logger.info("Returning original mesh")
        return cell_mesh


def main():
    """Main execution function."""

    data_dir = Path(__file__).parent.parent / "data"
    cell_path = data_dir / "cell_clean.obj"
    output_path = data_dir / "cell_no_spines.obj"

    logger.info(f"Loading cell mesh from {cell_path}")
    cell_mesh = trimesh.load(cell_path, force="mesh")

    logger.info(
        f"Cell mesh: {len(cell_mesh.vertices)} vertices, {len(cell_mesh.faces)} faces"
    )
    logger.info(f"Cell mesh is watertight: {cell_mesh.is_watertight}")

    spine_files = sorted(data_dir.glob("TS*_alone.obj"))
    logger.info(f"Found {len(spine_files)} spine files")

    for spine_file in spine_files:
        logger.info(f"\nProcessing {spine_file.name}")

        spine_mesh = trimesh.load(spine_file, force="mesh")
        logger.info(
            f"  Spine: {len(spine_mesh.vertices)} vertices, {len(spine_mesh.faces)} faces"
        )

        cell_mesh = subtract_spine(cell_mesh, spine_mesh, spine_file.stem)

        cell_mesh = repair_mesh(cell_mesh)

        logger.info(
            f"  Result: {len(cell_mesh.vertices)} vertices, {len(cell_mesh.faces)} faces"
        )

    logger.info(f"\nSaving result to {output_path}")
    cell_mesh.export(output_path)

    logger.info("Done!")
    logger.info(
        f"Final mesh: {len(cell_mesh.vertices)} vertices, {len(cell_mesh.faces)} faces"
    )
    logger.info(f"Final mesh is watertight: {cell_mesh.is_watertight}")


if __name__ == "__main__":
    main()
