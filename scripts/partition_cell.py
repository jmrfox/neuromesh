"""
Partition a neuron mesh into soma+dendrites and individual spines
based on local thickness/radius.
"""

import logging
from pathlib import Path

import numpy as np
import trimesh
from trimesh import graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def partition_by_thickness(
    mesh: trimesh.Trimesh, radius_threshold: float = 2.0, min_spine_faces: int = 50
) -> tuple[trimesh.Trimesh, list[trimesh.Trimesh]]:
    """
    Partition mesh into main body and spines based on local thickness.

    Strategy: Compute local radius at face adjacencies. Remove connections
    where radius is below threshold (thin necks). This isolates thin
    structures (spines) from thick structures (soma+dendrites).

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input neuron mesh
    radius_threshold : float
        Radius threshold - connections below this are broken (thin necks)
    min_spine_faces : int
        Minimum number of faces for a component to be considered a spine

    Returns
    -------
    main_body : trimesh.Trimesh
        Soma and dendrites (largest component)
    spines : list of trimesh.Trimesh
        Individual spine meshes
    """
    logger.info("Computing face adjacency radius...")
    radii, span = graph.face_adjacency_radius(mesh)

    logger.info(
        f"Radius stats: min={np.min(radii):.3f}, "
        f"max={np.max(radii):.3f}, "
        f"median={np.median(radii):.3f}"
    )

    valid_radii = radii[np.isfinite(radii)]
    if len(valid_radii) > 0:
        logger.info(
            f"Valid radius stats: min={np.min(valid_radii):.3f}, "
            f"max={np.max(valid_radii):.3f}, "
            f"median={np.median(valid_radii):.3f}"
        )

    thick_connections = radii > radius_threshold
    thick_connections[~np.isfinite(radii)] = True

    logger.info(
        f"Keeping {np.sum(thick_connections)} / "
        f"{len(thick_connections)} connections "
        f"(threshold={radius_threshold})"
    )

    filtered_adjacency = mesh.face_adjacency[thick_connections]

    logger.info("Splitting mesh into connected components...")
    components = graph.split(mesh, only_watertight=False, adjacency=filtered_adjacency)

    logger.info(f"Found {len(components)} components")

    component_sizes = [len(c.faces) for c in components]
    for i, size in enumerate(component_sizes):
        logger.info(f"  Component {i}: {size} faces")

    largest_idx = np.argmax(component_sizes)
    main_body = components[largest_idx]

    spines = []
    for i, component in enumerate(components):
        if i == largest_idx:
            continue
        if len(component.faces) >= min_spine_faces:
            spines.append(component)
            logger.info(f"Spine {len(spines)}: {len(component.faces)} faces")
        else:
            logger.info(
                f"Skipping small component {i}: " f"{len(component.faces)} faces"
            )

    logger.info(f"\nPartitioning complete:")
    logger.info(f"  Main body: {len(main_body.faces)} faces")
    logger.info(f"  Spines: {len(spines)} components")

    return main_body, spines


def main():
    """Main execution function."""

    data_dir = Path(__file__).parent.parent / "data"
    cell_path = data_dir / "cell_clean.obj"
    output_dir = data_dir / "partitioned"
    output_dir.mkdir(exist_ok=True)

    logger.info(f"Loading cell mesh from {cell_path}")
    cell_mesh = trimesh.load(cell_path, force="mesh")

    logger.info(
        f"Cell mesh: {len(cell_mesh.vertices)} vertices, "
        f"{len(cell_mesh.faces)} faces"
    )

    main_body, spines = partition_by_thickness(
        cell_mesh, radius_threshold=2.0, min_spine_faces=50
    )

    main_body_path = output_dir / "soma_dendrites.obj"
    logger.info(f"\nSaving main body to {main_body_path}")
    main_body.export(main_body_path)

    for i, spine in enumerate(spines):
        spine_path = output_dir / f"spine_{i:03d}.obj"
        logger.info(f"Saving spine {i} to {spine_path}")
        spine.export(spine_path)

    logger.info("\nDone!")
    logger.info(f"Saved {len(spines)} spine files and 1 main body file")


if __name__ == "__main__":
    main()
