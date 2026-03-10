"""
Test partition algorithm on real test mesh.
"""

import sys
from pathlib import Path

import numpy as np
import trimesh
from trimesh import graph

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from partition_cell import partition_by_thickness


def test_spiny_neuron_mesh():
    """Test partitioning on the provided spiny neuron mesh."""

    mesh_path = (
        Path(__file__).parent.parent / "data" / "test_meshes" / "test_spiny_neuron.obj"
    )

    if not mesh_path.exists():
        print(f"Test mesh not found at {mesh_path}")
        return

    print("\n" + "=" * 60)
    print("TESTING REAL SPINY NEURON MESH")
    print("=" * 60)

    mesh = trimesh.load(mesh_path, force="mesh")

    print(f"\nMesh loaded:")
    print(f"  Path: {mesh_path}")
    print(f"  Vertices: {len(mesh.vertices)}")
    print(f"  Faces: {len(mesh.faces)}")
    print(f"  Is watertight: {mesh.is_watertight}")
    print(f"  Bounding box: {mesh.bounds}")

    radii, span = graph.face_adjacency_radius(mesh)
    finite_radii = radii[np.isfinite(radii)]

    print(f"\nFace adjacency radius analysis:")
    print(f"  Total adjacencies: {len(radii)}")
    print(f"  Finite radii: {len(finite_radii)}")
    print(f"  Infinite radii: {np.sum(np.isinf(radii))}")

    if len(finite_radii) > 0:
        print(f"\nRadius statistics:")
        print(f"  Min: {np.min(finite_radii):.4f}")
        print(f"  Max: {np.max(finite_radii):.4f}")
        print(f"  Mean: {np.mean(finite_radii):.4f}")
        print(f"  Median: {np.median(finite_radii):.4f}")
        print(f"  Std: {np.std(finite_radii):.4f}")

        percentiles = [10, 25, 50, 75, 90, 95, 99]
        print(f"\nPercentiles:")
        for p in percentiles:
            val = np.percentile(finite_radii, p)
            print(f"  {p:2d}th: {val:.4f}")

        print(f"\nRadius histogram:")
        hist, bin_edges = np.histogram(finite_radii, bins=10)
        for i, (count, edge) in enumerate(zip(hist, bin_edges[:-1])):
            next_edge = bin_edges[i + 1]
            bar = "#" * int(count / hist.max() * 40)
            print(f"  [{edge:6.2f} - {next_edge:6.2f}]: {bar} ({count})")

    print("\n" + "-" * 60)
    print("Testing different radius thresholds:")
    print("-" * 60)

    output_dir = Path(__file__).parent.parent / "data" / "test_meshes" / "partitioned"
    output_dir.mkdir(exist_ok=True, parents=True)

    if len(finite_radii) > 0:
        suggested_thresholds = [
            np.percentile(finite_radii, 10),
            np.percentile(finite_radii, 25),
            np.percentile(finite_radii, 50),
            np.median(finite_radii),
        ]
    else:
        suggested_thresholds = [0.5, 1.0, 2.0]

    for threshold in suggested_thresholds:
        print(f"\n{'='*60}")
        print(f"Threshold: {threshold:.4f}")
        print("=" * 60)

        main_body, spines = partition_by_thickness(
            mesh, radius_threshold=threshold, min_spine_faces=10
        )

        print(f"\nResults:")
        print(
            f"  Main body: {len(main_body.faces)} faces, "
            f"{len(main_body.vertices)} vertices"
        )
        print(f"  Spines found: {len(spines)}")

        for i, spine in enumerate(spines):
            print(
                f"    Spine {i}: {len(spine.faces)} faces, "
                f"{len(spine.vertices)} vertices"
            )

        threshold_str = f"{threshold:.4f}".replace(".", "_")
        main_body.export(output_dir / f"main_t{threshold_str}.obj")
        for i, spine in enumerate(spines):
            spine.export(output_dir / f"spine{i}_t{threshold_str}.obj")

        print(f"\nSaved to: {output_dir}")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_spiny_neuron_mesh()
