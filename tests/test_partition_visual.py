"""
Visual debugging tests for mesh partitioning.
Creates test meshes and saves them for inspection.
"""

import sys
from pathlib import Path

import numpy as np
import trimesh
from trimesh import graph

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from partition_cell import partition_by_thickness


def create_neuron_like_mesh():
    """
    Create a more realistic neuron-like test mesh.

    Structure:
    - Large sphere (soma)
    - Medium cylinder (dendrite)
    - Small spheres on stalks (spines)
    """
    soma = trimesh.creation.icosphere(subdivisions=3, radius=3.0)

    dendrite = trimesh.creation.cylinder(radius=0.8, height=10.0, sections=16)
    rotation = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
    dendrite.apply_transform(rotation)
    dendrite.apply_translation([5.0, 0, 0])

    meshes = [soma, dendrite]

    spine_configs = [
        {"x": 6.0, "angle": np.pi / 4},
        {"x": 7.5, "angle": -np.pi / 3},
        {"x": 9.0, "angle": np.pi / 2, "tilt": np.pi / 6},
    ]

    for config in spine_configs:
        x_pos = config["x"]
        angle = config["angle"]
        tilt = config.get("tilt", 0)

        radial_dir = np.array([0, np.cos(angle), np.sin(angle)])

        if abs(tilt) > 1e-6:
            tilt_matrix = trimesh.transformations.rotation_matrix(tilt, [1, 0, 0])
            radial_dir = tilt_matrix[:3, :3] @ radial_dir

        radial_dir = radial_dir / np.linalg.norm(radial_dir)

        dendrite_surface = np.array([x_pos, 0, 0]) + radial_dir * 0.8

        stalk = trimesh.creation.cylinder(radius=0.2, height=1.0, sections=8)

        z_axis = np.array([0, 0, 1])
        rotation_axis = np.cross(z_axis, radial_dir)
        if np.linalg.norm(rotation_axis) > 1e-6:
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            rotation_angle = np.arccos(np.clip(np.dot(z_axis, radial_dir), -1.0, 1.0))
            rotation_matrix = trimesh.transformations.rotation_matrix(
                rotation_angle, rotation_axis
            )
            stalk.apply_transform(rotation_matrix)

        stalk_translation = dendrite_surface + radial_dir * 0.5
        stalk.apply_translation(stalk_translation)

        head = trimesh.creation.icosphere(subdivisions=1, radius=0.4)
        head_translation = dendrite_surface + radial_dir * 1.0
        head.apply_translation(head_translation)

        meshes.extend([stalk, head])

    combined = trimesh.util.concatenate(meshes)

    return combined


def analyze_mesh_radius(mesh: trimesh.Trimesh):
    """
    Analyze and print radius statistics for a mesh.
    """
    print("\n" + "=" * 60)
    print("MESH ANALYSIS")
    print("=" * 60)

    print(f"\nMesh statistics:")
    print(f"  Vertices: {len(mesh.vertices)}")
    print(f"  Faces: {len(mesh.faces)}")
    print(f"  Is watertight: {mesh.is_watertight}")
    print(f"  Bounding box: {mesh.bounds}")

    radii, span = graph.face_adjacency_radius(mesh)

    finite_radii = radii[np.isfinite(radii)]

    print(f"\nFace adjacency radius statistics:")
    print(f"  Total adjacencies: {len(radii)}")
    print(f"  Finite radii: {len(finite_radii)}")
    print(f"  Infinite radii: {np.sum(np.isinf(radii))}")

    if len(finite_radii) > 0:
        print(f"\nFinite radius distribution:")
        print(f"  Min: {np.min(finite_radii):.4f}")
        print(f"  Max: {np.max(finite_radii):.4f}")
        print(f"  Mean: {np.mean(finite_radii):.4f}")
        print(f"  Median: {np.median(finite_radii):.4f}")
        print(f"  Std: {np.std(finite_radii):.4f}")

        percentiles = [10, 25, 50, 75, 90, 95, 99]
        print(f"\nPercentiles:")
        for p in percentiles:
            val = np.percentile(finite_radii, p)
            print(f"  {p}th: {val:.4f}")

        print(f"\nRadius histogram (finite values):")
        hist, bin_edges = np.histogram(finite_radii, bins=10)
        for i, (count, edge) in enumerate(zip(hist, bin_edges[:-1])):
            next_edge = bin_edges[i + 1]
            bar = "#" * int(count / hist.max() * 40)
            print(f"  [{edge:6.2f} - {next_edge:6.2f}]: {bar} ({count})")


def test_simple_sphere_cylinder():
    """Test with simple sphere + cylinder."""
    print("\n" + "=" * 60)
    print("TEST: Simple Sphere + Cylinder")
    print("=" * 60)

    sphere = trimesh.creation.icosphere(subdivisions=3, radius=3.0)
    cylinder = trimesh.creation.cylinder(radius=0.5, height=4.0, sections=8)
    cylinder.apply_translation([3.5, 0, 0])

    mesh = trimesh.util.concatenate([sphere, cylinder])

    analyze_mesh_radius(mesh)

    output_dir = Path(__file__).parent.parent / "data" / "test_meshes"
    output_dir.mkdir(exist_ok=True, parents=True)

    mesh.export(output_dir / "test_sphere_cylinder.obj")
    print(f"\nSaved test mesh to: {output_dir / 'test_sphere_cylinder.obj'}")

    print("\n" + "-" * 60)
    print("Testing partition with different thresholds:")
    print("-" * 60)

    for threshold in [0.5, 1.0, 2.0, 3.0, 5.0]:
        print(f"\nThreshold: {threshold}")
        main_body, spines = partition_by_thickness(
            mesh, radius_threshold=threshold, min_spine_faces=5
        )
        print(f"  Main body faces: {len(main_body.faces)}")
        print(f"  Spines found: {len(spines)}")
        for i, spine in enumerate(spines):
            print(f"    Spine {i}: {len(spine.faces)} faces")


def test_neuron_like():
    """Test with neuron-like structure."""
    print("\n" + "=" * 60)
    print("TEST: Neuron-like Structure")
    print("=" * 60)

    mesh = create_neuron_like_mesh()

    analyze_mesh_radius(mesh)

    output_dir = Path(__file__).parent.parent / "data" / "test_meshes"
    output_dir.mkdir(exist_ok=True, parents=True)

    mesh.export(output_dir / "test_neuron_like.obj")
    print(f"\nSaved test mesh to: {output_dir / 'test_neuron_like.obj'}")

    print("\n" + "-" * 60)
    print("Testing partition with different thresholds:")
    print("-" * 60)

    for threshold in [0.5, 1.0, 1.5, 2.0, 3.0]:
        print(f"\nThreshold: {threshold}")
        main_body, spines = partition_by_thickness(
            mesh, radius_threshold=threshold, min_spine_faces=5
        )
        print(f"  Main body faces: {len(main_body.faces)}")
        print(f"  Spines found: {len(spines)}")
        for i, spine in enumerate(spines):
            print(f"    Spine {i}: {len(spine.faces)} faces")

        main_body.export(output_dir / f"neuron_main_t{threshold:.1f}.obj")
        for i, spine in enumerate(spines):
            spine.export(output_dir / f"neuron_spine{i}_t{threshold:.1f}.obj")


def test_connected_components_directly():
    """Test graph.split directly to understand behavior."""
    print("\n" + "=" * 60)
    print("TEST: Direct Connected Components Analysis")
    print("=" * 60)

    mesh = create_neuron_like_mesh()

    print("\nOriginal mesh:")
    print(f"  Faces: {len(mesh.faces)}")
    print(f"  Face adjacencies: {len(mesh.face_adjacency)}")

    radii, span = graph.face_adjacency_radius(mesh)

    print("\nTesting different radius thresholds:")
    for threshold in [0.5, 1.0, 2.0, 5.0]:
        thin_connections = radii < threshold
        thin_connections[~np.isfinite(radii)] = False

        filtered_adjacency = mesh.face_adjacency[thin_connections]

        print(f"\nThreshold {threshold}:")
        print(
            f"  Connections kept: {len(filtered_adjacency)} / "
            f"{len(mesh.face_adjacency)}"
        )

        components = graph.split(
            mesh, only_watertight=False, adjacency=filtered_adjacency
        )

        print(f"  Components: {len(components)}")
        for i, comp in enumerate(components):
            print(
                f"    Component {i}: {len(comp.faces)} faces, "
                f"{len(comp.vertices)} vertices"
            )


if __name__ == "__main__":
    test_simple_sphere_cylinder()
    test_neuron_like()
    test_connected_components_directly()

    print("\n" + "=" * 60)
    print("TESTS COMPLETE")
    print("=" * 60)
    print("\nCheck data/test_meshes/ for output files")
