"""
Tests for mesh partitioning functionality.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from partition_cell import partition_by_thickness


def create_sphere_with_cylinders(
    sphere_radius: float = 5.0,
    cylinder_radius: float = 0.5,
    cylinder_height: float = 3.0,
    num_cylinders: int = 3,
) -> trimesh.Trimesh:
    """
    Create a test mesh: sphere (thick body) with thin cylinders (spines).

    This simulates a simplified neuron with soma (sphere) and spines (cylinders).
    """
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=sphere_radius)

    meshes = [sphere]

    angles = np.linspace(0, 2 * np.pi, num_cylinders, endpoint=False)

    for i, angle in enumerate(angles):
        cylinder = trimesh.creation.cylinder(
            radius=cylinder_radius, height=cylinder_height, sections=8
        )

        radial_direction = np.array([np.cos(angle), np.sin(angle), 0])

        z_axis = np.array([0, 0, 1])
        rotation_axis = np.cross(z_axis, radial_direction)
        if np.linalg.norm(rotation_axis) > 1e-6:
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            rotation_angle = np.arccos(np.dot(z_axis, radial_direction))
            rotation_matrix = trimesh.transformations.rotation_matrix(
                rotation_angle, rotation_axis
            )
            cylinder.apply_transform(rotation_matrix)

        contact_point = sphere_radius * radial_direction
        translation = contact_point + radial_direction * (cylinder_height / 2)
        cylinder.apply_translation(translation)

        meshes.append(cylinder)

    combined = trimesh.util.concatenate(meshes)

    return combined


def create_two_connected_spheres(
    large_radius: float = 5.0, small_radius: float = 1.0, separation: float = 5.0
) -> trimesh.Trimesh:
    """
    Create two spheres connected at a point (thick + thin structure).
    """
    large_sphere = trimesh.creation.icosphere(subdivisions=3, radius=large_radius)

    small_sphere = trimesh.creation.icosphere(subdivisions=2, radius=small_radius)
    small_sphere.apply_translation([separation, 0, 0])

    combined = trimesh.util.concatenate([large_sphere, small_sphere])

    return combined


def test_partition_basic():
    """Test basic partitioning with sphere and cylinders."""
    mesh = create_sphere_with_cylinders(
        sphere_radius=5.0, cylinder_radius=0.5, cylinder_height=3.0, num_cylinders=3
    )

    assert mesh.is_volume, "Test mesh should be a volume"

    main_body, spines = partition_by_thickness(
        mesh, radius_threshold=2.0, min_spine_faces=10
    )

    assert main_body is not None, "Should return main body"
    assert isinstance(spines, list), "Should return list of spines"
    assert len(main_body.faces) > 0, "Main body should have faces"

    print(f"\nTest results:")
    print(f"  Original mesh: {len(mesh.faces)} faces")
    print(f"  Main body: {len(main_body.faces)} faces")
    print(f"  Spines found: {len(spines)}")
    for i, spine in enumerate(spines):
        print(f"    Spine {i}: {len(spine.faces)} faces")


def test_partition_no_spines():
    """Test partitioning with only a sphere (no thin structures)."""
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=5.0)

    main_body, spines = partition_by_thickness(
        mesh, radius_threshold=2.0, min_spine_faces=10
    )

    assert main_body is not None, "Should return main body"
    assert len(spines) == 0, "Should find no spines in uniform sphere"
    assert len(main_body.faces) == len(mesh.faces), "All faces should be in main body"


def test_partition_two_spheres():
    """Test partitioning with two connected spheres of different sizes."""
    mesh = create_two_connected_spheres(
        large_radius=5.0, small_radius=1.0, separation=5.5
    )

    main_body, spines = partition_by_thickness(
        mesh, radius_threshold=3.0, min_spine_faces=10
    )

    assert main_body is not None, "Should return main body"
    print(f"\nTwo spheres test:")
    print(f"  Original mesh: {len(mesh.faces)} faces")
    print(f"  Main body: {len(main_body.faces)} faces")
    print(f"  Spines found: {len(spines)}")


def test_partition_threshold_sensitivity():
    """Test that different thresholds produce different results."""
    mesh = create_sphere_with_cylinders(
        sphere_radius=5.0, cylinder_radius=0.5, cylinder_height=3.0, num_cylinders=3
    )

    main_body_low, spines_low = partition_by_thickness(
        mesh, radius_threshold=1.0, min_spine_faces=10
    )

    main_body_high, spines_high = partition_by_thickness(
        mesh, radius_threshold=5.0, min_spine_faces=10
    )

    print(f"\nThreshold sensitivity test:")
    print(f"  Low threshold (1.0): {len(spines_low)} spines")
    print(f"  High threshold (5.0): {len(spines_high)} spines")

    assert len(spines_low) >= len(
        spines_high
    ), "Lower threshold should find same or more spines"


def test_partition_min_faces_filter():
    """Test that min_spine_faces parameter filters small components."""
    mesh = create_sphere_with_cylinders(
        sphere_radius=5.0, cylinder_radius=0.5, cylinder_height=3.0, num_cylinders=3
    )

    main_body_strict, spines_strict = partition_by_thickness(
        mesh, radius_threshold=2.0, min_spine_faces=100
    )

    main_body_loose, spines_loose = partition_by_thickness(
        mesh, radius_threshold=2.0, min_spine_faces=10
    )

    print(f"\nMin faces filter test:")
    print(f"  Strict filter (100): {len(spines_strict)} spines")
    print(f"  Loose filter (10): {len(spines_loose)} spines")

    assert len(spines_strict) <= len(
        spines_loose
    ), "Stricter filter should find fewer or equal spines"


def test_mesh_validity():
    """Test that output meshes are valid."""
    mesh = create_sphere_with_cylinders(
        sphere_radius=5.0, cylinder_radius=0.5, cylinder_height=3.0, num_cylinders=3
    )

    main_body, spines = partition_by_thickness(
        mesh, radius_threshold=2.0, min_spine_faces=10
    )

    assert len(main_body.vertices) > 0, "Main body should have vertices"
    assert len(main_body.faces) > 0, "Main body should have faces"

    for i, spine in enumerate(spines):
        assert len(spine.vertices) > 0, f"Spine {i} should have vertices"
        assert len(spine.faces) > 0, f"Spine {i} should have faces"
        assert spine.faces.max() < len(
            spine.vertices
        ), f"Spine {i} face indices should be valid"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
