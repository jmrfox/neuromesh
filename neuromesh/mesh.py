"""
Main mesh class
"""

import logging
import multiprocessing
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import trimesh

# Module-level logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def example_mesh(
    kind: str = "cylinder",
    *,
    # Cylinder params
    radius: float = 1,
    height: float = 10,
    sections: int | None = 16,
    # Torus params
    major_radius: float = 4,
    minor_radius: float = 1,
    major_sections: int | None = 32,
    minor_sections: int | None = 16,
    **kwargs,
) -> trimesh.Trimesh:
    """Create a simple demo mesh using trimesh primitives.

    Parameters
    ----------
    kind : {"cylinder", "torus"}
        Type of primitive to generate. Default "cylinder".
    radius : float
        Cylinder radius (when kind="cylinder"). Default 1.
    height : float
        Cylinder height (when kind="cylinder"). Default 10.
    sections : int or None
        Cylinder radial resolution (pie wedges). Default 16.
    major_radius : float
        Torus major radius (center of hole to centerline of tube). Default 4.
    minor_radius : float
        Torus minor radius (tube radius). Default 1.
    major_sections : int or None
        Torus resolution around major circle. Default 32.
    minor_sections : int or None
        Torus resolution around tube section. Default 16.
    **kwargs : dict
        Passed through to Trimesh constructor via trimesh.creation.* helpers
        (e.g., process=False).

    Returns
    -------
    trimesh.Trimesh
        Generated primitive mesh.

    Examples
    --------
    >>> m = example_mesh("cylinder", radius=0.4, height=1.5)
    >>> t = example_mesh("torus", major_radius=1.0, minor_radius=0.25)
    """
    k = (kind or "cylinder").lower()
    if k == "cylinder":
        return trimesh.creation.cylinder(
            radius=float(radius),
            height=float(height),
            sections=None if sections is None else int(sections),
            **kwargs,
        )
    elif k == "torus":
        # trimesh.creation.torus parameters
        return trimesh.creation.torus(
            major_radius=float(major_radius),
            minor_radius=float(minor_radius),
            major_sections=None if major_sections is None else int(major_sections),
            minor_sections=None if minor_sections is None else int(minor_sections),
            **kwargs,
        )
    else:
        raise ValueError("example_mesh kind must be 'cylinder' or 'torus'")


class MeshManager:
    """
    Unified mesh class handling loading, processing, and analysis.
    """

    def __init__(
        self,
        mesh: Optional[trimesh.Trimesh] = None,
        mesh_path: Optional[str] = None,
        verbose: bool = True,
    ):
        # Core mesh attributes
        self.mesh = mesh
        self.mesh_path = mesh_path

        # Attributes
        self.verbose = verbose
        self.stats = {
            "processed": 0,
            "successful": 0,
            "failed": 0,
            "volume_fixed": 0,
            "watertight_fixed": 0,
            "degenerate_removed": 0,
        }

        if mesh_path is not None:
            self.load_mesh(mesh_path)

    # =================================================================
    # MESH LOADING AND BASIC OPERATIONS
    # =================================================================

    def load_mesh(
        self, filepath: str, file_format: Optional[str] = None
    ) -> trimesh.Trimesh:
        """
        Load a mesh from file.

        Args:
            filepath: Path to mesh file
            file_format: Optional format specification (auto-detected if None)

        Returns:
            Loaded trimesh object
        """
        try:
            if file_format:
                mesh = trimesh.load(filepath, file_type=file_format)
            else:
                mesh = trimesh.load(filepath)

            # Ensure we have a single mesh
            if isinstance(mesh, trimesh.Scene):
                # If it's a scene, try to get the first geometry
                geometries = list(mesh.geometry.values())
                if geometries:
                    mesh = geometries[0]
                else:
                    raise ValueError("No geometry found in mesh scene")

            if not isinstance(mesh, trimesh.Trimesh):
                raise ValueError(f"Loaded object is not a mesh: {type(mesh)}")

            self.mesh = mesh

            if self.verbose:
                logger.info(
                    "Loaded mesh: %d vertices, %d faces",
                    len(mesh.vertices),
                    len(mesh.faces),
                )

            return mesh

        except Exception as e:
            raise ValueError(f"Failed to load mesh from {filepath}: {str(e)}")

    def save(self, filepath, file_format="obj"):
        self.mesh.export(filepath, file_type=file_format)

    def copy(self):
        return MeshManager(self.mesh.copy())

    def to_trimesh(self):
        return self.mesh

    # combining the functions from utils into this class

    def analyze_mesh(self) -> dict:
        """
        Analyze and return mesh properties for diagnostic purposes.
        This function performs pure analysis without modifying the input mesh.

        Returns:
            Dictionary of mesh properties including volume, watertightness, winding consistency,
            face count, vertex count, bounds, and potential issues.
        """

        mesh = self.to_trimesh()
        # Initialize results dictionary
        results = {
            "face_count": len(mesh.faces),
            "vertex_count": len(mesh.vertices),
            "bounds": mesh.bounds.tolist() if hasattr(mesh, "bounds") else None,
            "is_watertight": mesh.is_watertight,
            "is_winding_consistent": mesh.is_winding_consistent,
            "issues": [],
        }

        # Calculate volume (report actual value, even if negative)
        try:
            results["volume"] = mesh.volume
            if mesh.volume < 0:
                results["issues"].append(
                    "Negative volume detected - face normals may be inverted"
                )
        except Exception as e:
            results["volume"] = None
            results["issues"].append(f"Volume calculation failed: {str(e)}")

        # Check for non-manifold edges
        try:
            if hasattr(mesh, "is_manifold"):
                results["is_manifold"] = mesh.is_manifold
                if not mesh.is_manifold:
                    results["issues"].append("Non-manifold edges detected")
        except Exception:
            results["is_manifold"] = None

        # Calculate topological properties using trimesh's built-in methods
        try:
            # Use trimesh's built-in euler_number property for correct topology calculation
            # For a sphere: euler_number = 2
            # For a torus: euler_number = 0
            # For a double torus: euler_number = -2
            # Genus = (2 - euler_number) / 2

            results["euler_characteristic"] = mesh.euler_number

            # Only calculate genus for closed (watertight) meshes
            if mesh.is_watertight:
                # For a closed orientable surface: genus = (2 - euler_number) / 2
                results["genus"] = int((2 - mesh.euler_number) / 2)

                # Sanity check - genus should be non-negative for simple shapes
                if results["genus"] < 0:
                    results["genus"] = (
                        0  # Default to 0 for simple shapes like spheres, cylinders
                    )
                    results["issues"].append(
                        "Calculated negative genus, defaulting to 0"
                    )
            else:
                # For non-watertight meshes, genus is not well-defined
                results["genus"] = None
                results["issues"].append("Genus undefined for non-watertight mesh")
        except Exception as e:
            results["genus"] = None
            results["euler_characteristic"] = None
            results["issues"].append(f"Topology calculation failed: {str(e)}")

        # Analyze face normals
        try:
            if hasattr(mesh, "face_normals") and mesh.face_normals is not None:
                # Get statistics on face normal directions
                results["normal_stats"] = {
                    "mean": mesh.face_normals.mean(axis=0).tolist(),
                    "std": mesh.face_normals.std(axis=0).tolist(),
                    "sum": mesh.face_normals.sum(axis=0).tolist(),
                }

                # Check if normals are predominantly pointing inward (negative volume)
                if results.get("volume", 0) < 0:
                    results["normal_direction"] = "inward"
                else:
                    results["normal_direction"] = "outward"
        except Exception as e:
            results["normal_stats"] = None
            results["issues"].append(f"Normal analysis failed: {str(e)}")

        # Check for duplicate vertices and faces
        try:
            unique_verts = np.unique(mesh.vertices, axis=0)
            results["duplicate_vertices"] = len(mesh.vertices) - len(unique_verts)
            if results["duplicate_vertices"] > 0:
                results["issues"].append(
                    f"Found {results['duplicate_vertices']} duplicate vertices"
                )
        except Exception:
            results["duplicate_vertices"] = None

        # Check for degenerate faces (zero area)
        try:
            if hasattr(mesh, "area_faces"):
                degenerate_count = np.sum(mesh.area_faces < 1e-8)
                results["degenerate_faces"] = int(degenerate_count)
                if degenerate_count > 0:
                    results["issues"].append(
                        f"Found {degenerate_count} degenerate faces"
                    )
        except Exception:
            results["degenerate_faces"] = None

        # Check for connected components
        try:
            components = mesh.split(only_watertight=False)
            results["component_count"] = len(components)
            if len(components) > 1:
                results["issues"].append(
                    f"Mesh has {len(components)} disconnected components"
                )
        except Exception:
            results["component_count"] = None

        return results

    def print_mesh_analysis(self, verbose: bool = False) -> None:
        """
        Analyze a mesh and print a formatted report of its properties.

        Args:
            verbose: Whether to print detailed information
        """
        analysis = self.analyze_mesh()

        print("Mesh Analysis Report")
        print("====================")

        # Basic properties
        print("\nGeometry:")
        print(f"  * Vertices: {analysis['vertex_count']}")
        print(f"  * Faces: {analysis['face_count']}")
        if analysis.get("component_count") is not None:
            print(f"  * Components: {analysis['component_count']}")
        if analysis.get("volume") is not None:
            print(f"  * Volume: {analysis['volume']:.2f}")
        if analysis.get("bounds") is not None:
            min_bound, max_bound = analysis["bounds"]
            print(
                f"  * Bounds: [{min_bound[0]:.1f}, {min_bound[1]:.1f}, {min_bound[2]:.1f}] to "
                f"[{max_bound[0]:.1f}, {max_bound[1]:.1f}, {max_bound[2]:.1f}]"
            )

        # Mesh quality
        print("\nMesh Quality:")
        print(f"  * Watertight: {analysis['is_watertight']}")
        print(f"  * Winding Consistent: {analysis['is_winding_consistent']}")
        if analysis.get("is_manifold") is not None:
            print(f"  * Manifold: {analysis['is_manifold']}")
        if analysis.get("normal_direction") is not None:
            print(f"  * Normal Direction: {analysis['normal_direction']}")
        if analysis.get("duplicate_vertices") is not None:
            print(f"  * Duplicate Vertices: {analysis['duplicate_vertices']}")
        if analysis.get("degenerate_faces") is not None:
            print(f"  * Degenerate Faces: {analysis['degenerate_faces']}")

        # Topology
        if (
            analysis.get("genus") is not None
            or analysis.get("euler_characteristic") is not None
        ):
            print("\nTopology:")
            if analysis.get("genus") is not None:
                print(f"  * Genus: {analysis['genus']}")
            if analysis.get("euler_characteristic") is not None:
                print(f"  * Euler Characteristic: {analysis['euler_characteristic']}")

        # Issues
        if analysis["issues"]:
            print(f"\nIssues Detected ({len(analysis['issues'])}):")
            for i, issue in enumerate(analysis["issues"]):
                print(f"  {i + 1}. {issue}")
        else:
            print("\nNo issues detected")

        # Detailed stats
        if verbose and analysis.get("normal_stats") is not None:
            print("\nNormal Statistics:")
            mean = analysis["normal_stats"]["mean"]
            sum_val = analysis["normal_stats"]["sum"]
            print(f"  * Mean: [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
            print(f"  * Sum: [{sum_val[0]:.4f}, {sum_val[1]:.4f}, {sum_val[2]:.4f}]")

        print("\nRecommendation:")
        if analysis["issues"]:
            print("  Consider using repair_mesh() to fix the detected issues.")
        else:
            print("  Mesh appears to be in good condition.")
        print("====================")

    def repair_mesh(
        self,
        fix_holes: bool = True,
        remove_duplicates: bool = True,
        fix_normals: bool = True,
        remove_degenerate: bool = True,
        fix_negative_volume: bool = True,
        keep_largest_component: bool = False,
        merge_vertices: bool = True,
        verbose: bool = False,
    ) -> trimesh.Trimesh:
        """
        Repair common mesh issues to improve watertightness and quality.

        This method applies a series of repair operations to fix common mesh problems
        such as negative volume, duplicate/degenerate faces, inconsistent normals,
        holes, and disconnected components.

        Parameters
        ----------
        fix_holes : bool, default=True
            Attempt to fill holes in the mesh to make it watertight.
        remove_duplicates : bool, default=True
            Remove duplicate faces and vertices.
        fix_normals : bool, default=True
            Fix face normal winding consistency.
        remove_degenerate : bool, default=True
            Remove degenerate (zero-area) faces.
        fix_negative_volume : bool, default=True
            Invert faces if mesh has negative volume.
        keep_largest_component : bool, default=False
            Keep only the largest connected component, discarding smaller pieces.
        merge_vertices : bool, default=True
            Merge vertices that are very close together.
        verbose : bool, default=False
            Print detailed repair summary.

        Returns
        -------
        trimesh.Trimesh
            The repaired mesh. Also updates self.mesh.

        Examples
        --------
        >>> mm = MeshManager("broken_mesh.obj")
        >>> mm.repair_mesh(verbose=True)
        >>> mm.save("fixed_mesh.obj")
        """
        mesh = self.mesh.copy()
        repair_log = []
        initial_stats = {
            "faces": len(mesh.faces),
            "vertices": len(mesh.vertices),
            "watertight": mesh.is_watertight,
            "winding_consistent": mesh.is_winding_consistent,
        }

        if verbose:
            print("\n" + "=" * 50)
            print("MESH REPAIR")
            print("=" * 50)
            print(f"\nInitial state:")
            print(f"  Faces: {initial_stats['faces']}")
            print(f"  Vertices: {initial_stats['vertices']}")
            print(f"  Watertight: {initial_stats['watertight']}")
            print(f"  Winding consistent: {initial_stats['winding_consistent']}")
            print()

        # Step 1: Fix negative volume
        if fix_negative_volume:
            try:
                if hasattr(mesh, "volume") and mesh.volume < 0:
                    initial_volume = mesh.volume
                    mesh.invert()
                    msg = f"Inverted faces (volume: {initial_volume:.2f} → {mesh.volume:.2f})"
                    repair_log.append(msg)
                    if verbose:
                        print(f"✓ {msg}")
            except Exception as e:
                msg = f"Failed to fix negative volume: {e}"
                repair_log.append(msg)
                if verbose:
                    print(f"✗ {msg}")

        # Step 2: Merge nearby vertices
        if merge_vertices:
            try:
                initial_verts = len(mesh.vertices)
                mesh.merge_vertices()
                merged = initial_verts - len(mesh.vertices)
                if merged > 0:
                    msg = f"Merged {merged} nearby vertices"
                    repair_log.append(msg)
                    if verbose:
                        print(f"✓ {msg}")
            except Exception as e:
                msg = f"Failed to merge vertices: {e}"
                repair_log.append(msg)
                if verbose:
                    print(f"✗ {msg}")

        # Step 3: Remove duplicates
        if remove_duplicates:
            try:
                initial_faces = len(mesh.faces)
                initial_verts = len(mesh.vertices)

                # Remove duplicate faces using unique_rows
                unique_faces, inverse = trimesh.grouping.unique_rows(mesh.faces)
                if len(unique_faces) < len(mesh.faces):
                    mesh.update_faces(unique_faces)

                # Remove unreferenced vertices
                mesh.remove_unreferenced_vertices()

                removed_faces = initial_faces - len(mesh.faces)
                removed_verts = initial_verts - len(mesh.vertices)
                if removed_faces > 0 or removed_verts > 0:
                    parts = []
                    if removed_faces > 0:
                        parts.append(f"{removed_faces} duplicate faces")
                    if removed_verts > 0:
                        parts.append(f"{removed_verts} unreferenced vertices")
                    msg = f"Removed {', '.join(parts)}"
                    repair_log.append(msg)
                    if verbose:
                        print(f"✓ {msg}")
            except Exception as e:
                msg = f"Failed to remove duplicates: {e}"
                repair_log.append(msg)
                if verbose:
                    print(f"✗ {msg}")

        # Step 4: Remove degenerate faces
        if remove_degenerate:
            try:
                initial_faces = len(mesh.faces)
                # Get non-degenerate faces (area > 0)
                valid = mesh.area_faces > 1e-8
                if not valid.all():
                    mesh.update_faces(valid)
                    removed = initial_faces - len(mesh.faces)
                    msg = f"Removed {removed} degenerate faces"
                    repair_log.append(msg)
                    if verbose:
                        print(f"✓ {msg}")
            except Exception as e:
                msg = f"Failed to remove degenerate faces: {e}"
                repair_log.append(msg)
                if verbose:
                    print(f"✗ {msg}")

        # Step 5: Fix normals
        if fix_normals:
            try:
                was_consistent = mesh.is_winding_consistent
                if not was_consistent:
                    mesh.fix_normals()
                    if mesh.is_winding_consistent:
                        msg = "Fixed face normal winding consistency"
                        repair_log.append(msg)
                        if verbose:
                            print(f"✓ {msg}")
                    else:
                        msg = "Attempted to fix normals (still inconsistent)"
                        repair_log.append(msg)
                        if verbose:
                            print(f"⚠ {msg}")
            except Exception as e:
                msg = f"Failed to fix normals: {e}"
                repair_log.append(msg)
                if verbose:
                    print(f"✗ {msg}")

        # Step 6: Fill holes
        if fix_holes:
            try:
                was_watertight = mesh.is_watertight
                if not was_watertight:
                    # Suppress numpy warnings during fill_holes
                    import warnings

                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=RuntimeWarning)
                        mesh.fill_holes()

                    if mesh.is_watertight:
                        msg = "Filled holes - mesh is now watertight"
                        repair_log.append(msg)
                        if verbose:
                            print(f"✓ {msg}")
                    else:
                        msg = "Attempted to fill holes (still not watertight)"
                        repair_log.append(msg)
                        if verbose:
                            print(f"⚠ {msg}")
            except Exception as e:
                msg = f"Failed to fill holes: {e}"
                repair_log.append(msg)
                if verbose:
                    print(f"✗ {msg}")

        # Step 7: Keep largest component
        if keep_largest_component:
            try:
                components = mesh.split(only_watertight=False)
                if len(components) > 1:
                    volumes = [
                        abs(c.volume) if hasattr(c, "volume") else len(c.faces)
                        for c in components
                    ]
                    largest_idx = np.argmax(volumes)
                    mesh = components[largest_idx]
                    msg = f"Kept largest of {len(components)} components"
                    repair_log.append(msg)
                    if verbose:
                        print(f"✓ {msg}")
            except Exception as e:
                msg = f"Failed to isolate largest component: {e}"
                repair_log.append(msg)
                if verbose:
                    print(f"✗ {msg}")

        # Step 8: Final processing
        try:
            mesh.process(validate=True)
            msg = "Applied final mesh processing"
            repair_log.append(msg)
            if verbose:
                print(f"✓ {msg}")
        except Exception as e:
            msg = f"Final processing failed: {e}"
            repair_log.append(msg)
            if verbose:
                print(f"✗ {msg}")

        # Store repair log in metadata
        if not hasattr(mesh, "metadata"):
            mesh.metadata = {}
        mesh.metadata["repair_log"] = repair_log

        # Print summary
        if verbose:
            print(f"\nFinal state:")
            volume = mesh.volume if hasattr(mesh, "volume") else "N/A"
            print(
                f"  Faces: {len(mesh.faces)} (Δ {len(mesh.faces) - initial_stats['faces']:+d})"
            )
            print(
                f"  Vertices: {len(mesh.vertices)} (Δ {len(mesh.vertices) - initial_stats['vertices']:+d})"
            )
            print(f"  Volume: {volume if volume == 'N/A' else f'{volume:.2f}'}")
            print(f"  Watertight: {mesh.is_watertight}")
            print(f"  Winding consistent: {mesh.is_winding_consistent}")
            print("\n" + "=" * 50 + "\n")

        self.mesh = mesh
        return mesh

    def visualize_mesh_3d(
        self,
        title: str = "3D Mesh Visualization",
        color: str = "lightblue",
        backend: str = "auto",
        show_axes: bool = True,
        show_wireframe: bool = False,
        width: int = 800,
        height: int = 600,
        *,
        skel: Optional[Union["SkeletonGraph", List["SkeletonGraph"]]] = None,
        skel_color: Union[str, List[str]] = "crimson",
        skel_line_width: float = 3.0,
        skel_opacity: float = 0.95,
    ) -> Optional[object]:
        """
        Create a 3D visualization of a mesh.

        Args:
            title: Plot title
            color: Mesh color (named color or RGB tuple)
            backend: Visualization backend ('plotly' or 'matplotlib')
            show_axes: Whether to show coordinate axes
            show_wireframe: Whether to show wireframe overlay
            skel: Optional SkeletonGraph or list of SkeletonGraph to overlay as 3D lines
            skel_color: Color(s) for skeleton overlay. Can be a single color or list of colors (one per skeleton)
            skel_line_width: Line width for skeleton overlay
            skel_opacity: Opacity for skeleton overlay (plotly only)

        Returns:
            Figure object (backend-dependent) or None if visualization fails
        """
        if backend == "auto":
            # Try plotly first, then fallback to matplotlib
            try:
                import plotly.graph_objects as go  # noqa: F401

                backend = "plotly"
            except ImportError:
                try:
                    import matplotlib.pyplot as plt  # noqa: F401

                    backend = "matplotlib"
                except ImportError:
                    backend = "plotly"

        if backend == "plotly":
            return self._visualize_mesh_plotly(
                title,
                color,
                show_axes,
                show_wireframe,
                width,
                height,
                skel=skel,
                skel_color=skel_color,
                skel_line_width=skel_line_width,
                skel_opacity=skel_opacity,
            )
        elif backend == "matplotlib":
            return self._visualize_mesh_matplotlib(
                title,
                color,
                show_axes,
                show_wireframe,
                skel=skel,
                skel_color=skel_color,
                skel_line_width=skel_line_width,
            )
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _visualize_mesh_plotly(
        self,
        title,
        color,
        show_axes,
        show_wireframe,
        width=800,
        height=600,
        *,
        skel: Optional[Union["SkeletonGraph", List["SkeletonGraph"]]] = None,
        skel_color: Union[str, List[str]] = "crimson",
        skel_line_width: float = 3.0,
        skel_opacity: float = 0.95,
    ):
        """Plotly-based mesh visualization with optional SkeletonGraph overlay."""
        try:
            import plotly.graph_objects as go

            vertices = self.mesh.vertices
            faces = self.mesh.faces

            # Create mesh trace
            mesh_trace = go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                opacity=0.8,
                color=color,
                name="Mesh",
            )

            fig = go.Figure(data=[mesh_trace])

            # Add wireframe if requested
            if show_wireframe:
                edge_x = []
                edge_y = []
                edge_z = []
                for face in faces:
                    for i in range(3):
                        v1, v2 = face[i], face[(i + 1) % 3]
                        edge_x += [vertices[v1][0], vertices[v2][0], None]
                        edge_y += [vertices[v1][1], vertices[v2][1], None]
                        edge_z += [vertices[v1][2], vertices[v2][2], None]
                fig.add_trace(
                    go.Scatter3d(
                        x=edge_x,
                        y=edge_y,
                        z=edge_z,
                        mode="lines",
                        line=dict(color="black", width=1),
                        name="Wireframe",
                    )
                )

            # Add skeleton overlay if provided
            if skel is not None:
                # Normalize to list
                if isinstance(skel, (list, tuple)):
                    skel_list = skel
                else:
                    skel_list = [skel]

                # Normalize colors to list
                if isinstance(skel_color, str):
                    colors = [skel_color] * len(skel_list)
                else:
                    colors = skel_color
                    if len(colors) < len(skel_list):
                        colors = list(colors) + [colors[-1]] * (
                            len(skel_list) - len(colors)
                        )

                # Add each skeleton
                for skel_idx, skeleton in enumerate(skel_list):
                    if skeleton is None:
                        continue

                    # Draw edges directly from the graph
                    color = colors[skel_idx]

                    # Collect all edge segments for this skeleton
                    edge_x = []
                    edge_y = []
                    edge_z = []

                    for u, v in skeleton.edges():
                        pos_u = skeleton.get_node_position(u)
                        pos_v = skeleton.get_node_position(v)

                        # Add edge as a line segment (with None separator for discontinuous lines)
                        edge_x.extend([pos_u[0], pos_v[0], None])
                        edge_y.extend([pos_u[1], pos_v[1], None])
                        edge_z.extend([pos_u[2], pos_v[2], None])

                    # Add all edges as a single trace
                    if edge_x:
                        fig.add_trace(
                            go.Scatter3d(
                                x=edge_x,
                                y=edge_y,
                                z=edge_z,
                                mode="lines",
                                line=dict(color=color, width=float(skel_line_width)),
                                opacity=float(skel_opacity),
                                name=f"Skeleton {skel_idx}",
                                showlegend=False,
                            )
                        )

            # Configure layout
            fig.update_layout(
                title=title,
                autosize=False,
                width=width,
                height=height,
                scene=dict(
                    aspectmode="data",
                    xaxis=dict(visible=show_axes),
                    yaxis=dict(visible=show_axes),
                    zaxis=dict(visible=show_axes),
                ),
            )

            return fig

        except ImportError:
            print("Plotly not available")
            return None
        except Exception as e:
            print(f"Plotly visualization failed: {e}")
            return None

    def _visualize_mesh_matplotlib(
        self,
        title,
        color,
        show_axes,
        show_wireframe,
        *,
        skel: Optional[Union["SkeletonGraph", List["SkeletonGraph"]]] = None,
        skel_color: Union[str, List[str]] = "crimson",
        skel_line_width: float = 3.0,
    ):
        """Matplotlib-based mesh visualization with optional SkeletonGraph overlay."""
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")

            vertices = self.mesh.vertices
            faces = self.mesh.faces

            # Create mesh surface
            poly3d = Poly3DCollection(
                vertices[faces],
                alpha=0.7,
                facecolor=color,
                edgecolor="black" if show_wireframe else None,
            )
            ax.add_collection3d(poly3d)

            # Add skeleton overlay if provided
            if skel is not None:
                # Normalize to list
                if isinstance(skel, (list, tuple)):
                    skel_list = skel
                else:
                    skel_list = [skel]

                # Normalize colors to list
                if isinstance(skel_color, str):
                    colors = [skel_color] * len(skel_list)
                else:
                    colors = skel_color
                    if len(colors) < len(skel_list):
                        colors = list(colors) + [colors[-1]] * (
                            len(skel_list) - len(colors)
                        )

                # Add each skeleton
                for skel_idx, skeleton in enumerate(skel_list):
                    if skeleton is None:
                        continue

                    # Draw edges directly from the graph
                    color = colors[skel_idx]

                    for u, v in skeleton.edges():
                        pos_u = skeleton.get_node_position(u)
                        pos_v = skeleton.get_node_position(v)

                        # Draw edge as a line segment
                        ax.plot(
                            [pos_u[0], pos_v[0]],
                            [pos_u[1], pos_v[1]],
                            [pos_u[2], pos_v[2]],
                            color=color,
                            linewidth=float(skel_line_width),
                        )

            ax.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
            ax.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
            ax.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())

            ax.set_xlabel("X (µm)")
            ax.set_ylabel("Y (µm)")
            ax.set_zlabel("Z (µm)")
            ax.set_title(title)

            if not show_axes:
                ax.set_axis_off()

            plt.tight_layout()
            return fig

        except ImportError:
            print("Matplotlib not available")
            return None
        except Exception as e:
            print(f"Matplotlib visualization failed: {e}")
            return None

    def visualize_mesh_slice_interactive(
        self,
        title: str = "Interactive Mesh Slice",
        z_range: Optional[Tuple[float, float]] = None,
        num_slices: int = 50,
        slice_color: str = "red",
        mesh_color: str = "lightblue",
        mesh_opacity: float = 0.3,
    ) -> Optional[object]:
        """
        Create an interactive 3D visualization of a mesh with a controllable slice plane.

        This function displays a 3D mesh and calculates the intersection of the mesh
        with an xy-plane at a user-controlled z-value. The intersection is shown as a
        colored line on the mesh. A slider allows the user to interactively change the
        z-value of the intersection plane.

        Args:
            title: Plot title
            z_range: Tuple of (min_z, max_z) for slice range. Auto-detected if None.
            num_slices: Number of positions for the slider
            slice_color: Color for the intersection line
            mesh_color: Color for the 3D mesh
            mesh_opacity: Opacity of the 3D mesh (0-1)

        Returns:
            Plotly figure with interactive slider for controlling the z-value
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            print("Plotly is required for interactive visualization")
            return None

        mesh = self.mesh

        # Determine z-range if not provided
        if z_range is None:
            z_min, z_max = mesh.vertices[:, 2].min(), mesh.vertices[:, 2].max()
            # Add small padding
            padding = (z_max - z_min) * 0.05
            z_min -= padding
            z_max += padding
        else:
            z_min, z_max = z_range

        # Create the base figure with the mesh
        fig = go.Figure()

        # Add the mesh to the figure
        fig.add_trace(
            go.Mesh3d(
                x=mesh.vertices[:, 0],
                y=mesh.vertices[:, 1],
                z=mesh.vertices[:, 2],
                i=mesh.faces[:, 0],
                j=mesh.faces[:, 1],
                k=mesh.faces[:, 2],
                opacity=mesh_opacity,
                color=mesh_color,
                name="Mesh",
            )
        )

        # Function to create a slice at a given z-value
        def create_slice_trace(z_value):
            # Calculate intersection with plane at z_value
            section = mesh.section(plane_origin=[0, 0, z_value], plane_normal=[0, 0, 1])

            # If no intersection, return None
            if (
                section is None
                or not hasattr(section, "entities")
                or len(section.entities) == 0
            ):
                return None

            # Process all entities in the section to get 3D coordinates
            all_points = []

            for entity in section.entities:
                if hasattr(entity, "points") and len(entity.points) > 0:
                    # Get the actual 2D coordinates
                    points_2d = section.vertices[entity.points]

                    # Convert to 3D by adding z_value
                    points_3d = np.column_stack(
                        [points_2d, np.full(len(points_2d), z_value)]
                    )

                    # Add closing point if needed (to complete the loop)
                    if len(points_2d) > 2 and not np.array_equal(
                        points_2d[0], points_2d[-1]
                    ):
                        closing_point = np.array(
                            [points_2d[0][0], points_2d[0][1], z_value]
                        )
                        points_3d = np.vstack([points_3d, closing_point])

                    # Add to all points list
                    all_points.extend(points_3d.tolist())

                    # Add None to create a break between separate entities
                    all_points.append([None, None, None])

            # If we have points, create a scatter trace
            if all_points:
                x_coords = [p[0] if p is not None else None for p in all_points]
                y_coords = [p[1] if p is not None else None for p in all_points]
                z_coords = [p[2] if p is not None else None for p in all_points]

                return go.Scatter3d(
                    x=x_coords,
                    y=y_coords,
                    z=z_coords,
                    mode="lines",
                    line=dict(color=slice_color, width=5),
                    name=f"Slice at z={z_value:.2f}",
                )

            return None

        # Create initial slice
        initial_z = (z_min + z_max) / 2
        initial_slice = create_slice_trace(initial_z)

        # Add initial slice to figure if it exists
        if initial_slice:
            fig.add_trace(initial_slice)

        # Create frames for animation
        frames = []
        for i, z_val in enumerate(np.linspace(z_min, z_max, num_slices)):
            # Create a slice at this z-value
            slice_trace = create_slice_trace(z_val)

            # If we have a valid slice, add it to frames
            if slice_trace:
                frame_data = [fig.data[0], slice_trace]  # Mesh and slice
            else:
                frame_data = [fig.data[0]]  # Just the mesh

            frames.append(
                go.Frame(
                    data=frame_data,
                    name=f"frame_{i}",
                    traces=[0, 1],  # Update both traces
                )
            )

        # Create slider steps
        steps = []
        for i, z_val in enumerate(np.linspace(z_min, z_max, num_slices)):
            step = dict(
                args=[
                    [f"frame_{i}"],
                    {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"},
                ],
                label=f"{z_val:.2f}",
                method="animate",
            )
            steps.append(step)

        # Configure the slider
        sliders = [
            dict(
                active=num_slices // 2,  # Start in the middle
                currentvalue={
                    "prefix": "Z-value: ",
                    "visible": True,
                    "xanchor": "right",
                },
                pad={"t": 50, "b": 10},
                len=0.9,
                x=0.1,
                y=0,
                steps=steps,
            )
        ]

        # Configure the figure layout
        fig.update_layout(
            title=title,
            scene=dict(aspectmode="data", camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))),
            height=800,  # Taller to make room for slider
            margin=dict(l=50, r=50, b=100, t=100),  # Add margin at bottom for slider
            sliders=sliders,
            # Add animation controls
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    y=0,
                    x=0,
                    xanchor="left",
                    yanchor="top",
                    pad=dict(t=60, r=10),
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[
                                None,
                                {
                                    "frame": {"duration": 200, "redraw": True},
                                    "fromcurrent": True,
                                },
                            ],
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                },
                            ],
                        ),
                        dict(
                            label="Reset View",
                            method="relayout",
                            args=[{"scene.camera.eye": dict(x=1.5, y=1.5, z=1.5)}],
                        ),
                    ],
                )
            ],
        )

        # Set frames
        fig.frames = frames

        return fig

    # =================================================================
    # BOOLEAN OPERATIONS
    # =================================================================

    def subtract_mesh(
        self,
        other: Union[trimesh.Trimesh, "MeshManager"],
        repair_before: bool = True,
        repair_after: bool = True,
        engine: str = "blender",
        validate: bool = True,
    ) -> "MeshManager":
        """
        Robustly subtract another mesh from this mesh using boolean difference.

        This method performs a CSG (Constructive Solid Geometry) difference operation,
        subtracting the 'other' mesh from the current mesh. It includes options for
        automatic mesh repair before and after the operation to ensure robust results.

        Parameters
        ----------
        other : trimesh.Trimesh or MeshManager
            The mesh to subtract from the current mesh. If a MeshManager is provided,
            its underlying trimesh will be used.
        repair_before : bool, default=True
            Whether to repair both meshes before performing the subtraction.
            This can help ensure the boolean operation succeeds by fixing common
            mesh issues like non-watertight geometry or inconsistent winding.
        repair_after : bool, default=True
            Whether to repair the result mesh after subtraction.
            This helps clean up any artifacts from the boolean operation.
        engine : str, default="blender"
            The boolean operation engine to use. Options:
            - "blender": Uses Blender's boolean modifier (most robust, requires Blender)
            - "manifold": Uses the manifold library (fast, pure Python)
            - "scad": Uses OpenSCAD (requires OpenSCAD installation)
        validate : bool, default=True
            Whether to validate that the result is a valid mesh.
            If validation fails, raises an error with diagnostic information.

        Returns
        -------
        MeshManager
            A new MeshManager instance containing the resulting mesh after subtraction.

        Raises
        ------
        ValueError
            If the subtraction operation fails or produces invalid results.
        RuntimeError
            If the required boolean engine is not available.

        Examples
        --------
        >>> mm1 = MeshManager("cube.obj")
        >>> mm2 = MeshManager("sphere.obj")
        >>> result_mm = mm1.subtract_mesh(mm2)
        >>> result_mm.save("cube_minus_sphere.obj")

        Notes
        -----
        - For best results, ensure both meshes are watertight and manifold
        - The 'blender' engine is generally most robust but requires Blender installation
        - The 'manifold' engine is fastest and doesn't require external dependencies
        - If subtraction fails, try enabling repair_before=True or switching engines
        """
        # Extract trimesh from MeshManager if needed
        if isinstance(other, MeshManager):
            other_mesh = other.to_trimesh()
        else:
            other_mesh = other

        # Store original mesh for potential rollback
        original_mesh = self.mesh.copy()

        try:
            # Step 1: Repair meshes before operation if requested
            if repair_before:
                if self.verbose:
                    logger.info("Repairing meshes before subtraction...")

                # Repair current mesh
                self.repair_mesh()

                # Repair other mesh (create temporary manager if needed)
                if isinstance(other, MeshManager):
                    other.repair_mesh()
                    other_mesh = other.to_trimesh()
                else:
                    temp_manager = MeshManager(other_mesh.copy(), verbose=self.verbose)
                    temp_manager.repair_mesh()
                    other_mesh = temp_manager.to_trimesh()

            # Step 2: Perform the boolean difference operation
            if self.verbose:
                logger.info("Performing boolean difference with engine: %s", engine)

            result_mesh = self.mesh.difference(other_mesh, engine=engine)

            # Check if result is valid
            if result_mesh is None or len(result_mesh.vertices) == 0:
                raise ValueError(
                    "Boolean difference produced empty or null result. "
                    "Meshes may not intersect or operation failed."
                )

            # Step 3: Repair result if requested
            if repair_after:
                if self.verbose:
                    logger.info("Repairing result mesh...")

                temp_manager = MeshManager(result_mesh, verbose=self.verbose)
                temp_manager.repair_mesh()
                result_mesh = temp_manager.to_trimesh()

            # Step 4: Validate result if requested
            if validate:
                if self.verbose:
                    logger.info("Validating result mesh...")

                issues = []
                if not result_mesh.is_watertight:
                    issues.append("Result mesh is not watertight")
                if not result_mesh.is_winding_consistent:
                    issues.append("Result mesh has inconsistent winding")
                if result_mesh.volume <= 0:
                    issues.append(
                        f"Result mesh has invalid volume: {result_mesh.volume}"
                    )

                if issues and self.verbose:
                    logger.warning("Validation warnings: %s", "; ".join(issues))

            # Step 5: Create new MeshManager instance with result
            result_manager = MeshManager(result_mesh, verbose=self.verbose)

            if self.verbose:
                logger.info(
                    "Subtraction successful: %d vertices, %d faces",
                    len(result_mesh.vertices),
                    len(result_mesh.faces),
                )

            return result_manager

        except Exception as e:
            # Rollback to original mesh on failure
            self.mesh = original_mesh

            error_msg = f"Mesh subtraction failed: {str(e)}"

            # Provide helpful error messages based on common issues
            if "engine" in str(e).lower():
                error_msg += (
                    f"\n\nThe '{engine}' engine may not be available. "
                    "Try installing the required dependencies or use a different engine:\n"
                    "  - 'manifold': pip install manifold3d\n"
                    "  - 'blender': requires Blender installation\n"
                    "  - 'scad': requires OpenSCAD installation"
                )
            elif "watertight" in str(e).lower() or "manifold" in str(e).lower():
                error_msg += (
                    "\n\nMeshes may not be watertight or manifold. "
                    "Try setting repair_before=True to fix mesh issues before subtraction."
                )

            if self.verbose:
                logger.error(error_msg)
                logger.debug("Full traceback:", exc_info=True)

            raise ValueError(error_msg) from e
