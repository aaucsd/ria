"""RIA — locate radiation interactions from a plenoptic single-photon image,
and design the lens array that makes it possible.

Every physical quantity is explicit; nothing about the geometry is assumed.

    import ria

    system = ria.imager()
    system.addScint(size=(30, 30, 40), working_distance=5.0)
    system.addLens(lenses=[{"position": (0.0, 0.0), "f": 9.0, "aperture": 1.5}],
                   blur_floor=0.12)
    system.addDetector(size=(46.15, 32.84), pixels=(24, 24),
                       imaging_distance=20.477, dark_count=1e-9)

    s1    = system.addVertex([2, 2, 20], photons=80)
    image = system.forward(s1, seed=1)
    print(system.invert(image, truth=s1, budget=30))
"""
from .system import imager, Imager, Scintillator, Vertex, Inversion
from .optics import (LensPlane, SensorPlane, hex_layout, grid_layout,
                     image_rate, blob_geometry, n_visible)
from .design import (LensArray, depth_crb, depth_fisher, crb_map,
                     design_score, optimise_focal_lengths,
                     emit_design_model, optimise_focal_lengths_rekin,
                     blob_centres, blur_radii)

__all__ = [
    # building and running a system
    "imager", "Imager", "Scintillator", "Vertex", "Inversion",
    # optics, stated explicitly
    "LensPlane", "SensorPlane", "hex_layout", "grid_layout",
    "image_rate", "blob_geometry", "n_visible",
    # design (Fisher / Cramér-Rao)
    "LensArray", "depth_crb", "depth_fisher", "crb_map", "design_score",
    "optimise_focal_lengths", "emit_design_model",
    "optimise_focal_lengths_rekin", "blob_centres", "blur_radii",
]

__version__ = "0.3.0"
