# Loopless Riemannian Subspace Projection for High-Density BCI Fault Tolerance

Fault-tolerant Brain-Computer Interface (BCI) pipeline optimizing Riemannian manifold geometry for embedded edge hardware under real-time sensor dropout conditions. Collapses missing-channel data repair and tangent-space classification weights into an $O(C^2)$ element-wise trace product. Cross-validated on PhysioNet MI and Graz datasets via MOABB for embedded edge deployment.

## Key Engineering Innovation
Instead of running expensive analytical matrix operations ($O(C^3)$ matrix log-maps) at runtime to interpolate dropped electrodes, this architecture moves the transformation matrices into a one-time static calibration phase. By pulling the linear classifier's weights backward through a partitioned matrix layer, live inference is reduced to a flat, low-power element-wise product.

## Repository Structure
```text
loopless-riemannian-bci/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── generate_manifest.py     # Master execution script
└── src/
    ├── __init__.py          # Module boundary indicator
    └── engine.py            # Core Riemannian manifold processing operations
