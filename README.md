# Random Walks on Polytopes

Experiments with polytope generation and (future) random walks in R^n.

## Structure

```
random_walks_polytopes/
├── point_generators.py   # Gaussian / Uniform / Dirichlet, normal & elongated
├── polytope_builder.py   # ConvexHull -> H-repr (Ax ≤ b), volestipy wrapper, I/O
├── visualization.py      # 2D/3D plotting only
├── main.py               # CLI pipeline orchestration
└── requirements.txt
```

## Install

TODO, near future I hope

## Run

```bash
# Default: 2D, 200 points, all distributions & variants
python main.py

# 3D, 500 points
python main.py --dim 3 --npoints 500

# 5D, no visualization
python main.py --dim 5 --npoints 1000 --no-viz

# Specific distribution/variant only
python main.py --dim 2 --distributions gaussian --variants elongated

# With volestipy wrapping
python main.py --dim 2 --volestipy

# Interactive plots
python main.py --dim 2 --show
```

## Pipeline

```
generate(n, N, dist, variant)
       │
       ▼
build_polytope(points)  ->  HPolytope(A, b, vertices)
       │
       ├── export_polytope()  ->  outputs/<tag>/polytope.{npz,json}
       ├── export_points()    ->  outputs/<tag>/points.npz
       ├── to_volestipy()     ->  volestipy.HPolytope  [ready for random walks]
       └── visualize()        ->  outputs/<tag>/plot_<tag>.png  (n=2,3 only)
```

## Distributions

| Name       | Description                              | Elongated transform |
|------------|------------------------------------------|---------------------|
| gaussian   | Isotropic standard normal                | axis k scaled by 2^k |
| uniform    | Uniform on [-1,1]^n hypercube            | axis k scaled by 2^k |
| dirichlet  | Dirichlet(0.5,...,0.5) on the simplex    | axis k scaled by 2^k |


## Outputs

Each experiment produces a sub-directory in `outputs/`:

```
outputs/
└── gaussian_normal_n2/
    ├── points.npz          # raw point cloud
    ├── polytope.npz        # A (m×n), b (m,), vertices (k×n)
    └── polytope.json       # metadata + constraint/vertex counts
    └── plot_gaussian_normal_n2.png
```

## Extending to Random Walks

The `HPolytope` dataclass and `to_volestipy()` are designed for easy extension.

## Some troubleshooting for the volestipy

The process I used to make volestipy work was the following:

### 1. Clone the repository

```bash
git clone https://github.com/GeomScale/volestipy.git
cd volestipy
volestipy requires building the C++ backend and Python bindings from source.
```

### 2. Download external dependencies

The project depends on lp_solve, which I had to download manually:

```
wget -P external/ \
https://sourceforge.net/projects/lpsolve/files/lpsolve/5.5.2.11/lp_solve_5.5.2.11_source.tar.gz

tar xzf external/lp_solve_5.5.2.11_source.tar.gz -C external/
```

If needed, ensure the extracted folder is correctly placed inside external/.

### 3. Configure the build with CMake
```
mkdir -p build
cd build

cmake .. \
  -DVOLESTI_INCLUDE_DIR=../external/volesti/include \
  -DCMAKE_BUILD_TYPE=Release
```

### 4. Compile the project

```
cmake --build . -j$(nproc)
```

### 5. Fix shared library location (if needed)

In some cases, the compiled Python extension may need to be manually placed in the package folder:

```
cp build/_volestipy*.so ../volestipy/
```

### 6. Fix Python packaging configuration

Ensure pyproject.toml includes package discovery:

```
[tool.setuptools.packages.find]
where = ["."]
include = ["volestipy*"]
```

### 7. Install Python package (editable mode)

From the repository root:

```
pip install -e .
```

### 8. Verify installation
```
cd ~
python -c "import volestipy; print(volestipy.__file__)"
```

