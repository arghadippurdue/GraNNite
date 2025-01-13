# GraNNite: Enabling High-Performance Execution of Graph Neural Networks on Resource-Constrained Neural Processing Units

This repository provides tools and scripts to train, convert, and run inference on Graph Neural Networks (GNNs) using the Cora, Citeseer, and PubMed datasets. Leveraging OpenVINO's Intermediate Representation (IR) models ensures efficient deployment across multiple devices.

---

## Installation

Follow the steps below to set up the environment:

```bash
# Create and activate a virtual environment
conda create -n gnn
conda activate gnn

# Install PyTorch and Torch Geometric
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
pip install torch_geometric

# Install additional dependencies
pip install openvino onnx scipy
```

---

## Workflow Steps

1. **Train the Model**
   - **Script**: `train_gcn.py`
   - **Description**: Trains GCN models using the Cora, Citeseer, or PubMed dataset and saves the trained PyTorch model.
   - **Output**: Saved PyTorch models in the `torch_models/` directory.

2. **Convert PyTorch Model to OpenVINO IR**
   - **Script**: `convert_gcn.py`
   - **Description**: Converts the trained PyTorch models into OpenVINO IR format for optimized inference.
   - **Output**: Converted IR models in the `ov_model/` directory.

3. **Run Inference with OpenVINO IR Models**
   - **Script**: `infer_IR.py`
   - **Description**: Performs inference on the OpenVINO IR models. Provides accuracy and inference time metrics.
   - **Supported Devices**: CPU, GPU, VPU, etc.

4. **Benchmarking**
   - **Script**: `benchmark.py`
   - **Description**: Benchmarks the performance of OpenVINO IR models. Reports detailed latency and throughput metrics.

---

## Directory Structure

```plaintext
├── gcn/                # Original GCN implementation
├── torch_models/       # Trained PyTorch models
├── ov_model/           # OpenVINO IR models
├── train_gcn.py        # Script to train the model
├── convert_gcn.py      # Script to convert PyTorch models to OpenVINO IR
├── infer_IR.py         # Script to run inference with OpenVINO IR
├── benchmark.py        # Script for performance benchmarking
└── README.md           # Documentation
```

---

## Example Usage

1. **Train a GCN Model**:
   ```bash
   python train_gcn.py
   ```

2. **Convert the Model**:
   ```bash
   python convert_gcn.py
   ```

3. **Run Inference**:
   ```bash
   python infer_IR.py -d CPU
   ```

4. **Benchmark Performance**:
   ```bash
   python benchmark.py
   ```
