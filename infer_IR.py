import argparse
import time
import numpy as np
from pathlib import Path
import torch
from torch_geometric.datasets import Planetoid
import openvino as ov

def add_dummy_nodes(data, buffer_nodes):
    # Determine original dimensions
    original_num_nodes = data.x.size(0)
    num_features = data.x.size(1)

    # Add buffer nodes to x
    dummy_x = torch.zeros((buffer_nodes, num_features), dtype=data.x.dtype)
    data.x = torch.cat([data.x, dummy_x], dim=0)

    # Add dummy labels to y
    dummy_y = torch.full((buffer_nodes,), -1, dtype=data.y.dtype)
    data.y = torch.cat([data.y, dummy_y], dim=0)

    # Add False values to train_mask, val_mask, and test_mask
    dummy_mask = torch.full((buffer_nodes,), False, dtype=torch.bool)
    data.train_mask = torch.cat([data.train_mask, dummy_mask], dim=0)
    data.val_mask = torch.cat([data.val_mask, dummy_mask], dim=0)
    data.test_mask = torch.cat([data.test_mask, dummy_mask], dim=0)

    return data

# Define accuracy calculation
def accuracy(pred_y, y):
    return (pred_y == y).sum().item() / len(y)

# Preprocess data for inference
def preprocess_data(data):
    x = data.x.numpy().astype(np.float32)
    return x

def preprocess_data_norm(data):
    x = data.x.numpy().astype(np.float32)

    # Step 1: Form the adjacency matrix
    num_nodes = data.x.size(0)
    adjacency = torch.zeros(num_nodes, num_nodes, device=data.x.device)

    # Fill the adjacency matrix with edge connections
    adjacency[data.edge_index[0], data.edge_index[1]] = 1
    
    # Step 2: Add self-loops
    adjacency += torch.eye(num_nodes)

    # Step 3: Compute degree matrix
    degree = adjacency.sum(dim=1, keepdim=True)

    # Step 4: Normalize the adjacency matrix
    norm = torch.where(degree > 0, degree.pow(-0.5), torch.zeros_like(degree))
    norm = norm * norm.t()
    norm = norm * adjacency

    norm = norm.numpy().astype(np.float32)

    return x, norm

def main(model_name, device_name):
    # Load dataset
    dataset_name = 'citeseer' # Can be 'cora', 'citeseer', or 'pubmed'

    # Load dataset
    if dataset_name == 'cora':
        dataset = Planetoid(root='.', name="Cora")
    elif dataset_name == 'citeseer':
        dataset = Planetoid(root='.', name="CiteSeer")
    elif dataset_name == 'pubmed':
        dataset = Planetoid(root='.', name="PubMed")

    data = dataset[0]

    if model_name in ["GCN_GrAd_NodePad"]:
        num_dummy_nodes = 0 # 292   ## User input
        data = add_dummy_nodes(data, num_dummy_nodes)

    # Preprocess data for inference
    if model_name in ["GCN_GrAd_NodePad"]:
        x, norm = preprocess_data_norm(data)
    else:
        x = preprocess_data(data)

    # Load OpenVINO model
    core = ov.Core()
    if model_name in ["GCN_GrAd_NodePad"]:
        # model_path = Path(f"ov_model/{model_name}_{num_dummy_nodes}_fp32.xml")
        model_path = Path(f"ov_model/{model_name}_{num_dummy_nodes}_{dataset_name}_fp16.xml")
    else:
        # model_path = Path(f"ov_model/{model_name}_fp32.xml")
        model_path = Path(f"ov_model/{model_name}_{dataset_name}_fp16.xml")
    
    print('Compiling the model, ', model_path)
    compiled_model = core.compile_model(model=model_path, device_name=device_name)
    output_index = compiled_model.outputs[0].index

    # Print available devices and their full names
    devices = core.available_devices
    for device in devices:
        device_full_name = core.get_property(device, "FULL_DEVICE_NAME")
        print(f"{device}: {device_full_name}")

    # Perform inference
    num_iterations = 1
    num_warmup_iterations = 0
    total_time = 0

    for i in range(num_iterations):
        infer_request = compiled_model.create_infer_request()
        start_time = time.perf_counter()
        if model_name in ["GCN_GrAd_NodePad"]:
            infer_request.infer(inputs={"x": x, "norm": norm})
        else:
            infer_request.infer(inputs={"x": x})
        elapsed_time = time.perf_counter() - start_time
        result_infer = infer_request.get_output_tensor(output_index).data
        result_infer = np.squeeze(result_infer)

        if i >= num_warmup_iterations:
            total_time += elapsed_time

        if i == 0:
            # Print the first inference result for verification
            print('Model outputs shape: ', result_infer.shape)
            result_index = np.argmax(result_infer, axis=1)
            print('Model class outputs: ', result_index)
            print('Golden class outputs: ', data.y.numpy())
            test_acc = accuracy(result_index[data.test_mask], data.y.numpy()[data.test_mask])
            print(f"First inference test accuracy: {test_acc * 100:.2f}%")

    # Calculate the average time excluding the warmup iterations
    average_time = total_time / (num_iterations - num_warmup_iterations)
    print(f"Average inference time (from iteration {num_warmup_iterations} to {num_iterations}) = {average_time:.6f} seconds")

    # Final test accuracy
    final_result = np.argmax(result_infer, axis=1)
    final_test_acc = accuracy(final_result[data.test_mask], data.y.numpy()[data.test_mask])
    print(f"\n{model_name.upper()} final test accuracy: {final_test_acc * 100:.2f}%\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a saved model using OpenVINO IR format.")
    parser.add_argument("-m", "--model", type=str, default='GCN_GrAd_NodePad', choices=["GCN_StaGr", "GCN_GrAd_NodePad"],
                        help="The model to test (GCN_StaGr, GCN_GrAd_NodePad).")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="The target device to use for inference (default: CPU).")
    args = parser.parse_args()

    main(args.model, args.device)
