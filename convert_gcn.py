import torch
from torch_geometric.datasets import Planetoid
import openvino as ov
from pathlib import Path

# Load model function
def load_model(model_name, path, num_features, num_classes, data):
    # Import the model dynamically based on the model name
    if model_name == "GCN_StaGr":
        from models import GCN_StaGr as GCNModel
        model = GCNModel(num_features, num_classes, data.x, data.edge_index)
    elif model_name == "GCN_GrAd_NodePad":
        from models import GCN_GrAd_NodePad as GCNModel
        model = GCNModel(num_features, num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    model.load_state_dict(torch.load(path))  # Load the saved state dictionary
    return model


# Main execution
if __name__ == "__main__":
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
    
    # Specify model details
    model_name = "GCN_StaGr"
    model_path = f'torch_models/{model_name}_{dataset_name}.pth'
    
    # Load the trained model
    model = load_model(model_name, model_path, dataset.num_features, dataset.num_classes, data)

    ##################################### Extra nodes #####################################
    num_dummy_nodes = 0 #292   ## User input
    #######################################################################################
  
    # Convert model to OpenVINO format
    if model_name in ["GCN_GrAd_NodePad"]:
        ov_model = ov.convert_model(model, input=[("x", ov.Shape([data.num_nodes+num_dummy_nodes, dataset.num_features]), ov.Type.f32), ("norm", ov.Shape([data.num_nodes+num_dummy_nodes, data.num_nodes+num_dummy_nodes]), ov.Type.f32)])
    else:  # For models that do not use edge_index
        ov_model = ov.convert_model(model, input=[("x", ov.Shape([data.num_nodes, dataset.num_features]), ov.Type.f32)])

    # Save the model as IR (xml and bin files)
    output_dir = Path("ov_model")
    output_dir.mkdir(parents=True, exist_ok=True)
    if model_name in ["GCN_GrAd_NodePad"]:
        # ov.save_model(ov_model, str(output_dir / f"{model_name}_{num_dummy_nodes}_{dataset_name}_fp32.xml"), compress_to_fp16=False)
        ov.save_model(ov_model, str(output_dir / f"{model_name}_{num_dummy_nodes}_{dataset_name}_fp16.xml"), compress_to_fp16=True)
    else:
        # ov.save_model(ov_model, str(output_dir / f"{model_name}_{dataset_name}_fp32.xml"), compress_to_fp16=False)
        ov.save_model(ov_model, str(output_dir / f"{model_name}_{dataset_name}_fp16.xml"), compress_to_fp16=True)

    print(f"Model {model_name} converted and saved successfully.")
