import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj
from models_gcn import GCN_StaGr, GCN_GrAd_NodePad

# Print information
def print_dataset_info(dataset, data):
    print(dataset)
    print('------------')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')
    print(f'Graph: {data}')
    print(f'x = {data.x.shape}')
    print(data.x)
    print(data.x.dtype)
    print(f'edge_index = {data.edge_index.shape}')
    print(data.edge_index)
    print(data.edge_index.dtype)

    A = to_dense_adj(data.edge_index)[0].numpy().astype(int)
    print(f'A = {A.shape}')
    print(A)
    print(f'y = {data.y.shape}')
    print(data.y)
    print(f'train_mask = {data.train_mask.shape}')
    print(data.train_mask)
    print(f'Edges are directed: {data.is_directed()}')
    print(f'Graph has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Graph has loops: {data.has_self_loops()}')


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
    return ((pred_y == y).sum() / len(y)).item()


def train(model, data, norm, optimizer, criterion, epochs=100, model_type="default"):
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        if model_type in ["GCN_pyG", "GCN_pyT", "GCN_v2"]:
            z = model(data.x, data.edge_index)
        elif model_type in ["GCN_v6", "GCN_v7", "GCN_v8", "GCN_v9", "GCN_v10"]:
            z = model(data.x, norm)
        else:
            z = model(data.x)
        
        # Calculate training loss and accuracy using train_mask
        loss = criterion(z[data.train_mask], data.y[data.train_mask])
        acc = accuracy(z[data.train_mask].argmax(dim=1), data.y[data.train_mask])
        
        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        val_loss = criterion(z[data.val_mask], data.y[data.val_mask])
        val_acc = accuracy(z[data.val_mask].argmax(dim=1), data.y[data.val_mask])

        if epoch % 10 == 0:
            print(f'Epoch {epoch:>3} | Train Loss: {loss:.2f} | Train Acc: {acc*100:.2f}% | Val Loss: {val_loss:.2f} | '
                      f'Val Acc: {val_acc*100:.2f}%')
    return model

# Testing function
def test(model, data, norm, model_type="default"):
    model.eval()
    with torch.no_grad():
        if model_type in ["GCN_GrAd_NodePad"]:
            z = model(data.x, norm)
        else:
            z = model(data.x)
        acc = accuracy(z.argmax(dim=1)[data.test_mask], data.y[data.test_mask])
    print(f'Test Accuracy: {acc * 100:.2f}%')
    return acc

# Save model function
def save_model(model, path):
    torch.save(model.state_dict(), path)

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
    print("Before update:")
    print_dataset_info(dataset, data)
    
    # Define models
    models = {
        "GCN_StaGr": GCN_StaGr(dataset.num_features, dataset.num_classes, data.x, data.edge_index),
        "GCN_GrAd_NodePad": GCN_GrAd_NodePad(dataset.num_features, dataset.num_classes)
    }

    # Define optimizer and loss function
    criterion = torch.nn.CrossEntropyLoss()
    optimizer_params = {'lr': 0.01, 'weight_decay': 5e-4}  # Define common optimizer parameters

    # Train, save, and test each model
    for model_name, model in models.items():
        # Add dummy nodes
        if model_name in ["GCN_GrAd_NodePad"]:
            num_dummy_nodes = 292   ## User input
            data = add_dummy_nodes(data, num_dummy_nodes)
            print("After update:")
            print_dataset_info(dataset, data)

        # Step 1: Form the adjacency matrix
        num_nodes = data.x.size(0)
        adjacency = torch.zeros(num_nodes, num_nodes, device=data.x.device)

        # Fill the adjacency matrix with edge connections
        adjacency[data.edge_index[0], data.edge_index[1]] = 1
        
        # Step 2: Add self-loops
        adjacency += torch.eye(num_nodes)

        # Step 3: Compute degree matrix
        degree = adjacency.sum(dim=1, keepdim=True)
        # print(degree)

        # Step 4: Normalize the adjacency matrix
        norm = torch.where(degree > 0, degree.pow(-0.5), torch.zeros_like(degree))
        norm = norm * norm.t()
        norm = norm * adjacency

        print(f"\nTraining {model_name}...")
        
        optimizer = torch.optim.Adam(model.parameters(), **optimizer_params)
        
        trained_model = train(model, data, norm, optimizer, criterion, epochs=101, model_type=model_name)
        
        os.makedirs("torch_models", exist_ok=True)
        save_model(trained_model, path=f"torch_models/{model_name}_{dataset_name}.pth")
        
        test(trained_model, data, norm, model_type=model_name)

        # Print final embeddings and outputs
        if model_name in ["GCN_GrAd_NodePad"]:
            z = model(data.x, norm)
        else:
            z = trained_model(data.x)

        print(f'Final outputs = {z.shape}')
        print(z.argmax(dim=1))
        print(f'Golden classes = {data.y.shape}')
        print(data.y)
