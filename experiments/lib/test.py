

def cuda_to_numpy(tensor):
    """Converts a cuda tensor to a numpy array in place
    Positional argument:
        tensor -- Tensor to convert to numpy array 
    """
    import torch

    if tensor.requires_grad:
        return tensor.detach().cpu().numpy()
    else:
        return tensor.cpu().numpy()

def load_test_data(data_path, test_size):
    from torch.utils.data import DataLoader
    import h5py
    
    testdata = h5py.File(data_path, "r")
    
    testset = testdata["X"][:test_size]
    testlabels = testdata["pwave"][:test_size]
    
    testset = list(zip(testset, testlabels))

    batch_size=250
    
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

    testdata.close()
    
    return testloader

def test_model(model, data_path, test_size, device="cpu"):
    """
    Arguments:
        model -- Model to validate with validation data 
        testloader -- DataLoader with labels and data
    Returns:
        None
    """
    import numpy as np
    import torch

    testloader = load_test_data(data_path, test_size)
    
    y_pred = np.array([])
    y_true = np.array([])
    y_probs = np.array([])
    
    for batch, labels in testloader:
        batch, labels = batch.to(device), labels.to(device)
        # forward pass
        log_probs = model.forward(batch)
        # Calculate class labels
        probs = torch.exp(log_probs)
        top_p, top_class = probs.topk(1, dim=1)
        
        y_pred = np.append(y_pred, cuda_to_numpy(top_class))
        y_true = np.append(y_true, cuda_to_numpy(labels))
        y_probs = np.append(y_probs, cuda_to_numpy(top_p))

    return (y_true, y_pred, y_probs)