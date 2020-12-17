from torch.utils.data import TensorDataset, DataLoader

def mean_std(train_loader):
    """Useful tool to compute the mean and the standard deviation of the train loader. This is used either to check if the data
    loader is normalized, or to compute the mean and std for the normalizer in data_loader.

    Args:
        train_loader (Dataloader): Pytorch dataloader
    """
    nimages = 0
    mean = 0.
    std = 0.
    for batch, _ in train_loader:
        # Rearrange batch to be the shape of [B, C, W * H]
        batch = batch.view(batch.size(0), batch.size(1), -1)
        # Update total number of images
        nimages += batch.size(0)
        # Compute mean and std here
        mean += batch.mean(2).sum(0) 
        std += batch.std(2).sum(0)

    # Final step
    mean /= nimages
    std /= nimages

    print(mean)
    print(std)
