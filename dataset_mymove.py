from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import pickle
def parse_data(data: pd.DataFrame, missing_ratio = 0.1):
        data = data.to_numpy()
        # randomly set some percentrage as ground-truth
        observed_values = data.to_numpy()
        observed_masks = np.full(observed_values.shape, True)
        masks = observed_masks.reshape(-1).copy()
        obs_indices = np.where(masks)[0].tolist()
        miss_indices = np.random.choice(
        obs_indices, (int)(len(obs_indices) * missing_ratio), replace=False
        )
        masks[miss_indices] = False
        gt_masks = masks.reshape(observed_masks.shape)

        observed_values = np.nan_to_num(observed_values)
        observed_masks = observed_masks.astype("float32")
        gt_masks = gt_masks.astype("float32")

        return observed_values, observed_masks, gt_masks

class MymoveDataset(Dataset):
    def __init__(self, input_path, eval_length=48, use_index_list=None, missing_ratio=0.0, seed=0):
        self.eval_length = eval_length
        np.random.seed(seed)  # seed for ground truth choice

        self.observed_values = []
        self.observed_masks = []
        self.gt_masks = []
        self.save_path = (
            "./data/mymove" + str(missing_ratio) + "_seed" + str(seed) + ".pk"
        )
        dataset = pd.read_csv(input_path)
        self.missing_ratio = missing_ratio

        # normalization
        for col in dataset.columns:
            dataset[col] = (dataset[col] - dataset[col].mean()) / dataset[col].std()
        self.dataset = dataset
        
        observed_values, observed_masks, gt_masks =parse_data(self.dataset, self.missing_ratio)
        # self.observed_values = np.array_split(observed_values, len(observed_values)//120)
        # self.observed_masks = np.array_split(observed_masks, len(observed_masks)//120)
        # self.gt_masks = np.array_split(gt_masks, len(gt_masks)//120)
        self.observed_values = observed_values
        self.observed_masks = observed_masks
        self.gt_masks = gt_masks
        with open(self.save_path, "wb") as f:
                pickle.dump(
                    [self.observed_values, self.observed_masks, self.gt_masks], f
                )
    def __getitem__(self, index) :
          return {"observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.gt_masks[index],
            "timepoints": np.arange(self.eval_length)}
    
    def __len__(self):
        return len(self.dataset)
# check this

def get_dataloader(seed=1, nfold=None, batch_size=16, missing_ratio=0.1):

    # only to obtain total length of dataset
    dataset = MymoveDataset(missing_ratio=missing_ratio, seed=seed)
    indlist = np.arange(len(dataset))

    np.random.seed(seed)
    np.random.shuffle(indlist)

    # 5-fold test
    start = (int)(nfold * 0.2 * len(dataset))
    end = (int)((nfold + 1) * 0.2 * len(dataset))
    test_index = indlist[start:end]
    remain_index = np.delete(indlist, np.arange(start, end))

    np.random.seed(seed)
    np.random.shuffle(remain_index)
    num_train = (int)(len(dataset) * 0.7)
    train_index = remain_index[:num_train]
    valid_index = remain_index[num_train:]

    dataset = MymoveDataset(
        use_index_list=train_index, missing_ratio=missing_ratio, seed=seed
    )
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=1)
    valid_dataset = MymoveDataset(
        use_index_list=valid_index, missing_ratio=missing_ratio, seed=seed
    )
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=0)
    test_dataset = MymoveDataset(
        use_index_list=test_index, missing_ratio=missing_ratio, seed=seed
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=0)
    return train_loader, valid_loader, test_loader
    

