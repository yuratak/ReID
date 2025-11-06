import torch
from torch.utils.data import Dataset, DataLoader


class sequence_data(Dataset):
    def __init__(self, input_timeseries_ , input_features_, label_, len):
        self.input_timeseries_ = input_timeseries_ 
        self.input_features_ = input_features_
        self.label_ = label_
        self.len = len
        self.data_size = input_timeseries_.size(0)
        return

    def __getitem__(self, index):
        return self.input_timeseries_[index], self.input_features_[index], self.label_[index], self.len[index]

    def __len__(self):
        return self.data_size
    

class DataManager(object):
    def __init__(self, train_data, test_data, batch_size):
        train_data_timeseries, test_data_timeseries = train_data["timeseries"], test_data["timeseries"]
        train_data_features, test_data_features = train_data["features"], test_data["features"]
        train_label, test_label = train_data["label"], test_data["label"]

        self.train_input_timeseries = torch.FloatTensor(train_data_timeseries.astype(float))
        self.train_input_features = torch.FloatTensor(train_data_features.astype(float))
        self.train_label = torch.FloatTensor(train_label.astype(float).reshape(-1,1,1))

        self.train_len = self.train_input_timeseries[:, :, 0] != -1
        self.train_len = self.train_len.sum(axis=1)

        self.test_input_timeseries = torch.FloatTensor(test_data_timeseries.astype(float))
        self.test_input_features = torch.FloatTensor(test_data_features.astype(float))
        self.test_label = torch.FloatTensor(test_label.astype(float).reshape(-1,1,1))

        self.test_len = self.test_input_timeseries[:, :, 0] != -1
        self.test_len = self.test_len.sum(axis=1)

        self.train_data = sequence_data(self.train_input_timeseries, 
                                        self.train_input_features, 
                                        self.train_label,
                                        self.train_len)
        self.test_data = sequence_data(self.test_input_timeseries, 
                                       self.test_input_features, 
                                       self.test_label, 
                                       self.test_len)

        self.train_loader = DataLoader(dataset=self.train_data,
                                       batch_size=batch_size,
                                       shuffle=True)
        self.test_loader = DataLoader(dataset=self.test_data,
                                      batch_size=batch_size,
                                      shuffle=False)

        return