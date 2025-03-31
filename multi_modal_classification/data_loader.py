import torch
from torch.utils.data import Dataset
import pandas as pd

class MultiModalEmotionDataset(Dataset):
    def __init__(self, dataframe, prob_columns, label_column):
        """
        Args:
            dataframe (pd.DataFrame): Pandas DataFrame containing the probability distributions and labels.
            prob_column (str): Name of the column containing the nested probabilities of each class.
            label_column (str): Name of the column containing the labels.
        """
        self.data = dataframe
        self.prob_columns = prob_columns
        self.label_column = label_column

        self.label_mapping = {
            'angry': 0,
            'disgust': 1,
            'fearful': 2,
            'happy': 3,
            'neutral': 4,
            'sad': 5
        }
        # label_map_crema = {
        #     'angry': 'ANG',
        #     'disgust': 'DIS',
        #     'fearful': 'FEA',
        #     'happy': 'HAP',
        #     'neutral': 'NEU',
        #     'sad': 'SAD'
        # }
        print(f"Total rows in dataset: {len(self.data)}")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Concatenate the probabilities from each modality
        multi_mod_probs = []
        for modality_probs in self.prob_columns:
            self.data[modality_probs] = self.data[modality_probs]
            multi_mod_probs.extend(self.data.iloc[idx][modality_probs])

        x = torch.tensor(multi_mod_probs, dtype=torch.float32)
        y = torch.tensor(self.label_mapping[self.data.iloc[idx][self.label_column]], dtype=torch.long)
        original_index = self.data['original_index'].iloc[idx]
        
        return x, y, original_index
