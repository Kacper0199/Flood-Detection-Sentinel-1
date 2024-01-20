import cv2
import numpy as np
from torch.utils.data import Dataset


def load_and_normalize_grayscale_image(img_path):
    return cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)/255


class S1Dataset(Dataset):
    def __init__(self, dataset, flood_label, transform=None):
        self.dataset = dataset
        self.flood_label = flood_label
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset.iloc[index]
        output = {}

        vv_channel = load_and_normalize_grayscale_image(sample["vv_channel"])
        vh_channel = load_and_normalize_grayscale_image(sample["vh_channel"])

        ratio = np.clip(np.nan_to_num(vv_channel/vh_channel, 0), 0, 1)
        rgb = np.stack((vv_channel, vh_channel, 1-ratio), axis=2)

        if not self.flood_label:
            output["image"] = rgb.transpose((2, 0, 1)).astype("float32")
            return output

        flood_mask = load_and_normalize_grayscale_image(sample["flood_label"])
        if self.transform:
            rgb, flood_mask = self.transform(rgb, flood_mask)

        output["image"] = rgb.transpose((2, 0, 1)).astype("float32")
        output["mask"] = flood_mask.astype("int64")

        return output
