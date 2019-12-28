
import os
import glob
from torch.utils.data import Dataset
from PIL import Image

class TinyImagenetDataset(Dataset):
    def __init__(self, path="data", ):
