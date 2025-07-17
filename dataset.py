import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import cv2

class FacialKeypointsDataset(Dataset):
    """Набор данных с ключевыми точками лица."""

    def __init__(self, csv_file, train=True, transform=None):
        """
        Аргументы:
            csv_file (string): Путь к csv-файлу с аннотациями.
            transform (callable, optional): Необязательное преобразование, которое будет применяться
                к сэмплу.
        """
        self.keypoints_frame = pd.read_csv(csv_file)
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.keypoints_frame)

    def __getitem__(self, idx):
        row = self.keypoints_frame.iloc[idx]
        
        image = np.fromstring(self.keypoints_frame.iloc[idx, 30], sep=' ').reshape(96, 96)
        
        # Выбираем только 4 ключевые точки с низким количеством пропущенных значений
        keypoints = self.keypoints_frame.iloc[idx, [0, 1, 2, 3, 20, 21, 28, 29]].values.astype('float').reshape(-1, 2)
        
        if self.train:
            # Аугментация данных
            if np.random.rand() > 0.5:
                # Горизонтальное отражение
                image = cv2.flip(image, 1)
                keypoints[:, 0] = 96 - keypoints[:, 0]
                
            if np.random.rand() > 0.5:
                # Поворот
                angle = np.random.uniform(-10, 10)
                M = cv2.getRotationMatrix2D((48, 48), angle, 1)
                image = cv2.warpAffine(image, M, (96, 96))
                keypoints = np.dot(keypoints - 48, np.linalg.inv(M[:, :2])) + 48
                
        sample = {'image': image, 'keypoints': keypoints}

        if self.transform:
            sample = self.transform(sample)

        return sample
