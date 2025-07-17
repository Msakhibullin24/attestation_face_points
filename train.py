import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import FacialKeypointsDataset
from model import KeypointModel
from torch.optim.lr_scheduler import StepLR

# Гиперпараметры
EPOCHS = 200
BATCH_SIZE = 64
LEARNING_RATE = 0.0001

def train():
    # Создаем набор данных и загрузчик данных
    dataset = FacialKeypointsDataset(csv_file='training.csv', train=True)
    dataset.keypoints_frame.dropna(inplace=True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # Инициализируем модель, функцию потерь и оптимизатор
    model = KeypointModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    # Цикл обучения
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for i, data in enumerate(dataloader):
            # Получаем входные данные
            images = data['image']
            keypoints = data['keypoints']

            # Обнуляем градиенты параметров
            optimizer.zero_grad()

            # Прямой проход + обратный проход + оптимизация
            outputs = model(images.float())
            loss = criterion(outputs, keypoints.view(keypoints.size(0), -1).float())
            loss.backward()
            optimizer.step()

            # Печатаем статистику
            running_loss += loss.item()
            if i % 10 == 9:    # Печатаем каждые 10 мини-батчей
                print(f'Эпоха: {epoch + 1}, Батч: {i + 1}, Потери: {running_loss / 10:.3f}')
                running_loss = 0.0
        scheduler.step()

    print('Обучение завершено')

    # Сохраняем обученную модель
    torch.save(model.state_dict(), 'keypoint_model.pth')

if __name__ == '__main__':
    train()
