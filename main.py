import cv2
import torch
import numpy as np
from model import KeypointModel
import argparse

# Загружаем обученную модель
model = KeypointModel()
model.load_state_dict(torch.load('keypoint_model.pth'))
model.eval()

# Функция для отрисовки ключевых точек на кадре
def draw_keypoints(frame, keypoints):
    for (x, y) in keypoints:
        cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)
    return frame

# Функция для обнаружения ключевых точек
def detect_keypoints(frame, model):
    # Преобразуем кадр в оттенки серого
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Изменяем размер кадра до 96x96
    resized_frame = cv2.resize(gray, (96, 96))
    
    # Преобразуем кадр в тензор PyTorch
    tensor_frame = torch.from_numpy(resized_frame).float().unsqueeze(0)
    
    # Получаем предсказания модели
    with torch.no_grad():
        keypoints = model(tensor_frame)
    
    # Изменяем форму ключевых точек на (4, 2)
    keypoints = keypoints.view(-1, 2).numpy()
    
    # Масштабируем ключевые точки до исходного размера кадра
    keypoints = keypoints * (frame.shape[1] / 96, frame.shape[0] / 96)
    
    # Рисуем ключевые точки на исходном кадре
    output_frame = draw_keypoints(frame.copy(), keypoints)
    
    return output_frame

def main():
    # Разбираем аргументы командной строки
    parser = argparse.ArgumentParser(description='Обнаружение ключевых точек лица в реальном времени.')
    parser.add_argument('--camera', type=int, default=0, help='Индекс используемой камеры.')
    parser.add_argument('--video', type=str, help='Путь к видеофайлу.')
    parser.add_argument('--output', type=str, help='Путь к выходному видеофайлу.')
    args = parser.parse_args()

    # Открываем соединение с веб-камерой или видеофайлом
    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"Ошибка: Не удалось открыть видеофайл: {args.video}")
            return
    else:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print("Ошибка: Не удалось открыть веб-камеру.")
            return

    # Получаем свойства видео
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Определяем кодек и создаем объект VideoWriter
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps, (frame_width, frame_height))

    while True:
        # Захватываем кадр за кадром
        ret, frame = cap.read()
        if not ret:
            print("Ошибка: Не удалось получить кадр (конец потока?). Выход ...")
            break

        # Обнаруживаем ключевые точки
        output_frame = detect_keypoints(frame, model)

        # Записываем кадр в выходной файл
        if args.output:
            out.write(output_frame)

        # Отображаем результирующий кадр
        cv2.imshow('Обнаружение ключевых точек лица', output_frame)

        # Прерываем цикл по нажатию клавиши 'q'
        if cv2.waitKey(1) == ord('q'):
            break

    # Освобождаем ресурсы
    cap.release()
    if args.output:
        out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
