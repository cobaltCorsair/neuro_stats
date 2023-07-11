import cv2
from torchvision import transforms, models
import torch

# Путь к вашему изображению
image_path = '../test.png'

# Загрузить изображение и преобразовать его в оттенки серого
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Преобразовать изображение в тензор и нормализовать его
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

image = transform(image)

# Добавить дополнительное измерение
image = image.unsqueeze(0)  # shape should be (1, 1, H, W)

# Загрузите предварительно обученную модель
model = models.resnet50(pretrained=True)

# Установите модель в режим предсказания
model.eval()

# Пропустите изображение через модель
output = model(image)

# Получите наиболее вероятный класс
_, predicted = torch.max(output, 1)
print('Predicted class:', predicted.item())
