import torch

from torchvision import transforms, models

from PIL import Image
classes = [
    "earthquake_major",
    "earthquake_minor",
    "flood_high",
    "flood_low",
    "flood_medium"
]
model = models.mobilenet_v2()

model.classifier[1] = torch.nn.Linear(
    1280,
    5
)

model.load_state_dict(
    torch.load("models/mobilenet.pth")
)

model.eval()
image = Image.open("test.jpg")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

image = transform(image)

image = image.unsqueeze(0)
with torch.no_grad():

    outputs = model(image)

    predicted = torch.argmax(outputs)

print(
    "Prediction:",
    classes[predicted]
)