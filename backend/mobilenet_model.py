import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(
    'datasets/mobilenet',
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True
)

model = models.mobilenet_v2(pretrained=True)

model.classifier[1] = nn.Linear(1280, 5)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

for epoch in range(5):

    running_loss = 0

    for images, labels in loader:

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print("Epoch:", epoch,
          "Loss:", running_loss)

torch.save(
    model.state_dict(),
    'models/mobilenet.pth'
)

print("Model Saved")