import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import pandas as pd
import os
from PIL import Image

MODEL_NAME = 'project2.pth'

class CustomDataset(Dataset):
    def __init__(self, path, transform=None):
        self.data_frame = None
        self.get_data_frame(path)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_path, label = self.data_frame.iloc[idx]
        image = self.__get_image(img_path)
        return image, label

    def __get_image(self, file_path, target_size=(224, 224)):
        pil_image = Image.fromarray(np.uint8(pd.read_fwf(file_path).to_numpy())).convert('RGB')

        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
        ])
        image = transform(pil_image)

        return image

    def get_data_frame(self, path):
        if path.endswith('.txt'):
            df = pd.read_csv(path, sep=' ', header=None, names=['filenames', 'labels'])
        else:
            df = pd.DataFrame()
            filenames = []
            labels = []
            image_class = 0

            for directory in os.listdir(path):
                if not directory.startswith("."):
                    images = os.listdir(os.path.join(path, directory))

                    for image in images:
                        filenames.append(os.path.join(path,directory, image))
                        labels.append(image_class)

                    image_class = 1

            df["filenames"] = filenames
            df["labels"] = labels

        self.data_frame = df

class linLayer(nn.Module):
    def __init__(self, inFeatures, outFeatures, bias=True):
        super().__init__()
        self.inFeatures = inFeatures
        self.outFeatures = outFeatures
        self.bias = bias
        self.weights = torch.nn.Parameter(torch.randn(outFeatures, inFeatures))
        if bias:
            self.bias = torch.nn.Parameter(torch.rand(outFeatures))
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        r, c = input.shape
        if c != self.inFeatures:
            sys.exit(f'Dimensions do not match. Input must have {self.inFeatures} columns.')
        output = input @ self.weights.t() + self.bias
        return output

class convLayer(nn.Module):
    def __init__(self, inFeatures, outFet Accuracy: {accuracy}')

if __name__ == "__main__":
    resolution = 224
    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    first_set= ImageFolder(root='./data/training', transform=transform)
    training_set, eval_set = torch.utils.data.random_split(first_set, [2412, 270])

    def custom_collate_fn(batch):
        images, labels = zip(*batch)
        return torch.stack(images), torch.tensor(labels)

    training_loader = DataLoader(training_set, batch_size=20, shuffle=True, collate_fn=custom_collate_fn)
    eval_loader=DataLoader(eval_set,batch_size=1, shuffle=True, collate_fn=custom_collate_fn)
    testing_set = CustomDataset("./data/test", transform=transform)
    testing_loader = DataLoader(testing_set, batch_size=1, shuffle=False)

    model = create_model(resolution, load_previous_model=False)
    train(model, training_loader,eval_loader, epochs=5, save=True)
    model_test(model,testing_loader)
atures, kernelSize, bias=True, activation='ReLU'):
        super(convLayer, self).__init__()
        self.conv = nn.Conv2d(inFeatures, outFeatures, kernelSize)

    def forward(self, x):
        x = self.conv(x)
        return x


class CustomModel(nn.Module):
    def __init__(self, resolution):
        super(CustomModel, self).__init__()
        self.conv1 = convLayer(3, 16, kernelSize=3)
        self.activation1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.2)
        self.conv2 = convLayer(16, 16, kernelSize=3)
        self.activation2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout(0.2)
        self.conv3 = convLayer(16, 64, kernelSize=3)
        self.activation3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2)
        self.dropout3 = nn.Dropout(0.2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = linLayer(64, 512)
        self.activation4 = nn.ReLU()
        self.dropout4 = nn.Dropout(0.5)
        self.fc2 = linLayer(512, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation1(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.activation2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        x = self.conv3(x)
        x = self.activation3(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.activation4(x)
        x = self.dropout4(x)
        x = self.fc2(x)
        return x

def create_model(resolution, load_previous_model=True):
    if os.path.isfile(MODEL_NAME) and load_previous_model:
        return torch.load(MODEL_NAME)
    else:
        model = CustomModel(resolution)
        return model

def train(model, training_set,eval_loader, epochs, save):
    val_accuracy_log = []
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    eval1=0
    for epoch in range(epochs):
        print(f'Start of Epoch {epoch + 1}')
        running_loss = 0.0
        for inputs, labels in training_set:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.view(-1, 1).float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(training_set)}')
        if save:
          torch.save(model, MODEL_NAME)
        eval1=model_eval(model,eval_loader)
        val_accuracy_log.append(eval1)
    plt.plot(list(range(1,epochs+1)),val_accuracy_log,'+')
    plt.legend(['eval'])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()

def model_eval(model, eval_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in eval_loader:
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy}')

def model_test(model, testing_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in testing_loader:
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Tes
