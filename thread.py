import torch
from torchvision.datasets import MNIST
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from matplotlib import pyplot as plt
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

"""
질문은 여기에 남겨주시죠
1. MNIST사이즈 뭐임? 1 * 28 * 28
ㅇㅋㅇㅋ 마지막 인코딩하는 차원을 파라미터로 받는다는거지????????????????????????????????"""


class AutoEncoder(nn.Module):
    def __init__(self, latent_dimension):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1 * 28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dimension)  # 10984378943120814870913에 의미 없음
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dimension, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1 * 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        out = self.decoder(x)
        assert out.shape[2] + out.shape[3] == 28 + 28
        return out


def train(model: nn.Module, train_epoch: int, batch_size: int) -> None:
    train_dataloader = DataLoader(
        MNIST("./data", train=True, transform=torchvision.transforms.ToTensor(), download=True),
        batch_size=batch_size,
        shuffle=True
    )

    optimizer = torch.optim.AdamW(model.parameters())
    criterion = torch.nn.MSELoss()

    model.train()
    criterion.to(DEVICE)
    model.to(DEVICE)

    loss_history = []
    for epoch in range(train_epoch):
        epoch_loss = 0
        with tqdm(train_dataloader) as t:
            for data in t:
                img, _, = data.to(DEVICE)
                prediction = model(img)
                loss = criterion(img, prediction)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                t.set_postfix(epoch=f"{epoch + 1} of {train_epoch}", loss=loss.item())
                epoch_loss += loss.item()
        epoch_loss = epoch_loss / len(train_dataloader)
        loss_history.append(epoch_loss)
    plt.figure(figsize=(16, 9))
    plt.title("Loss History")
    plt.plot(*enumerate(loss_history))
    plt.xlabel("epoch")
    plt.show()

    with open("./model.pt", 'wb') as f:
        torch.save(model, f)
        print("Model saved at ./model.pt")


def evaluate(model):
    if os.path.exists("./model.pt"):
        with open("./model.pt", 'rb') as f:
            model = torch.load(f)

    test_dataloader = DataLoader(
        MNIST("./data", train=False, transform=torchvision.transforms.ToTensor(), download=True),
        batch_size=10000,
        shuffle=True
    )


    # TODO -> 구현하기, Accuracy 구하기
    with torch.no_grad():  # <- 중요 1   : 이 작업은 모델 학습에 영향을 주지 않게 하려고 선언함 + 역전파 계산 안함 (gradient tracing 비활성화)
        model.eval()  # <- 중요 2  :모델을 eval 모드로 바꿈 (dropout, batch_norm) 같은거 미적용 dropout

        criterion = torch.nn.MSELoss()
        criterion.to(DEVICE)
        total_loss = 0
        total_accuracy = 0
        with tqdm(test_dataloader) as t:
            for data in t:
                img, _, = data.to(DEVICE)
                prediction = model(img)

                loss = criterion(img, prediction) # Train 부분 보시길
                total_loss += loss.item()
                # total_accuracy += prediction.eq(img).soft()/len(prediction)

                pass
            total_loss = total_loss/len(test_dataloader)
            total_accuracy = 1-total_loss


def main():
    model = AutoEncoder(16)
    # TODO -> 파라미터 채우기
    train(model, 15, 64)
    evaluate(model)


if __name__ == '__main__':
    main()
