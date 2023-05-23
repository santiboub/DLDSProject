import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

from senet import SENet34

def create_image_grid(images, predictions, classes):
    fig, axs = plt.subplots(4, 4, figsize=(12, 12))
    axs = axs.flatten()

    for i in range(len(images)):
        image = images[i]
        ground_truth = image["label"]
        prediction = predictions[i]
        is_correct = prediction == ground_truth

        ground_truth_label = classes[ground_truth]
        prediction_label = classes[prediction]

        border_color = "green" if is_correct else "red"
        border_width = 5

        axs[i].imshow(image["image"])
        axs[i].set_title(ground_truth_label if is_correct else f"{prediction_label}, gt={ground_truth_label}")
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].spines["left"].set_color(border_color)
        axs[i].spines["left"].set_linewidth(border_width)
        axs[i].spines["right"].set_color(border_color)
        axs[i].spines["right"].set_linewidth(border_width)
        axs[i].spines["top"].set_color(border_color)
        axs[i].spines["top"].set_linewidth(border_width)
        axs[i].spines["bottom"].set_color(border_color)
        axs[i].spines["bottom"].set_linewidth(border_width)

    plt.tight_layout()
    plt.show()

def predict_images(model, testloader, classes):
    images = []
    predictions = []

    with torch.no_grad():
        train_features, train_labels = next(iter(testloader))
        outputs = model(train_features)

        _, predicted = torch.max(outputs.data, 1)

        for i, (image, label) in enumerate(zip(train_features, train_labels)):
            #print(f"Predicted: {classes[predicted[i]]}, Ground Truth: {classes[label]}")

            images.append({"image": image.permute(1, 2, 0), "label": label})
            predictions.append(predicted[i].item())

        create_image_grid(images, predictions, classes)


def main():
    CIFAR100 = False
    if CIFAR100:
        DATASET = torchvision.datasets.CIFAR100
        MODEL_PTH = "saved_models/20230522-194537_best_model_cifar100/100/baseline_model.pth"
    else:
        DATASET = torchvision.datasets.CIFAR10
        MODEL_PTH = "saved_models/20230522-080404_best_model_cifar10/100/baseline_model.pth"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((64, 64)),
    ])

    testset = DATASET(root="./data", train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=True, num_workers=2)

    classes = testset.classes

    model = SENet34(num_classes=len(classes), small_kernel=True)
    model.load_state_dict(torch.load(MODEL_PTH))
    model.eval()

    while True:
        predict_images(model, testloader, classes)


if __name__ == "__main__":
    main()
