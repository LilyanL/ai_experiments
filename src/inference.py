import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def build_preprocess():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def load_model(device):
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.eval()
    model.to(device)
    return model


def predict_image(path, model, preprocess, classes, device):

    # 1) Load image
    img = load_image(path)

    # 2) Preprocess image
    input_batch = preprocess(img).unsqueeze(0).to(device)

    # 3) Add batch
    #input_batch = input_tensor.unsqueeze(0)

    # 4) Move to device
    input_batch = input_batch.to(device)

    # 5) Predict
    with torch.no_grad():
        output = model(input_batch)

    # 6) soft max
    probs = F.softmax(output, dim=1)

    # 7) top-5
    top5_prob, top5_idx = torch.topk(probs, 5, dim=1)

    # 8) map results to class names and return
    results = []
    for i in range(5):
        label = classes[top5_idx[0][i]]
        prob = top5_prob[0][i].item()
        results.append((label, prob))

    return results