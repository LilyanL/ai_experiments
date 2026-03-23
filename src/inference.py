import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
import urllib.request

IMAGENET_CLASSES_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"

def load_image(path):
    img = cv2.imread(path)

    if img is None:
        raise ValueError(f"Error: image not found: {path}")

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

def resolve_device(device_arg):
    if device_arg == "cpu":
        return torch.device("cpu")

    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        major, minor = torch.cuda.get_device_capability()
        if (major, minor) < (7, 0):
            raise RuntimeError(
                f"CUDA requested but GPU compute capability {major}.{minor} "
                "is not supported by this PyTorch build."
            )
        return torch.device("cuda")

    # auto
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        if (major, minor) >= (7, 0):
            return torch.device("cuda")
        print(
            f"GPU detected (compute capability {major}.{minor}) "
            "but not supported by the installed PyTorch build. Falling back to CPU."
        )
    return torch.device("cpu")

def load_model(device):
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.eval()
    model.to(device)
    return model

def load_classes():
    try:
        classes = urllib.request.urlopen(IMAGENET_CLASSES_URL).read().decode("utf-8").splitlines()
    except Exception as e:
        raise RuntimeError("Failed to load ImageNet classes") from e
    return classes

def predict_image(path, model, preprocess, classes, device, topk=5):

    # 1) Load image
    img = load_image(path)

    # 2) Preprocess image & 3) Add batch
    input_batch = preprocess(img).unsqueeze(0).to(device)
    
    # 4) Move to device
    input_batch = input_batch.to(device)

    # 5) Predict
    with torch.no_grad():
        output = model(input_batch)

    # 6) soft max
    probs = F.softmax(output, dim=1)

    # 7) top-k
    topk_prob, topk_idx = torch.topk(probs, topk, dim=1)

    # 8) map results to class names and return
    results = []
    for i in range(topk):
        label = classes[topk_idx[0][i]]
        prob = topk_prob[0][i].item()
        results.append((label, prob))

    return results