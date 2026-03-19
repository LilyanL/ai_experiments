import argparse
import torch
import urllib

from inference import load_model, build_preprocess, predict_image



def main():
    parser = argparse.ArgumentParser(description="Image classification with ResNet-18")
    parser.add_argument("--image", type=str, required = True, help="Path to the input image")
    parser.add_argument("--topk", type=int, default=5, help="Number of top predictions to return")

    args = parser.parse_args()

    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        if (major, minor) >= (7, 0):
            device = torch.device("cuda")
        else:
            print(
                f"GPU detected (compute capability {major}.{minor}) "
                "but not supported by the installed PyTorch build. Falling back to CPU."
            )
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    model = load_model(device)
    preprocess = build_preprocess()

    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    classes = urllib.request.urlopen(url).read().decode("utf-8").splitlines()

    results = predict_image(args.image, model, preprocess, classes, device)

    for label, prob in results:
        print(f"{label}: {prob*100:.4f}%")

if __name__ == "__main__":
    main()
