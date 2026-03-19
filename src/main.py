import argparse
import torch
import urllib

from inference import load_model, build_preprocess, predict_image



def main():
    parser = argparse.ArgumentParser(description="Image classification with ResNet-18")
    parser.add_argument("--image", type=str, required = True, help="Path to the input image")
    parser.add_argument("--topk", type=int, default=5, help="Number of top predictions to return")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    preprocess = build_preprocess()

    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    classes = urllib.request.urlopen(url).read().decode("utf-8").splitlines()

    results = predict_image(args.image, model, preprocess, classes, device)

    for label, prob in results:
        print(f"{label}: {prob*100:.4f}%")

if __name__ == "__main__":
    main()
