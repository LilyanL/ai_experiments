import argparse
import torch
import urllib

from inference import load_classes, load_model, build_preprocess, predict_image, resolve_device



def main():
    parser = argparse.ArgumentParser(description="Image classification with ResNet-18")
    parser.add_argument("--image", type=str, required = True, help="Path to the input image")
    parser.add_argument("--topk", type=int, default=5, help="Number of top predictions to return")
    parser.add_argument("--device", type=str, default="auto", 
                        choices=["cpu", "cuda", "auto"], help="Device to use (cpu, cuda, or auto)")

    args = parser.parse_args()

    try:
        device = resolve_device(args.device)
        model = load_model(device)
        preprocess = build_preprocess()
        results = predict_image(args.image, model, preprocess, classes, device, topk=args.topk)
    except Exception as e:
        print(e)
        return

    classes = load_classes()

    try:
        results = predict_image(args.image, model, preprocess, classes, device, args.topk)
    except Exception as e:
        print(e)
        return

    for label, prob in results:
        print(f"{label}: {prob*100:.4f}%")

if __name__ == "__main__":
    main()
