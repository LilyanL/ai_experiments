import argparse
import torch
import urllib

from inference import load_model, build_preprocess, predict_image



def main():
    parser = argparse.ArgumentParser(description="Image classification with ResNet-18")
    parser.add_argument("--image", type=str, required = True, help="Path to the input image")
    parser.add_argument("--topk", type=int, default=5, help="Number of top predictions to return")
    parser.add_argument("--device", type=str, default="auto", 
                        choices=["cpu", "cuda", "auto"], help="Device to use (cpu, cuda, or auto)")

    args = parser.parse_args()

    if args.device == "cpu":
        device = torch.device("cpu")

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        major, minor = torch.cuda.get_device_capability()
        if (major, minor) < (7, 0):
            raise RuntimeError(
                f"CUDA requested but GPU compute capability {major}.{minor} "
                "is not supported by this PyTorch build."
            )
        device = torch.device("cuda")
            
    else: # auto
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

    try:
        results = predict_image(args.image, model, preprocess, classes, device, args.topk)
    except Exception as e:
        print(e)
        return

    for label, prob in results:
        print(f"{label}: {prob*100:.4f}%")

if __name__ == "__main__":
    main()
