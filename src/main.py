import argparse
import torch
import urllib

from inference import load_image, load_folder_images, build_preprocess, resolve_device, load_model, load_classes, predict_image



def main():
    parser = argparse.ArgumentParser(description="Image classification with ResNet-18")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to the input image")
    group.add_argument("--folder", type=str, help="Path to a folder containing images")

    parser.add_argument("--topk", type=int, default=5, help="Number of top predictions to return")
    parser.add_argument("--device", type=str, default="auto", 
                        choices=["cpu", "cuda", "auto"], help="Device to use (cpu, cuda, or auto)")

    args = parser.parse_args()

    try:
        device = resolve_device(args.device)
        model = load_model(device)
        preprocess = build_preprocess()
        classes = load_classes()
        if args.image:
            results = predict_image(args.image, model, preprocess, classes, device, topk=args.topk)
            for label, prob in results:
                print(f"{label}: {prob*100:.4f}%")

        elif args.folder:
            print(f"Loading images from folder: {args.folder}")
            images = load_folder_images(args.folder)
            results = []
            for filename, img in images:
                print(f"Processing {filename}...")
                img_results = predict_image(args.folder + "/" + filename, model, preprocess, classes, device, topk=args.topk)
                results.append((filename, img_results))

            for filename, img_results in results:
                print(f"\nResults for {filename}:")
                for label, prob in img_results:
                    print(f"{label}: {prob*100:.4f}%")


    except Exception as e:
        print(e)
        return

    

if __name__ == "__main__":
    main()
