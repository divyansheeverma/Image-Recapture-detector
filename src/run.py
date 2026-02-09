import argparse
from detector import is_image_of_image_plus

def main():
    parser = argparse.ArgumentParser(description="Photo-of-Photo Detection")
    parser.add_argument("--image", required=True, help="Path to image")
    parser.add_argument("--debug", action="store_true", help="Show debug windows")
    parser.add_argument("--debug_ms", type=int, default=1200)
    args = parser.parse_args()

    result, info = is_image_of_image_plus(
        args.image,
        debug_show=args.debug,
        debug_ms=args.debug_ms
    )

    print("\n RESULT")
    print("Image-of-Image:", result)
    print("Reason:", info.get("decision_reason"))

if __name__ == "__main__":
    main()
