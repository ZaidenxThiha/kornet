from pathlib import Path


def main():
    root = Path("data/mvtec").resolve()
    root.mkdir(parents=True, exist_ok=True)
    print("MVTec AD may be downloaded from its official dataset page for research use.")
    print(f"Extract categories directly into: {root}")
    print("Expected example: data/mvtec/bottle/train/good/*.png")
    print("No automatic download is performed; review and accept the dataset's license yourself.")


if __name__ == "__main__":
    main()
