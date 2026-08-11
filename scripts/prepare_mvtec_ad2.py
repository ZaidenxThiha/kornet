from pathlib import Path


def main():
    root = Path("data/mvtec_ad2").resolve()
    root.mkdir(parents=True, exist_ok=True)
    print("Request/download MVTec AD 2 from its official page after accepting its terms.")
    print(f"Extract category folders into: {root}")
    print("Private-test labels must remain private and must never be used for optimization.")


if __name__ == "__main__":
    main()
