from pathlib import Path


def main():
    root = Path("data/visa").resolve()
    root.mkdir(parents=True, exist_ok=True)
    print("Obtain VisA from its official release and review its terms.")
    print(f"Extract the release into: {root}")
    print("Keep split_csv/1cls.csv and the category folders intact.")


if __name__ == "__main__":
    main()
