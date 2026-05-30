import pandas as pd


def main():
    df = pd.read_csv("results/evaluation_sheet.csv")

    grouped = df.groupby("method").mean(numeric_only=True)

    print("\n=== Answer-Level Evaluation ===")
    print(grouped)


if __name__ == "__main__":
    main()