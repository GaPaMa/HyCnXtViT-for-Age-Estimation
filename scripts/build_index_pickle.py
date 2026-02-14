import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True, help='CSV with columns: Image,Age,Fold')
    ap.add_argument('--out', required=True, help='Output .pkl path')
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    required = {'Image', 'Age', 'Fold'}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"CSV missing columns: {sorted(missing)}")

    df.to_pickle(args.out)
    print(f"Wrote: {args.out} ({len(df)} rows)")


if __name__ == '__main__':
    main()
