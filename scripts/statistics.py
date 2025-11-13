import argparse

from hyperparameters import stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help="Path to file to analyse")
    parser.add_argument("-d", action="store_true", help="Trigger dataset analysis")
    args = parser.parse_args()

    if args.d:
        print(str(stats.DatasetInfo(args.file)))