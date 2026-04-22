import argparse
import os
import subprocess

THICK_SPACER = "=" * 50
THIN_SPACER = "-" * 50
TMP_PATH = "/tmp/hyph-bench"

def implement_fixes(wordlist: str, fixes: str):
    subprocess.call(f"python -m scripts.replace_in_wordlist --wordlist {wordlist} --fixed {fixes} --output-dir {TMP_PATH}", shell=True)
    subprocess.call(f"mv -f fixed.wlh {wordlist}", shell=True)

def compare_fixes(paths: list[str], result_dir: str):
    if len(paths) == 1:
        print("Only one file provided, no comparision will be done")

    print("Comparing between provided files")
    args = " ".join(paths)
    result_path = os.path.join(result_dir, "comparision.csv")
    subprocess.call(f"python -m scripts.compare_annotations {args} > {result_path}", shell=True)
    print(f"Comparision written to {result_path}")

def run_optimizer(params: str):
    print("\nRunning optimizer")
    print(THIN_SPACER)
    subprocess.call("python -m scripts.optimize -b " + params, shell=True)

def get_fix_files():
    print("\nOptimizer output a file to its results dir with words which were incorrectly hyphenated")
    print(THIN_SPACER)
    
    response = input("Please re-hyphenate the words and input paths to the results or press enter to stop:\n")
    paths = response.strip().split()

    if paths == []:
        return [], -1

    print("Following files provided:")
    for i in range(len(paths)):
        print(f"{i + 1}. {paths[i]}")

    path_i_str = input("Which of the files will be used to update the wordlist?\n")
    path_i = int(path_i_str) - 1

    print(f"Path {paths[path_i]} will be used")
    return paths, path_i

def main():
    parser = argparse.ArgumentParser(
        description="Iterate over provided wordlist for n iternations or until convergence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--input", required=True, type=str,
                        help="Hyphenated input wordlist (one word per line)")
    parser.add_argument("--output-dir", required=False, type=str, default="iter_results",
                        help="Output directory path")
    parser.add_argument("--optimizer-params", required=False, type=str, default="--objective bounded_bad --lang uk",
                        help="Params which will be passed to the optimizer; see python -m scripts.optimize --help\n" \
                             "They must point to the same wordlist as the input param, the results folder should be left to default\n" \
                             "Default params: --objective bounded_bad --lang uk")

    args = parser.parse_args()
    output = args.output_dir
    params = args.optimizer_params
    wordlist = os.path.join(TMP_PATH, args.input)
    iter_folders = os.path.join(output, "iter*")

    os.makedirs(args.output_dir, exist_ok=True)
    subprocess.call(f"cp {args.input} {wordlist}", shell=True)
    subprocess.call(f"rm -rf {iter_folders}", shell=True)

    print(THICK_SPACER)
    print(f"Iterating on {args.input}")
    print(f"Results will be stored to {output}")
    print("Once there are few enough words in the bad.txt file output by the optimizer, we can stop")
    print(THICK_SPACER + "\n")

    n = 1
    while (True):
        print(f"ITERATION {n}")
        print(THIN_SPACER)

        iteration_output = os.path.join(output, f"iter{n}")
        os.makedirs(iteration_output)
        run_optimizer(params)

        print(THIN_SPACER)
        print("Incorrectly hyphenated words found:")
        subprocess.call(f"wc -l results/bad.txt", shell=True)
        print(THIN_SPACER)
        
        subprocess.call(f"cp results/bad.txt {iteration_output}", shell=True)

        files, fix_i = get_fix_files()

        if not files:
            break

        compare_fixes(files, iteration_output)
        implement_fixes(wordlist, files[fix_i])

        n += 1

    print(THICK_SPACER)
    print(f"Final iterations {n}")
    print(f"Storing result to {output}")
    print(THICK_SPACER)

    subprocess.call(f"mv {wordlist} {output}", shell=True)


if __name__ == "__main__":
    main()