import argparse
import os


def main():
    parser = argparse.ArgumentParser(
        description="Replaces all in a provided wordlist with the fixed hyphenations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument("--wordlist", required=True, type=str,
                        help="Wordlist file you want to hyphenate")
    
    parser.add_argument("--fixed", required=True, type=str,
                        help="Fixes in the format {word}={fixed}")
    
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for results (default: results)')
    
    args = parser.parse_args()
    wordlist = args.wordlist
    fixed_file = args.fixed

    os.makedirs(args.output_dir, exist_ok=True)
    wordlist_name = os.path.basename(wordlist).split(".")[0]
    output_path = os.path.join(args.output_dir, f"fixed.wlh")

    fixed_dict = {}
    with open(fixed_file, "r", encoding="utf-8") as fixed:
        for line in iter(lambda: fixed.readline(), ''):
            entry = line.split('=')
            fixed_dict[entry[0]] = entry[1]

    with open(wordlist, "r", encoding="utf-8") as to_fix, \
         open(output_path, "w", encoding="utf-8") as output:
        for word in iter(lambda: to_fix.readline(), ''):
            no_hyphens = word.strip().replace('-', '')

            replacement = fixed_dict.get(no_hyphens)
            output.write(replacement if replacement is not None else word)

if __name__ == "__main__":
    main()