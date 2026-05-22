import argparse
import os
import pyphen

"""
Simple script for hyphenating word list using patgen

Can be used for bootstrapping patterns created from smaller datasets onto new
ones from larger one

Each word that should be hyphenated should be on separate line

Usage:
    python -m scripts.hyphenate_wordlist --dict uk.dict --pat uk.pat --trans uk.trans --lr-hypen-min 2 2
"""

def main():
    parser = argparse.ArgumentParser(
        description="Hyphenate new wordlist with already existing patterns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument("--dict", required=True, type=str,
                        help="Wordlist file you want to hyphenate")
    
    parser.add_argument("--separator", type=str, default="-",
                        help="Provide alternative separator character")
    
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for results (default: results)')
    
    args = parser.parse_args()

    dict_file = args.dict

    os.makedirs(args.output_dir, exist_ok=True)
    dict_file_name = os.path.basename(dict_file)
    output_path = os.path.join(args.output_dir, f"{dict_file_name}.hyph")

    hyphenator = pyphen.Pyphen(lang="uk")

    dictionary = open(dict_file, "r")
    output = open(output_path, "w+")

    for line in iter(lambda: dictionary.readline(), ''):
        hypenated = hyphenator.inserted(line.strip(), args.separator)
        output.write(f"{hypenated}\n")

    dictionary.close()
    output.close()


if __name__ == '__main__':
    main()