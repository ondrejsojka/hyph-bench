""" Hyphenation, using Frank Liang's algorithm.
    Adapted from: https://nedbatchelder.com/code/modules/hyphenate
    and from Ondrej Sojka thesis: https://is.muni.cz/auth/th/s17d6/
"""

import re
import sys
import regex

class Hyphenator:
    def __init__(self, pattern_file):
        self.tree = {}

        with open(pattern_file, "r") as f:
            lines = f.readlines()

        for line in lines:
            patterns = line.strip().split()
            
            for pattern in patterns:
                self._insert_pattern(pattern)

    def _insert_pattern(self, pattern):
        # Convert the a pattern like 'a1bc3d4' into a string of chars 'abcd'
        # and a list of points [ 0, 1, 0, 3, 4 ].
        chars = re.sub(r'[0-9]', '', pattern)
        points = [int(d or 0) for d in regex.split(r'\D', pattern)]

        # Insert the pattern into the tree.  Each character finds a dict
        # another level down in the tree, and leaf nodes have the list of
        # points.
        t = self.tree
        for c in chars:
            if c not in t:
                t[c] = {}
            t = t[c]
        t[None] = points

    def hyphenate_word(self, word):
        """ Given a word, returns a list of pieces, broken at the possible
            hyphenation points.
        """
        # Short words aren't hyphenated.
        if len(word) <= 4:
            return [word]
        
        work = '.' + word.lower() + '.'
        points = [0] * (len(work)+1)
        for i in range(len(work)):
            t = self.tree
            for c in work[i:]:
                if c in t:
                    t = t[c]
                    if None in t:
                        p = t[None]
                        for j in range(len(p)):
                            points[i+j] = max(points[i+j], p[j])
                else:
                    break
        # No hyphens in the first two chars or the last two.
        points[1] = points[2] = points[-2] = points[-3] = 0

        # Examine the points to build the pieces list.
        pieces = ['']
        for c, p in zip(word, points[2:]):
            pieces[-1] += c
            if p % 2:
                pieces.append('')
        return pieces

def main():
    if len(sys.argv) < 4 or len(sys.argv) > 5:
        print("Usage: python hyphenate.py <pattern file> <input file> <output file> [<hyph char>]")

    pattern_file = sys.argv[1]
    input_file = sys.argv[2]
    output_file = sys.argv[3]
    hyphen = "-" if len(sys.argv) == 4 else sys.argv[4]

    hyphenator = Hyphenator(pattern_file=pattern_file)
    
    with open(input_file, "r") as input, open(output_file, "w") as output:

        for line in iter(lambda: input.readline(), ''):
            chunks = hyphenator.hyphenate_word(line.strip())
            hyphenated = hyphen.join(chunks) + "\n"
            output.write(hyphenated)


if __name__ == '__main__':
    main()
