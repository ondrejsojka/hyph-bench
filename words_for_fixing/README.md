Tohle je můj pokus o vygenerování těch 500 slov. Nejblíže jsem se dostal 512, což by podle mě mohlo být v pohodě.
Docílil jsem toho primárně upravováním trie_weight. Zkoušel jsem měnit R v F_{1/R}, ale nepovedlo se mi to dojít 
k těm 500ti blíže.

Výsledný seznam je tedy vytvořený s těmito parametry
- R = defaultní, takže se pořád používá f1/7
- trie_weight = 0.00085
- trie_normalizer = 15714, takže velikost původního wordlistu

### Soubory
- find.sh je script který jsem použil na zkoušení trie_weight hodnot (původně byl v kořenu adresáře, takže teď cesty nesedí)
- weight_settings.csv jsou všechny trie_weight hodnoty které jsem vyzkoušel pomocí skriptu
- uk_bad.txt jsou tedy ta slova, která se mají opravit
- uk.pat jsou vzory, které vznikly z gaussian procesu
- uk_fixed.txt je výsledek následujícího promptu v Gemini 3.1 pro
- uk_hyphenation_rules.txt jsou přepsaná pravidla

### Prompt
Using syllabic hyphenation and the rules provided in the uk_hyphenation_rules.txt file hyphenate each word in 
the uk_bad.txt file and output them in plaintext in the following format: {input word}={input word hyphenated}. 
Other than inserting hyphens at the hyphenation points, do not change the words in any other way. 

