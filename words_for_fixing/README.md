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