# Results

### 1. Simple SUK, Target bad hyphenations
Weight = 1
Annotations = claude_new_prompt

Objective SUK = f17_trie
Gamma SUK = 0.00085

Objective FUK = f17_target
Bad target = 250
Bad tolerance = 50

FUK optimization results:
Best parameters: (15, 18, 2, 25, 1)
  bad_weights=(15, 18, 2, 25), threshold=1
Results:
  good=13155556, bad=255, missed=136
  n_patterns=13888, trie_nodes=24585
  score=1.0000

### 2. Weighted SUK, Minimize patterns size
Weight = 9
Annotations = claude_new_prompt

Objective SUK = f17_trie
Gamma SUK = 0.00085

Objective FUK = f17_trie
Gamma FUK = 0.05

Best parameters: (13, 30, 3, 1, 1)
  bad_weights=(13, 30, 3, 1), threshold=1
Results:
  good=13153841, bad=23, missed=1851
  n_patterns=13144, trie_nodes=23666
  score=0.9997

### 3. Simple SUK, Minimize patterns size
Weight = 1
Annotations = claude_new_prompt

Objective SUK = f17_trie
Gamma SUK = 0.00085

Objective FUK = f17_trie
Gamma FUK = 0.05

Best parameters: (13, 30, 3, 1, 1)
  bad_weights=(13, 30, 3, 1), threshold=1
Results:
  good=13153841, bad=23, missed=1851
  n_patterns=13144, trie_nodes=23666
  score=0.9997

### 4. Weighted SUK, Target bad hyphenations
Weight = 9
Annotations = claude_new_prompt
Gamma SUK = 0.00085

Objective SUK = f17_trie
Gamma SUK = 0.00085

Objective FUK = f17_target
Bad target = 250
Bad tolerance = 50

Best parameters: (23, 30, 2, 20, 1)
  bad_weights=(23, 30, 2, 20), threshold=1
Results:
  good=13155410, bad=216, missed=282
  n_patterns=15033, trie_nodes=25827
  score=1.0000

Cross validace
