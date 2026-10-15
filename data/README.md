# Training and held-out evaluation data

These files contain mission commands and the answers we expect the parser to produce. Each line is one example: `text` is the human command, `expected` contains the six mission fields, and evaluation rows also have a `set` label for grouping results.

## What “held out” means

A held-out command is kept out of the model’s training examples and used afterward to test it. Think of `train_set.jsonl` as the practice questions and the evaluation files as the exam questions. The training step learns from `train_set.jsonl` only. Baseline, trained-model, and GGUF evaluations use the same nine evaluation files so scores stay comparable.

The numbered held-out files are separate batches inherited from the Schema V3 project. They are not training rounds, epochs, or files you need to run manually. The standard evaluation command combines all nine evaluation files into one 792-command evaluation. The separate files and group labels help locate weaknesses afterward.

## What each file contains

| File | Examples | Purpose |
|---|---:|---|
| `train_set.jsonl` | 3,855 | Training practice; 1,986 normalized unique commands, with intentional repeats to give selected examples more weight. |
| `eval_set.jsonl` | 150 | Main evaluation: core57 (57), ext (51), and ext2 (42). Despite its name, this is only part of the full evaluation. |
| `heldout_set.jsonl` | 201 | Evaluation groups labeled familiar (128), contested (44), and novel (30). “Familiar” is a group name, not permission to train on these rows. |
| `heldout2_set.jsonl` | 80 | Additional evaluation batch labeled heldout2. |
| `heldout3_set.jsonl` | 84 | Additional evaluation batch labeled heldout3. |
| `heldout4_set.jsonl` | 50 | Additional evaluation batch labeled heldout4. |
| `heldout5_set.jsonl` | 52 | Additional evaluation batch labeled heldout5. |
| `heldout6_set.jsonl` | 52 | Region and route cases: staying in halves/quadrants, temporary restrictions, avoidance, and pathways. |
| `heldout7_set.jsonl` | 63 | New stay-region wording and distinctions such as pause versus abort, abort versus return, and resume versus retry. Also includes avoidance, chip-reading, and negative route wording. |
| `heldout8_set.jsonl` | 51 | Stop wording, positive/negative wording pairs, resume/retry distinctions, and control examples. |
| `heldout9_set.jsonl` | 99 | Fresh set written by a different model than the training data and never trained on: indirect target locations, avoid idioms, inverted stay wording, corrections. Scores here run ~20 points below the older sets. |
| **All evaluation files** | **882** | The published accuracy tables predate heldout9, five ambiguous-label removals and four self-answered question rows, and use 792. |

## Why there are no paragraphs at the top of the data files

`.jsonl` means JSON Lines: every line must be a valid data record. A heading, comment, or extra explanatory record would break the existing reader or change the frozen data fingerprints. This README explains the files without changing their contents. GitHub displays it when you open the `data/` folder.

Keep these files unchanged when reproducing the current study. The checks verify their fingerprints, allowed fields, and absence of normalized command overlap between training and evaluation. Repeatedly examining these results makes them a development benchmark; a future final validation should use new, untouched commands too.
