# Thesis Word Bank

Binding vocabulary and phrasing rules for all prose in `documents/thesis`.
Any new writing or edit must comply. When a rule and a sentence conflict,
rewrite the sentence. Add new entries here whenever the user flags a word.

## 1. Preferred term vs. avoid

| Use | Not |
|-----|-----|
| dataset | benchmark, corpus |
| combine / combining / combined | pool / pooling / pooled (for datasets/sets) |
| test set | testbed |

(One term per concept — do not alternate synonyms for variety. Pick the
"Use" word and keep it everywhere: prose, figures, tables, captions.)

**Exempt:** the neural-network *pooling operation* (mean-pool, average-pooling,
max-pool) is a technical term and keeps the word "pool". The rule only bans
"pool/pooling" in the sense of combining datasets or collections.

## 2. Style rules

- **No long adjective chains.** Do not stack three or more modifiers before a
  noun. Break them into a clause instead.
  - Avoid: "a widely used in-the-wild engagement dataset of short frontal
    webcam clips".
  - Prefer: "an engagement dataset of short webcam clips, recorded in the
    wild".
- **No long noun phrases / compound pile-ups.** Keep pre-modifier stacks to
  two words; move the rest after the noun with "of", "for", "that".
- **One idea per sentence** where a chain of clauses is doing the work of a
  list.
- **No thousands separators in numbers.** Write digits plain, with no comma or
  `{,}` grouping: `1500`, `1221`, `10000` — never `1{,}500` or `1,500`.

## 3. Banned words and phrases (AI tells)

Do not use: "sits above", "payoff", "compound" (as a verb of accumulation),
"delve", "leverage" (verb), "underscore", "showcase", "seamless", "robustly"
as filler, "it is worth noting", "notably" as filler.

(See memory `feedback_thesis_prose_voice` for the wider voice rules.)

## 4. Cross-reference direction

- **Prefer backward references.** A section explains its own material and, when
  it depends on an earlier idea, points back to it. Do not forward-reference a
  later section to justify content in the current one; let the later section
  reference back instead.
- Do not describe a subsection from within a sibling subsection (e.g. do not
  mention §3.2.2 inside §3.2.1).
