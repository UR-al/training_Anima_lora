# dataset_tags — character & series name lists

User-owned lists that drive the **dataset-view tag sorter** (Utils → Dataset → *Tag
order*). They are **authoritative**: any caption tag matching a name here is forced into
the character / series bucket (overriding the anima-tagger `vocab.json`), so the
keep-tokens head always comes out in this order:

```
metadata  →  count (1girl…)  →  character  →  series  →  @artist  →  general
```

## Files

| File | Holds | Bucket |
|------|-------|--------|
| `characters.txt` | character names | `character` |
| `series.txt` | series / works (copyright) | `series` |

One name per line. `#` starts a comment; blank lines are ignored. Matching is
case-insensitive and underscore/space-insensitive (`hatsune_miku` == `hatsune miku`).

## Swapping in your real lists

These ship as **placeholders**. Replace the contents of `characters.txt` and
`series.txt` with your real lists — that's all; no code change. In the GUI you can also
point at files elsewhere via the *Characters file… / Series file…* buttons, then
*Reload*.

## Sorting the whole dataset

In the Dataset tab, **Sort ALL in dataset** reorders every `.txt` caption in the loaded
folder at once (with the keep-tokens separator if the checkbox is on). **Sort current**
does only the open image.
