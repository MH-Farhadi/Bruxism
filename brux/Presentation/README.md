# Presentation — scope update, 27 Aug 2026

A short deck for the group meeting explaining why the bruxism paper changed direction
(from an audio–EMG classifier to a stage-by-stage audit of the whole pipeline), why the
microphone was dropped, what stands, and where to submit.

| File | What it is |
|---|---|
| `scope_update_deck.pptx` | The deck — 18 slides, 16:9, speaker notes embedded in every slide (six slides on the microphone evidence, two on venues with impact factors and deadlines) |
| `scope_update_deck.pdf` | A preview rendered with LibreOffice (Carlito stands in for Calibri, so spacing differs slightly from PowerPoint) |
| `script.md` | The presentation script (same text as the notes pane), running order with timings, anticipated Q&A, and the assumptions made while building the deck |
| `venues.md` | The venue research behind slide 11: every conference checked, why it fails the US + New England + full-paper conditions, the journal shortlist with fees, and the sources |
| `build_deck.py` | Regenerates the deck, the script and the budget chart from scratch |
| `assets/` | Figures embedded in the deck (copied from `Paper/…/Figures` and `advisor/figures`) and `budget_chart.png`, drawn from Table 8 of `main_v8.tex` |

Rebuild (needs `python-pptx`, `matplotlib`, `Pillow`; all present in the project environment):

```bash
python3 Presentation/build_deck.py
```

Every number on the slides is quoted from `Paper/K_Farhadi_Paper_Bruxism/main_v8.tex`
(26 Aug 2026), `advisor/briefing.tex` (19 Aug 2026), `cause.md` / `audio.md` (git HEAD)
and the Scientific Reports decision (`Paper/Reviews/`). Nothing was recomputed here.
