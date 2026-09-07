# Script — “Where Does the Performance Come From?” (scope update, August 2026)

Deck: `scope_update_deck.pptx` (18 slides, 16:9). The same text is in each slide’s notes pane. Planned length about 24 minutes plus discussion; the per-slide budget is in brackets.

Every number is quoted from `main_v8.tex` (26 Aug), `advisor/briefing.tex` (19 Aug), `cause.md` / `audio.md` (git HEAD) and the Scientific Reports decision letter (`Paper/Reviews/`). Venue facts were checked on 28 Aug 2026; sources are in `venues.md`. Nothing was recomputed for the deck.

## Running order

1. Title — Where Does the Performance Come From? [0:45]
2. Where we were: v1–v3 sold a fusion classifier for bruxism [1:45]
3. How we got here: three audits, six weeks, one surviving modality [1:30]
4. Defect 1: the 60-Hz notch removed the one frequency that was already gone [1:30]
5. Defect 2: the microphone column is not per-participant audio [1:30]
6. Microphone: one waveform, replayed under five participants [1:30]
7. Microphone: what broke — ring-buffer offsets, copies grouped by condition [1:15]
8. Microphone: the audio ablation scored the training set [1:30]
9. Microphone: not usable audio even if it were unique [1:15]
10. Microphone: clenching registers quieter than silence [0:45]
11. The new scope (v8) [1:30]
12. What stands: EMG alone, on participants the model never saw [1:45]
13. The main result: signal quality sets the ceiling; representation beats size [1:30]
14. What we tell readers: the five-step procedure [1:30]
15. What it means for the group and the NIH direction [1:45]
16. Where to submit: journals, with impact factors and fees [1:30]
17. Conference calendar: dates, deadlines, and the New England test [1:15]
18. Decisions [0:45]

---

## Slide 1 — Title — Where Does the Performance Come From?  [0:45]

Good afternoon. This is a short update on where the bruxism paper stands and why it looks so different from the manuscript we sent to Scientific Reports.

The one-sentence version: we set out to sell an audio-plus-EMG classifier for bruxism-related activities, and along the way we found that two things nobody had measured were defective: the signal entering the model, and the microphone channel itself. The EMG survived. The paper now audits the whole pipeline, prices every stage, and tells readers how to do the same on their own data.

I will go through: where we were, the three audits, the two defects, with the microphone evidence in some detail, the new scope and results, the advice we give readers, what this means for the group and the NIH direction, venues with their dates and impact factors, and the decisions I need today. About twenty minutes, then discussion.

Everything I quote is from v8 of the manuscript (26 Aug), the advisor briefing (19 Aug) and the audit notes; nothing here is a new number.

---

## Slide 2 — Where we were: v1–v3 sold a fusion classifier for bruxism  [1:45]

Where we were. The submitted manuscript, v1, was a detection story. It claimed 85 percent accuracy and an 80 percent F1 across four activities, movement, clenching, grinding and chewing, with no resting class, and it credited the microphone with a seventeen-point gain. It called itself real-time.

Scientific Reports rejected it on 3 July. Two reviewers were positive, two were not, and the editor sided with the two critics. Reviewer 2 is a bruxism clinician: the definition we used, bruxism equals tooth contact, is outdated; the phenotype was unspecified; and they wanted the STAB tool, the biopsychosocial model and TMD standard-of-care discussed. Reviewer 4 was technical and, frankly, correct: chewing is easy and inflates the headline; without a resting class it is not detection; the sentence about generalising to unseen individuals over-claims from five people; and the events were emulated.

Versions 2 and 3 were the resubmission draft. They fixed the protocol problems, added quiet rest, nested leave-one-subject-out, participant-level metrics, and wrote the clinical sections Reviewer 2 asked for. But the story was still fusion: the fused model at 82.1 percent accuracy, and RQ2, how much does audio add, and RQ3, does our dual-branch network beat the baselines, were still the spine.

What none of those versions had done was measure the signal or the channel the whole story rested on.

---

## Slide 3 — How we got here: three audits, six weeks, one surviving modality  [1:30]

How we got here, in six weeks and three audits.

First, the code audit at the end of July. When we rebuilt the prototype as a proper package, we found the held-out subject had been used for early stopping and checkpoint selection and then reported as the test score. There was window-level K-fold on overlapping windows, a focal-loss bug, mislabelled wavelet bands. And the published 85 percent confusion matrix cannot be reproduced under any labelling policy: its total equals the count including 595 rest windows that the matrix does not contain. The rebuild seals the outer fold structurally, has 196 tests, and regenerates every number from a saved prediction ledger.

Second, the signal audit on 3 August. The acquisition hardware had already notched 60 hertz. The interference that survived sat at 180, 300 and 420. Our textbook chain notched 60, the one frequency that was absent, and passed everything else. In the quiet-rest recordings, 91 to 99.8 percent of in-band power was mains.

Third, the channel audit on 12 August, which I will show in detail: the microphone column is 37 waveforms replayed across 100 recordings.

The versions after that, v4 to v8, are the paper being rewritten around what the data can actually support: the microphone came out in v5 on 15 August, and the advisor briefing on 19 August put that direction, with the evidence, to the advisor.

---

## Slide 4 — Defect 1: the 60-Hz notch removed the one frequency that was already gone  [1:30]

Defect one, in one figure. Panel a is the superseded chain drawn over the measured spectrum of the recordings. The notch lands at 60 hertz, where the hardware had already removed everything, and the spikes at 180, 300 and 420 pass straight through. Panel b is the corrected chain, seven constant-width notches on the harmonics. Panel c is the residual mains share per participant: above 0.9 before, under 0.01 after.

What it was worth: the matched network went from 43.5 to 72.8 percent macro-F1 on byte-identical windows, folds and labels, 29.3 points. Quiet-rest windows misread as clenching or grinding fell from 30 percent to 6. And two participants who had been at or below chance, P1 at 18 and P2 at 5 percent macro-F1, are now in the 61 to 82 range with everyone else.

Two things to stress. The chain looked correct on its own response plot; the defect only appeared when the response was drawn over the data. And it did not look like a signal problem; it looked like between-participant heterogeneity, which is the most familiar failure mode in this literature and the one that motivates transfer learning and bigger cohorts. That is why the check is now a pipeline stage that refuses to run, not a review step.

---

## Slide 5 — Defect 2: the microphone column is not per-participant audio  [1:30]

Defect two, starting with the whole finding in one picture. The figure compares every recording against every other by exact waveform identity. Left is the microphone, right is EMG channel 1 from the same 100 files. Red means two recordings contain the same samples and belong to different people. The microphone panel is striped with red across every participant boundary; the EMG panel has nothing off the diagonal.

The numbers: 37 distinct waveforms in 100 recordings; 83 recordings share a waveform with a different participant; of 63 pairs we tested, all 63 are exact circular rotations, the largest residual after alignment is zero. All four EMG channels are 100 out of 100 distinct, so every EMG result stands.

The consequence is that leave-one-subject-out never held out the audio. The next four slides show what that looks like in the signals themselves, what broke, what it did to the numbers, and why the channel would not have been usable audio even if it were unique.

The decision at the end of it: we dropped the microphone from every result, recorded RQ2 as untested rather than refuted, and kept the check, a rotation-invariant fingerprint of every channel on every manifest build that refuses a run if a held-out participant's signal is in its own training set.

---

## Slide 6 — Microphone: one waveform, replayed under five participants  [1:30]

This is the strongest single exhibit, and it is the one to linger on if anyone doubts the finding.

Panel A is the microphone channel as stored, for the deviation-left-right condition, from five different people, recorded in five sessions across three days in August 2025. Look at the shapes: the same dropout, the same bumps, just displaced in time. They look shifted, not different.

Panel B undoes a circular shift for each trace, 1.90 seconds for S02, 3.12 for S03, minus 0.99 for S04, 1.90 for S05, and the five collapse onto one line.

Panel C is the difference from S01 after alignment. The largest absolute difference anywhere in the full 60-second recording is zero counts. Not correlated. Not similar. Identical, every sample, bit for bit.

Panel D is the EMG from those same five files: five clearly different people. Whatever failed, failed only on the microphone column.

One more detail worth saying out loud: S02 and S05 needed the same 1.90-second shift. That is what you would see if S02's column had been copied directly into S05's files, and the audit confirms it: S05's four duplicated recordings carry S02's exact sample offsets.

---

## Slide 7 — Microphone: what broke — ring-buffer offsets, copies grouped by condition  [1:15]

What broke, mechanically. We can say more than the files are duplicated, and the advisor will ask whether this could recur.

Left: the shifts across all 63 confirmed pairs. They are small, mostly within a few seconds, and they wrap in both directions around zero. A resampling or export bug does not produce circular rotations; a buffer whose read pointer is not reset between sessions does. That is a ring-buffer trace.

Right: how the copies are organised. Each row is one waveform stored under several participants, under the same condition. Four waveforms appear under all five people; sixteen more under four. The copies are grouped by condition, not by session, which is what turns a duplicated file into a labelled-data problem: the model sees the same audio under the same label in training and in test.

Corroboration from outside the CSVs: the three numpy companions we have, all from S01, match the CSV exactly on EMG and trigger and have an all-zero microphone column, so the acquisition array held no microphone data. And all 100 video files parse to one video stream and zero audio streams. There is nothing to recover the true audio from.

The requirements for the next collection follow directly: audio in its own file on its own clock, never a shared buffer, a hardware sync marker on both streams, and a fingerprint check at ingest.

---

## Slide 8 — Microphone: the audio ablation scored the training set  [1:30]

What it did to the results. Left: for each held-out participant, how many of their 20 recordings carry a microphone waveform that was also in the training set. S01, S03 and S04: all twenty. S02: nineteen. S05: four. Leave-one-subject-out held out the participant; it did not hold out their audio. For the four S01-to-S04 rest recordings, which share a single waveform, every fold had already been trained on the exact rest audio three times.

Right: S05 is the control nobody designed. It is the only participant whose audio was mostly not copied, and it is the worst participant on audio-only macro-F1, 0.354, while being the best on EMG-only, 0.805. If S05 were simply a hard subject it would be bad on both. That dissociation is the signature of leakage.

The table gives the per-participant numbers from the ablation ledger. And the one cross-condition swap behaves as predicted: S05's protrusion recording carries the incisor-clench waveform, and its 63 audio-only predictions across three seeds were clenching, grinding and rest, never movement.

So the audio-only and fusion rows of the old Table 4 are withdrawn, along with the reading that the microphone helped separate quiet rest from tooth contact, and three to four screening points that came from seven microphone features. The direction of the bias is not even uniform, so we cannot say the true value is lower; we can only say the number is not what its name says.

---

## Slide 9 — Microphone: not usable audio even if it were unique  [1:15]

Could we still use it as audio if it were not duplicated? No, and this slide is why.

Left: where the microphone's energy is. Ninety-six percent of its power lies below 10 hertz, the shaded band. The EMG, for comparison, is broadband to 450. The production 20-hertz high-pass, the dashed line, throws away almost everything the channel contains: a median 1.19 percent of the variance survives.

Middle: all 100 recordings agree, 91 to 99 percent of power below 10 hertz in every one. A microphone recording of tooth contact puts its energy in the kilohertz range; at a 1200-hertz sampling rate the Nyquist limit is 600, so that range does not exist in these files. This is a sound-level envelope, not an acoustic waveform.

Right: the samples themselves. Integer-valued, one-count steps, 15 to 145 distinct values in a minute; a three-second excerpt here takes two values. After the high-pass, what survives for clenching and grinding sits at or below the one-count quantisation floor, so for the two classes the study exists to measure the audio branch was being fed converter dither.

---

## Slide 10 — Microphone: clenching registers quieter than silence  [0:45]

This is the version that needs no signal-processing background, and the one I would show a clinician.

One point per recording: each activity's amplitude divided by that same participant's own quiet-rest session, on a log scale. Above the dashed line is louder than rest; below it is quieter.

Left, the EMG in the analysis band: every activity sits above that person's own quiet rest, and the ordering is the physiological one, chewing 5.8 times, grinding 4.1, clenching 2.4, movement 1.7.

Right, the raw microphone channel: chewing 1.8 times, movement 1.3, and then grinding at 0.44 and clenching at 0.30. Clenching and grinding, the two behaviours the study exists to measure, register quieter than a closed, silent mouth. No working microphone does that.

That contrast, side by side on identical recordings, is the fastest way to see that the column is not what it was labelled as, and it is the check we would run first on any new collection.

---

## Slide 11 — The new scope (v8)  [1:30]

The new scope. The title is now "Where does the performance come from? A stage-by-stage analysis of a biosignal classification pipeline, with surface-EMG jaw-activity recognition as the case study."

The thesis is simple. Pipelines are reported through a summary accuracy and a block diagram, which says how well a pipeline scored and nothing about which stage earned it. We vary one stage at a time on identical windows, folds and labels, and put every stage on one scale.

Four research questions. RQ1: how much of what the classifier learns is physiology and how much is mains. RQ2: can four EMG channels separate quiet rest from four instructed jaw activities on an unseen participant. RQ3: which part of the model does the work, representation, architecture or parameter count. RQ4: how do those compare with choices fixed before training, window length, temporal context, normalisation scope.

On the right, what is kept, gone and new. Kept: the five-class EMG task with rest, the nested LOSO protocol, the compact wavelet CNN, and the dataset, which is now foregrounded with a section on why it had to be collected. Gone: fusion and the microphone, the detection and real-time claims, and the clinical sections we wrote for Reviewer 2, because the target venue is engineering. New: the filter contrast, the architecture sweep, the pre-model choices, the budget table, guards that refuse, a positioning table against ten prior studies, and a procedure section that transfers to EEG, ECG and inertial pipelines.

---

## Slide 12 — What stands: EMG alone, on participants the model never saw  [1:45]

What stands. The pipeline is ordinary on purpose: notch bank, band-pass, one-second windows z-scored with training-participant statistics, a db4 wavelet decomposition, three small convolutional branches on the A4, D3 and D1 bands, and an MLP. 5,901 parameters, about a millisecond per window on a CPU.

On participants it never saw, nested LOSO with three seeds: 81.7 percent accuracy, 72.0 macro-F1, 0.958 macro AUC, on 6,173 windows. Rest is the best-recalled class at 91 percent. Movement is the weak one, and it is a precision problem: it is the smallest class and absorbs chewing windows. Clenching and grinding are the hard pair, as they should be.

The number I would point the clinicians at: 7.7 percent of quiet-rest windows land on the tooth-contact side. Most work in this area classifies activities against each other and cannot say anything about false alarms; this is the closest analogue our design can produce.

The architecture sweep at the bottom answers Reviewer 4's question about baselines. Both wavelet models placed above both raw-signal models, and the largest model placed third. The extended model, which is our model with the time axis kept and a rhythm head, is the strongest at 75.0, but the 2.8-point edge rests on three of five participants, so it is reported as secondary and the compact model stays the characterised reference.

---

## Slide 13 — The main result: signal quality sets the ceiling; representation beats size  [1:30]

This is the main result of the paper: the budget. Six choices priced on the same 6,173 windows and five folds, in points of participant-level macro-F1.

Signal quality sets the ceiling at 29.3 points. Below that ceiling nothing helps: through the superseded chain the best model scored 43.5, which is lower than the 61.3 the weakest baseline reaches on the corrected signal. No architecture could recover what the filter passed through.

Above the ceiling, what a model can represent mattered more than how big it was: the learned wavelet representation beats the band energies it is built from by 8.1 points; an explicit rhythm representation adds 2.8; and 3.2 times the parameters on the raw waveform loses 3.3.

And two choices fixed before any model was trained, temporal context at 7.4 and normalisation scope at 9.1, are worth as much as the architecture or more. Neither is a modelling decision.

The caveats are in the paper and I will say them here too: the entries are not additive; three rows are screening estimates, optimistic in level; and the magnitudes belong to this cohort and this lab. What transfers is the ordering and the practice of measuring it.

---

## Slide 14 — What we tell readers: the five-step procedure  [1:30]

This is the advice we give readers, and it is the part that generalises. Five steps, none specific to EMG or bruxism.

One: audit the signal against what the analysis assumes, on the recordings you have. Draw every filter response over the measured spectrum, not alone. Fingerprint every channel of every recording. Publish the interference statistic beside the accuracy.

Two: fix the data, meaning the windows, folds and labels, verify that they are identical ledger to ledger, and vary one stage at a time.

Three: price the choices made before training: window length, temporal context, normalisation scope, label source. Here two of them were worth more than the architecture, and nobody in this literature prices them.

Four: report a budget beside the accuracy, one row per stage, naming its harness and its baseline, and say what is not additive.

Five: keep a prediction ledger and regenerate every number and figure from it, so a figure cannot disagree with the table it depicts.

And build guards that refuse rather than warn: the leakage assertion on every fold, the check that no branch reads a band its own filter deleted, and the check that no run trains on a held-out participant's own waveform. The whole thing costs one pass over the data.

Both of our defects had this shape: a correct check, scoped one step too narrowly, to the filter rather than the data, to the file rather than the dataset.

---

## Slide 15 — What it means for the group and the NIH direction  [1:45]

What this means for the group, and for the NIH direction, which I know is the real concern.

Why the reframe is right. A leave-one-subject-out number on a channel that was not held out is the specific kind of result that gets papers corrected. A caveat cannot fix it, because the number is mislabelled, not merely uncertain. The EMG-only paper has a clear question, a quantified answer, and a method others can adopt. And finding two measurement-chain defects in one dataset, with the checks that catch each, reads as a group that measures its own instruments. The bad version of this story is the one where someone else finds it.

Why it helps a microphone-plus-EMG grant. Nothing here refutes the acoustic hypothesis; these files contain no usable audio, so it is untested. It removes a latent leakage artifact before it could sit in a Preliminary Studies section. It is concrete rigor-and-reproducibility evidence: a measurement, a versioned policy, tests, an ingest gate, which reviewers score. And it converts "we will collect audio" into a protocol-plus-validation aim; the nine-requirement acquisition spec already exists in the repo.

The proposal: publish the EMG paper now; run a small audio re-collection pilot, five to ten people, against that spec, with audio at 16 kilohertz or better in its own file on its own clock, a hardware sync marker, and a fingerprint check at ingest; and let that pilot be the grant's preliminary data. It gives a better audio aim than the contaminated data ever could, for the cost of one pilot.

---

## Slide 16 — Where to submit: journals, with impact factors and fees  [1:30]

Where to submit, starting with journals, because the recommendation is a journal first: rolling submission, no location constraint, and a methods-and-audit paper suits twenty pages better than six. The impact factors are the 2025 Journal Citation Reports values released in June, cross-checked against the publishers' own pages where those show them; fees are the 2026 open-access charges.

First choice is JBHI, impact factor 7.7: it publishes exactly this kind of sensor-to-decision pipeline work and values the leakage-free protocol. TBME, 4.4, if we want the strongest line on a CV and can wait. Physiological Measurement, 2.5, is a genuinely good fit for the audit framing; its scope statement is literally the development and validation of methods of measurement, and it states a median 49 days to first decision. If we need something citable before the grant is written, OJEMB at 3.1 and IEEE Access at 4.2 are the fast, open-access options at 2,160 dollars each for 2026, less with IEEE or EMBS membership. Computers in Biology and Medicine, 6.3, Biomedical Signal Processing and Control, 5.7, and the Journal of Electromyography and Kinesiology, 2.7, are the methods-and-signal alternatives; JEK is where the notch finding would land hardest with EMG people.

Below the table: Computer Methods and Programs in Biomedicine at 6.4, MBEC at 3.1, IEEE Sensors Journal at 4.5, and the Journal of Oral Rehabilitation at 3.8, which remains the strategic option if we want the work visible to the clinical community, but would need reframing.

---

## Slide 17 — Conference calendar: dates, deadlines, and the New England test  [1:15]

Conferences, with dates and deadlines. I applied three conditions: in the US, in New England, and full papers published in proceedings rather than abstracts. As of the check date, nothing announced for 2026 or 2027 meets all three.

The top block is the watch-list. BHI 2027 is the best fit: confirmed for the US in November 2027, city to be announced, full papers in Xplore, and on this year's pattern the deadline will fall around June 2027. BSN 2027 has not been announced; it was Chicago in 2024 and Los Angeles in 2025 before Porto, so a US edition is plausible. CHASE and MLHC are US-based with deadlines around February and April. NEBEC rotates through the Northeast, but it has run on abstracts in recent years, so it fails the full-paper condition unless the 2027 host changes the format.

The middle block is US but not New England. AMIA Amplify in Atlanta is the only deadline open right now, 3 September, and the fit is moderate. SPMB in Philadelphia closed on 1 July.

The bottom block is everything else we checked, all outside the US: EMBC in Singapore, NER in Milan, I2MTC in Chongqing, MeMeA in Rome with a January deadline, BHI 2026 in Hong Kong, SENSORS in Rotterdam, ML4H in Sydney.

The practical reading: submit to a journal this fall, and hold BHI 2027 as the conference target once its call appears.

---

## Slide 18 — Decisions  [0:45]

Four decisions I need from this meeting.

One: approve the v8 framing and title, the general stage-by-stage analysis with EMG jaw activity as the case study. It is 26 pages with references from page 20, and every number traces to a saved artifact.

Two: pick the journal. My recommendation is JBHI; OJEMB or IEEE Access if being citable before the grant matters more than the venue.

Three: approve the audio re-collection pilot and its acceptance criteria. The grant's audio aim rests on it, and the spec is already written.

Four: close the open items before submission. The hardware, gain and ADC documentation, which the paper currently reports as undocumented; the IRB identifier, where the records carry conflicting values; the data-release scope; and co-author sign-off on the dropped microphone and clinical sections.

Nothing in the EMG results changes if the answers are yes. Everything in the microphone story changes if the pilot is not run. Thank you; questions.

---

## Anticipated questions (from the 19 Aug briefing, plus three for the new framing)

**"Can't we just fix or realign the audio?"** No. The circular offset is known only *between copies*, never relative to that recording's own EMG, and the misalignment differs per recording (median best lag 18 s, nothing under 0.1 s). There is also nothing to realign to: the `.avi` files have no audio stream and the `.npy` companions have a zero microphone column.

**"Can we at least use S05, whose audio is mostly unique?"** S05 has 4 of 20 recordings duplicated, so leakage is milder there, but duplication is the *second* problem. S05's audio is the same sub-10-Hz, 1-count-step envelope as everyone else's, at the quantisation floor for clenching and grinding. One participant is also n = 1. It is a useful control (it is the worst participant on audio and the best on EMG, which is what leakage looks like), not a dataset.

**"Could we report the ablation with a strong caveat?"** We drafted it and rejected it. The caveat cannot rescue the number because the number is mislabelled rather than uncertain: it is presented as held-out and it is not. The bias is not even uniform in direction. It would be the one thing in the paper a careful reviewer could disqualify us for.

**"Does admitting this make the group look careless?"** The paper's whole argument is that summary accuracies hide where performance comes from, and that cheap measurements on your own data catch what design review does not. We found two independent measurement-chain defects and report both with the checks that catch them. The bad version of this story is the one where somebody else finds it.

**"How do we know the EMG is really clean?"** The same fingerprint test, on every EMG channel of every recording: 100 distinct waveforms out of 100 on all four channels, zero cross-participant sharing. It is asserted on every manifest build, so it cannot silently stop being true.

**"Is the trigger channel safe? Every label depends on it."** 95 distinct waveforms in 100 recordings; the five collisions are all-zero quiet-rest triggers, which is expected for a binary channel with no events. The trigger was written by the same acquisition software, and a targeted trigger audit is logged as an open item before the next collection.

**"Isn't the new paper just a negative result?"** No: the headline is a five-class result on unseen participants with a quiet-rest class, plus a measured budget of what each stage was worth. The filter finding is the first row of that budget, not the thesis; v6 tried the "measurement chain outweighs the model" framing and it was hard to place, which is why v7/v8 sell the whole pipeline.

**"Why claim generality from five participants?"** The claim is for the *procedure*, never the numbers. The paper says explicitly that a lab with clean mains would find the first row worth nothing, that magnitudes belong to this cohort, and that whether the ordering recurs elsewhere is a question the study poses and does not answer.

**"The extended model beats the reference. Why not promote it?"** A 2.8-point mean difference resting on three of five participants (and losing 6.7 on P2) is a direction, not an effect size; the compact model delivers 96% of the macro-F1 at 33% of the parameters and 46% of the latency; and the two are one family, so the gap is evidence *for* the representational argument. Promotion remains available if a reviewer pushes; the checkpoints exist.

**"Why is the impact factor for Computers in Biology and Medicine dated 2024?"** Its JCR 2025 value had not been posted by the aggregator or the publisher page at the check date; the 2024 value (6.3) is the latest confirmed one. Check Clarivate directly before citing it.

---

## Assumptions made while building this deck

- **Audience:** the co-author group and the advisor, i.e. people who know the project but have not read v8. The deck therefore spends six slides on the microphone evidence (the advisor briefing's figures, in its order) and one on the NIH argument.
- **"Version 2 and 3":** read as the original `Main_2.tex` (28 Jul, the Scientific Reports resubmission draft with `\TBD` placeholders) and the current `Main_2.tex` (9 Aug, same direction with real numbers). The v1 claims quoted (85.0% / 80.3% / +17.4) are those the reviewers saw; an earlier draft in `misc/` reported 81.8% / 80.0% / +14.2.
- **Journal named in the request:** there is no "IEEE Open Journal of Computers in Medicine and Biology"; the closest venues are IEEE OJEMB (Open Journal of Engineering in Medicine and Biology) and Elsevier's *Computers in Biology and Medicine*. Both are on the journals slide.
- **Impact factors** are Clarivate JCR 2025 values (released 17 Jun 2026) as reported by publisher pages where those render (IOP, Wiley, Springer, IEEE Sensors Council, IEEE Access) and by a year-by-year JCR aggregator for the rest; the OpenAlex-based "Journal Metrics Score" that some sites label as an impact factor was *not* used. Verify in JCR before quoting in a cover letter.
- **Conference conditions** (US, New England, full papers published): applied strictly. Nothing announced meets all three as of 28 Aug 2026, so the calendar slide says so; "expected" deadlines are inferred from the previous edition and are flagged as such.
- **Pronouns:** the advisor is referred to by role throughout.
