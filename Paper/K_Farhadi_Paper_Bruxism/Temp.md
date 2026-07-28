# Revision & Resubmission Plan
## "Sensor-based Bruxism Detection with Dual-Branch Wavelet CNNs and Audio-EMG Data Fusion"

**Status:** Rejected from *Scientific Reports* (split decision; editor sided with the two critical reviewers).
**Purpose of this document:** a complete, prioritized list of every change needed before resubmission, plus the concrete analyses to run and the exact citations to add. Nothing here is optional if you want the paper to survive competent review a second time.

**How to read this:** Work top-to-bottom. Tier 1 items are blockers — if any one is left unaddressed, a competent reviewer will reject again. Tier 2 items are things reviewers *will* raise. Tier 3 is polish. The traceability matrix (§10) maps every original reviewer comment to an action so you can prove nothing was dropped.

**One blunt framing note up front:** the reject was defensible. Reviewer 4's technical points are correct, and the paper as written oversells a modest-but-real result. The core finding — that adding cheap audio to EMG resolves the chewing/grinding confusion — is legitimate. The problem is that the *headline* (85% "bruxism detection") is inflated by design choices, and two competent reviewers caught it. The fix is not to defend the framing; it's to right-size it. A paper that honestly reports ~75% on the clinically relevant task, adds a resting class, and is framed as a sensing/ML proof-of-concept is *more* publishable than the current one, not less.

---

## 1. The central reframe (do this first — everything else depends on it)

Before touching individual comments, commit to one reframing decision, because ~70% of the reviewer objections collapse once you do:

> **This paper is a proof-of-concept for a low-cost sensing + ML pipeline that discriminates awake, teeth-contact oral behaviors (clenching and grinding) from non-bruxism activity. It is NOT a validated clinical bruxism-detection system.**

Three consequences follow, and you must apply them *consistently* across title, abstract, intro, results framing, discussion, and conclusion:

1. **"Teeth-contact" bruxism, not "bruxism."** Current consensus defines bruxism as also including *bracing* and *thrusting* — muscle activity **without** tooth contact. Your sensors and your task only address the tooth-contact subset (clench + grind). Say so explicitly. (This is Reviewer 2's #1 point and it is correct.)
2. **"Emulated/voluntary episodes by people who have bruxism," not "authentic pathological bruxism."** Your Methods currently try to claim both, and they contradict each other (see Tier 1, item C). Pick the honest one.
3. **Report the honest number.** The 85%/80.3% headline includes chewing, which is ~47% of your data and your easiest class. The number that matters for the clinical claim is ~75% (see Tier 1, item A for the arithmetic). Lead with the honest number; you can still report the 4-class result as a secondary/scientific result.

If you accept this frame, most of the "over-claiming" objections resolve themselves and the remaining work is mechanical.

---

## 2. TIER 1 — Blockers (fix these or expect another reject)

### A. Right-size the headline metric — the chewing-inflation problem
**Raised by:** Reviewer 4, Comments 1 & 2.
**Why it's fatal:** Chewing is 5,604 / 11,845 windows (**47.3%** of the dataset) and is your easiest class (F1 = 97.2%). It props up every global number. A reviewer who removes it in their head — as Reviewer 4 did — sees a very different paper.

**The honest arithmetic, straight from your own confusion matrix (Fig. 2):**
- Overall 4-class accuracy = (1541 + 1725 + 1422 + 5380) / 11,845 = **85.0%** ✓ (matches your claim)
- On the three clinically relevant classes only (movement, clenching, grinding), counting windows misrouted to "chewing" as errors:
  - Correct = 1541 + 1725 + 1422 = 4,688
  - Total non-chewing windows = 1877 + 2503 + 1861 = 6,241
  - **Estimated 3-class accuracy ≈ 4,688 / 6,241 = 75.1%**
- Weighted F1 across those three classes (from Table 3): ≈ **74.2%**

> **Caveat you must state honestly:** the 75% figure is an *estimate read off the existing confusion matrix*, not a retrained 3-class model. A proper 3-class (or bruxism-vs-rest) result requires retraining — the per-class precision would shift (e.g., grinding precision improves once the 213 chewing→grinding false positives disappear). **Retrain and report the real number.** Do not just quote 75% from arithmetic.

**Action:**
1. Retrain and report results **with and without** chewing.
2. Make the clinically relevant metric (without chewing, or bruxism-vs-rest — see item B) the *headline*. Demote the 4-class number to a secondary result.
3. Explicitly address Reviewer 4's accusation that chewing was included to inflate results. The honest defense is: chewing was included because distinguishing pathological grinding from eating is the *actual* real-world challenge — but you agree it should not dominate the reported metric, so you report both.

### B. Add a resting / null class — the single highest-value change you can make with no new data
**Raised by:** Reviewer 4, Comment 2 ("It is damaging to the study that the 'resting' class was not included").
**Why it's fatal:** "Detection" implies bruxism-vs-everything-else. You built bruxism-vs-three-other-*active*-behaviors. In real deployment the jaw is at rest the overwhelming majority of the time, and your system has literally never been tested against rest.

**You already have the data.** Your protocol (§6.1) recorded **"1 minute of quiet rest"** per participant in the baseline phase. Use it.
- Approximate yield: 60 s at 1 s window / 0.5 s stride ≈ **~119 windows/participant × 5 ≈ ~595 rest windows** (approximate; depends on exact captured duration and boundary exclusion). Comparable in size to your grinding class (1,861) but smaller than chewing.

**Action:**
1. Add rest as a class and retrain. Report a 5-class result *and* the clinically meaningful **binary** (bruxism = clench+grind vs. non-bruxism = rest+movement+chewing) and/or **ternary** (bruxism vs. chewing vs. rest+movement) framings.
2. **Expect the headline to drop.** That is the point. A lower, honest, rest-inclusive number is far more credible and deployment-relevant than 85%.
3. **State the residual limitation honestly:** even with rest added, your lab rest data (~1 min/participant) is tiny relative to real deployment where rest dominates. Adding it helps but does not fully solve deployment realism. Say this.

### C. Resolve the "emulated vs. authentic" contradiction — your Methods currently argue against themselves
**Raised by:** Reviewer 4, Comment 5.
**The contradiction, verbatim from your own text:**
- §6.1: participants performed *"3-minute continuous trials **emulating** natural bruxing."*
- §6.2: *"the grinding episodes captured in this dataset reflect **genuine pathological activity rather than simulated behaviors**."*

These cannot both be true. Grinding performed on command in a lab by someone who has bruxism is still **voluntary, emulated grinding** — it is not spontaneously captured pathological activity. Recruiting real bruxers improves realism; it does not make on-command episodes "natural." Voluntary/overplayed grinding is plausibly *easier* to detect than subtle spontaneous bruxism, which biases your accuracy upward.

**Action:**
1. Drop the "genuine, not simulated" claim. State plainly: *episodes were voluntarily performed (emulated) by participants who have clinically confirmed bruxism.*
2. Add explicit discussion (Reviewer 4 asked for exactly this): you have **no data** on whether emulated episodes match spontaneous ones in intensity/duration/pattern, and note the likely direction of bias (emulated → easier).
3. This is the deepest issue in the paper. Once you concede it and adopt the proof-of-concept frame (§1), it stops being a fatal flaw and becomes an honest limitation.

### D. Kill the over-generalization sentences — they directly contradict your own Limitations
**Raised by:** Reviewer 4, Comment 4.
**The offending sentence (Conclusion, §5):**
> *"The leave-one-subject-out validation provides evidence that the learned representations generalize to unseen individuals, supporting the validity of our approach for awake bruxism detection."*

This says the exact opposite of your Limitations section, which correctly states N=5 cannot characterize population variability. Reviewer 4 called it "a perfect example of over-generalization" — because it is.

**Action:**
1. Delete or rewrite this sentence. Acceptable replacement: *"Under leave-one-subject-out evaluation, per-subject accuracy was tightly clustered (83.9–86.5%), suggesting stable performance across our five participants; however, five participants are far too few to support any claim of population-level generalization."*
2. **Audit the abstract and conclusion end-to-end** and make every claim consistent with the Limitations. Specifically check: the abstract's "clinical potential" language, and any sentence implying validated detection. LOSO on 5 subjects controls for *within-sample* subject leakage; it does **not** establish generalization to the population.

### E. The audio-fusion contribution may be mostly "detecting eating" — you must prove it isn't
**Raised by:** implied by Reviewer 4 Comments 1 & 2 taken together; this is the item most likely to sink the paper on resubmission if you don't pre-empt it. **Not explicitly stated by any reviewer — this is a gap I'm flagging so you're not blindsided.**

Your own text says the +17.4-point audio benefit comes *"primarily by differentiating chewing from grinding"* (chewing→grinding misclassification drops from 24.1% to 3.8% with audio). But Reviewer 4's whole point is that **chewing is the easy, arguably-irrelevant class** ("can be assessed by asking the patient to deactivate the device during eating").

Put those together and there's an uncomfortable possibility: **if most of the 17.4-point gain is chewing disambiguation, then once you deprioritize chewing, the audio modality may add little to the actual clinical task** (clench vs. grind vs. rest). If that's the case, "audio-EMG fusion" — your headline contribution — largely reduces to "audio detects eating," which the reviewer already said is trivial.

**Action (this is a make-or-break analysis, not optional):**
1. Recompute the audio benefit **excluding chewing** — i.e., how much does audio improve clench-vs-grind-vs-movement, and bruxism-vs-rest? Report the fusion vs. EMG-only delta *on the clinically relevant classes specifically*.
2. If audio still helps meaningfully there → you have a strong, defensible contribution. Feature it prominently.
3. If audio mostly helps chewing → be honest about it and reframe the contribution as "audio provides cheap, reliable eating rejection for a continuously-worn device," which is still useful but is a *different, smaller* claim than the current one. Either way, the current unqualified "17.4-point improvement" headline cannot stand alone.

---

## 3. TIER 2 — Important (reviewers will raise these)

### F. Add AUC / ROC (and, better, PR curves)
**Raised by:** Reviewer 3 ("I am unsure whether you have any data to provide regarding the AUC").
- You have softmax outputs, so this is low-effort/high-credibility.
- Report per-class one-vs-rest **ROC-AUC** and macro-average.
- **More important given your class imbalance: report Precision-Recall AUC (average precision).** PR curves are more informative than ROC when classes are imbalanced (and yours are 3:1, worse once rest is added). Report both; emphasize PR-AUC.
- Add a figure (ROC and/or PR curves for all classes).

### G. Confront the 90%-EMG-only prior work head-on
**Raised by:** Reviewer 4, Comment 3.
Your intro cites Sonmezocak & Kurt [10] achieving **>90% with EMG alone**. Your EMG-only is **67.6%**. As written, a reviewer concludes your fancy fusion underperforms 2021 EMG-only work. Address the gap directly:
- **Different classes:** [10] uses jaw opening / jaw clenching / rhythmical grinding / fatigue-pain — no chewing, includes fatigue/pain. Not comparable class sets.
- **Validation rigor:** you use **leave-one-subject-out** (generalizing to unseen people), which is substantially harder than within-subject validation. If [10] validated within-subject, your numbers are not directly comparable and yours is the harder test. State this explicitly.
- **Hardware:** [10] uses controlled/clinical wired EMG; your premise is *low-cost* sensors. Lower accuracy is part of the accessibility trade-off you already argue.
- **Action:** add a short paragraph (Discussion) making these three points, and if feasible add a comparison row/table. Do not hope nobody notices — Reviewer 4 already noticed.

### H. Resolve the clench-vs-grind inconsistency
**Raised by:** Reviewer 4, Comment 1 ("why did they differentiate?").
You split clenching and grinding (justified in Methods as centric vs. eccentric bruxism), then call the distinction "clinically less critical" in Limitations. Pick a coherent position:
- **Recommended:** keep the fine-grained split for *scientific* interest (report it), but ALSO report a merged **"bruxism (clench+grind)"** class for the *clinical* detection claim. This simultaneously (a) answers Reviewer 4, (b) gives you the bruxism-vs-rest detection metric from item B, and (c) removes the label-noise problem you describe (windows of a grinding episode that are physiologically clenching).
- Make Methods and Limitations tell the same story about *why* the split exists and *when* it matters.

### I. Clarify whether the Table 1 baselines use fusion (possible confound)
**Not raised by reviewers — I'm flagging it.** Table 1 compares architectures (your CNN vs. LSTM/RF/MLP); Table 2 compares modalities. It is **not clear from the text whether the Table 1 baselines receive EMG+audio fusion or EMG only.** If your method gets fusion and the baselines get EMG-only, the architectural comparison is confounded — you'd be partly measuring "fusion helps," not "our architecture helps."
- **Action:** state explicitly that all Table 1 methods use the same EMG+audio input. If they don't, fix it — give every baseline the same inputs, or the comparison is invalid.

### J. Confirm augmentation and normalization are strictly training-fold only
**Not raised by reviewers — I'm flagging it as a foolproofing check.** You apply minority-class augmentation (amplitude scaling, Gaussian noise, circular shifts) and normalize with training-set statistics. You already note normalization is training-only (good). **State explicitly that augmentation was applied only within training folds, never to validation/test.** If any augmented or normalized-with-test-stats data leaked into evaluation, the minority-class metrics are inflated. A careful reviewer will ask; answer pre-emptively.

### K. Fix the "latency" overclaim
**Not raised by reviewers.** You report 12.3 ms CPU inference and call the system real-time. But your **detection latency is bounded by the window length (≥1 s)** plus the 0.5 s stride — you get a decision every 0.5 s with ≥1 s of signal behind it. 12.3 ms is *compute* latency, not *detection* latency.
- **Action:** distinguish "inference compute time (12.3 ms)" from "detection latency (≥1 s, set by windowing)." Otherwise "low-latency" in your title/abstract is misleading.

### L. Strengthen the subject-level statistics as primary
Reviewers 3 and 4 both flagged that N=5 is too small even for a pilot. You already handle the window-independence problem well (window-level CIs labeled descriptive; per-subject Table 4). Go one step further:
- Lead with **subject-level** results (Table 4: 85.0 ± 1.1%) as the headline statistics, and clearly demote window-level CIs to "within-sample, descriptive only." You're 80% there in the text — make it unambiguous and consistent in the abstract too.

---

## 4. TIER 3 — Clinical terminology, framing, and polish (mostly Reviewer 2)

Reviewer 2 is a bruxism/orofacial-pain domain expert. Much of the comment is "cite my community's literature," but the underlying points about terminology are legitimate and cheap to fix. **How deeply you engage depends on venue** (see §7): an engineering venue needs the terminology *corrected*; a clinical/dental venue needs the *full* citation engagement below.

### M. Update the bruxism definition to current consensus
- Your paper leans on the 2018 consensus [3]. There is a **newer 2025 consensus you should cite** (see §8): Verhoeff, Lobbezoo et al., *"Updating the Bruxism Definitions: Report of an International Consensus Meeting,"* J Oral Rehabil 2025. Notably, it **removed the "in otherwise healthy individuals" addendum** from the sleep/awake bruxism definitions — directly relevant to how you frame your clinically-confirmed-bruxer recruitment.
- State clearly that current definitions include bracing/thrusting (no tooth contact), and that your work addresses only the **tooth-contact** subset.

### N. Specify the phenotype
**Raised by:** Reviewer 2. State unambiguously and early: **awake, teeth-contact bruxism (clenching and grinding)**. Not sleep. Not bracing/thrusting. Put this in the title, abstract, and first paragraph.

### O. Soften causal bruxism→damage/TMD claims
Your abstract asserts bruxism "causes substantial dental and musculoskeletal damage" and the intro lists TMD as a consequence. Current thinking questions simple causation — bruxism's aetiology may be unrelated to its clinical consequences, and it is not straightforwardly a cause of TMD.
- **Action:** soften to associative language ("is associated with," "may contribute to"). Cite Manfredini et al. 2016 (§8) on decoupling aetiology from consequences. When discussing TMD, engage the INfORM/IADR TMD management framing Reviewer 2 requested (§8).

### P. Add the biopsychosocial model
**Raised by:** Reviewer 2. One or two sentences in the intro/discussion acknowledging bruxism as multifactorial (biological, psychological, social) rather than a purely mechanical dental phenomenon. Cite the consensus/Manfredini work (§8).

### Q. Engage the measurement-fluctuation literature (Colonna & Bracci)
**Raised by:** Reviewer 2. This connects to your ecological-validity limitation: awake bruxism naturally fluctuates over the day, and single-session lab emulation misses that temporal structure. Discuss in Limitations/Future Work, citing the Bracci/Colonna EMA and smartphone-diary work (§8). You already cite Colonna [5] (the 24 h device) — extend it.

### R. Engage the terminology/education papers
**Raised by:** Reviewer 2 (Mungia, Näsänen, Sangalli). Add a sentence in the intro/discussion acknowledging the documented need for standardized bruxism terminology and the field's push for phenotype specification, citing these (§8). This also *demonstrates* you took Reviewer 2 seriously, which matters if any of the same community reviews the resubmission.

### S. Section ordering (IMRaD) — venue-dependent
**Raised by:** Reviewer 3. Your Methods currently come *after* Discussion (§6) — that's *Scientific Reports*/Nature house style. For most IEEE/Elsevier venues, **Methods precede Results**. Reorder to match the target journal's format. Also (Reviewer 3) tighten Methods for reproducibility — make it possible to reimplement without the code.

### T. Strengthen data/code availability
**Not raised by reviewers, but increasingly a reviewer/editor sticking point.** "Available from the corresponding author on reasonable request" is now viewed skeptically at many venues. If IRB permits, deposit de-identified data and the analysis code in a public repository (e.g., a DOI-minting archive) and cite it. At minimum, be ready for an editor to require it.

---

## 5. New analyses to run (consolidated checklist)

Everything you need is in the data you already have — no new collection required for these:

- [ ] **Retrain without chewing** → report 3-class accuracy/precision/recall/F1 (the honest headline). [item A]
- [ ] **Add rest class from baseline recordings** → retrain; report 5-class + binary (bruxism vs. non-bruxism) + ternary. [item B]
- [ ] **Merged bruxism (clench+grind) analysis** → report bruxism-vs-rest and bruxism-vs-chewing detection. [items B, H]
- [ ] **Audio-benefit recomputation excluding chewing** → fusion vs. EMG-only delta on clinically relevant classes only. **This is the critical one.** [item E]
- [ ] **ROC-AUC (macro + per class) and PR-AUC/average precision** → add figure. [item F]
- [ ] **Direct comparison framing vs. Sonmezocak & Kurt [10]** → table row and/or paragraph. [item G]
- [ ] **Confirm baselines use identical EMG+audio inputs** → re-run if not. [item I]
- [ ] **Confirm augmentation is training-fold only** → verify in code, state in text. [item J]
- [ ] **Subject-level statistics as primary** in abstract + results. [item L]

---

## 6. Section-by-section rewrite checklist

- [ ] **Title** — add "awake" and "teeth-contact" (e.g., reflect that this is teeth-contact bruxism / clenching & grinding). Consider softening "Detection" if you adopt the classification framing. [items A, N]
- [ ] **Abstract** — honest headline number (post-chewing-removal / with rest); "proof-of-concept" not "clinical potential" as a claim; phenotype specified; causal-damage language softened; every number consistent with Limitations. [items A, B, D, N, O]
- [ ] **Introduction** — updated definition (2025 consensus); phenotype specified; bracing/thrusting acknowledged as out of scope; biopsychosocial model; softened TMD causation; terminology/education citations. [items M, N, O, P, R]
- [ ] **Results** — lead with clinically relevant metric; 4-class demoted to secondary; rest-inclusive results; AUC/PR added; audio benefit reported on clinically relevant classes, not just overall. [items A, B, E, F]
- [ ] **Discussion** — reframe contribution honestly (esp. audio, per item E); confront [10]; fix clench/grind consistency; add fluctuation/EMA discussion; distinguish compute vs. detection latency. [items E, G, H, K, Q]
- [ ] **Limitations** — emulated-not-authentic stated plainly; rest-data-vs-deployment mismatch; N=5 too small even for pilot (agree with reviewers, don't minimize); make sure nothing elsewhere contradicts this section. [items B, C, L]
- [ ] **Conclusion** — delete the LOSO over-generalization sentence; align with Limitations. [item D]
- [ ] **Methods** — reorder if venue requires; clarify baseline inputs; state augmentation/normalization are training-only; tighten for reproducibility. [items I, J, S]
- [ ] **Data/Code availability** — strengthen toward public deposit if IRB allows. [item T]

---

## 7. Where to submit (and how it changes the work)

Given this is a sensing/ML proof-of-concept, an **engineering/biomedical-signal venue** weights your actual contribution correctly and is more forgiving of the clinical-terminology objections. Candidates you already cite in adjacent work:
- *IEEE Journal of Biomedical and Health Informatics* (you cite Castroflorio there — [9])
- *IEEE Sensors Journal* / *IEEE Transactions on Neural Systems and Rehabilitation Engineering (TNSRE)*
- *Biomedical Signal Processing and Control* (Elsevier)
- *Sensors* (MDPI) — fast, and you already cite several *Sensors* papers ([6], [19])

**Critical caveat:** an engineering venue is more lenient on *terminology*, but will hit you *just as hard* on the inflated metrics, the missing rest class, the emulated events, and the audio-benefit question. **Tier 1 and Tier 2 fixes are non-negotiable regardless of destination.** The only thing that flexes by venue is the *depth* of Tier 3 (Reviewer 2) clinical-citation engagement.

**If you instead target a dental/TMD journal** (e.g., *Journal of Oral Rehabilitation*, *Cranio*), you must do the *full* Tier 3 engagement and you'll face *more* Reviewer-2-style scrutiny, not less. Only go this route if you substantially strengthen the clinical framing.

---

## 8. Citation pack (verified references for Reviewer 2's requests)

These are real, verified citations (confirmed via lookup, not from memory). Use them to address Reviewer 2 regardless of venue, and to modernize your definition section.

**Current consensus definition (add — supersedes/updates your ref [3]):**
- Verhoeff MC, Lobbezoo F, Ahlberg J, et al. *Updating the Bruxism Definitions: Report of an International Consensus Meeting.* Journal of Oral Rehabilitation. 2025. doi:10.1111/joor.13985. *(Output of the INfORM/IADR bruxism consensus meeting, March 2024; removed "in otherwise healthy individuals" from the definitions.)*
- (Optional) "Five years after the 2018 consensus definitions of sleep and awake bruxism: an explanatory note." J Oral Rehabil. 2024;51:623–624.

**Bruxism as "medical gateway" editorial (Reviewer 2 gave this one directly):**
- Cranio. 2024;42(3):251–252. doi:10.1080/08869634.2024.2320624. *(Volume/issue/pages verified; confirm the exact title from the DOI before inserting.)*

**Aetiology decoupled from consequences / not a simple TMD cause (for softening causal claims + biopsychosocial framing):**
- Manfredini D, De Laat A, Winocur E, Ahlberg J. *Why not stop looking at bruxism as a black/white condition? Aetiology could be unrelated to clinical consequences.* Journal of Oral Rehabilitation. 2016;43:799–801.

**Terminology / education papers (Reviewer 2's Mungia / Näsänen / Sangalli list):**
- Mungia R, Lobbezoo F, Funkhouser E, Glaros A, Manfredini D, Ahlberg J, et al. *Dental practitioner approaches to bruxism: preliminary findings from the National Dental Practice-Based Research Network.* Cranio. 2025;43:480–488. doi:10.1080/08869634.2023.2192173.
- Näsänen J, Karaharju-Suvanto T, Lobbezoo F, Verhoeff MC, Lappalainen OP, Nykänen L. *Self-assessed competence in relation to bruxism among undergraduate dental students in Finland.* Cranio. 2025 (final 2026;44:160–170). doi:10.1080/08869634.2025.2472085.
- Sangalli L, Sawicki C, Fricton J, et al. *Current status of temporomandibular disorders education in U.S. predoctoral dental curricula: a nationwide survey.* Cranio. 2025:1–12. doi:10.1080/08869634.2025.2505784. *(Also: Sangalli L. "Evolving progress in temporomandibular disorder education for predoctoral dental programs," Cranio 2025 editorial, doi:10.1080/08869634.2025.2543482. Multiple Sangalli 2025 Cranio papers exist — pick the one matching your point.)*

**Measurement strategy / natural fluctuation / ecological momentary assessment (Reviewer 2's Colonna & Bracci request):**
- Bracci A, Lobbezoo F, Colonna A, et al. *Research routes on awake bruxism metrics: implications of the updated bruxism definition and evaluation strategies.* Journal of Oral Rehabilitation. 2023;51(1):150–161.
- Bracci A, Lobbezoo F, Häggman-Henrikson B, et al. *Current knowledge and future perspectives on awake bruxism assessment: expert consensus recommendations.* Journal of Clinical Medicine. 2022;11:5083.
- Colonna A, Manfredini D, Bracci A, et al. *The determination of patient-based experiences with smartphone-based report of awake bruxism using a diary.* Clinical Oral Investigations. 2025;29(1):40.
- (You already cite Colonna et al. [5], the 24 h EMG device — extend that thread.)

**INfORM/IADR TMD management (Reviewer 2's request):** Reviewer 2 refers to the International Network for Orofacial Pain and Related Disorders Methodology (INfORM, within IADR). Engage their TMD standard-of-care recommendations when you discuss possible TMD consequences. *Locate the specific current INfORM/IADR TMD management consensus paper and cite it directly — I did not fully verify a single canonical citation for this one, so confirm it yourself rather than inserting a guessed reference.*

> **Honesty note on citations:** every reference above had its key bibliographic details verified except where explicitly flagged ("confirm the exact title," "confirm it yourself"). Do not paste any citation into the manuscript without a final check against the DOI — reviewers do check, and a wrong reference undercuts the credibility you're trying to rebuild.

---

## 9. The one thing this plan cannot fix: N=5

Both Reviewers 3 and 4 flagged that **five participants is too few even for a pilot**, and they're right. Every item above manages the *claims* around this limitation; none of them changes the underlying fact. The single biggest lever on how strong this paper can ever be is **more participants.**

- If you can collect even **5–10 more** subjects, the paper stops being a "feasibility gesture" and becomes something a reviewer must engage with on its merits. This would do more for acceptance than all of Tier 3 combined.
- If new collection is genuinely impossible, then commit *fully* to the modest proof-of-concept framing (§1) and target an engineering venue (§7) that will accept a small-N feasibility study *provided the claims are honestly sized.*

Be realistic: with N=5 and emulated events, the ceiling on this paper is "credible proof-of-concept in a solid specialist venue," not a high-impact clinical result. That's a perfectly publishable outcome — but only if the framing matches it.

---

## 10. Reviewer-response traceability matrix

Even for a new venue, map every prior comment to an action — it forces completeness and is useful if any reviewer overlaps. ("—" = no manuscript change; empty praise or positive comment.)

| # | Reviewer | Comment (summary) | Action | Plan item |
|---|----------|-------------------|--------|-----------|
| R1 | 1 | Praise, no substance | — | — |
| R2-1 | 2 | Relies on outdated bruxism=tooth-contact; include bracing/thrusting; retitle | Reframe as teeth-contact bruxism; update title/text | §1, N, M |
| R2-2 | 2 | Vague phenotype; specify; "medical gateway" framing | Specify awake teeth-contact; cite Cranio 2024 editorial | N, O, §8 |
| R2-3 | 2 | Biopsychosocial model | Add to intro/discussion | P, §8 |
| R2-4 | 2 | INfORM/IADR TMD management for TMD consequences | Soften TMD causation; engage INfORM/IADR | O, §8 |
| R2-5 | 2 | Cite Mungia / Näsänen / Sangalli terminology papers | Add citations + a sentence | R, §8 |
| R2-6 | 2 | Colonna/Bracci fluctuation/home measurement | Discuss in limitations/future work | Q, §8 |
| R2-7 | 2 | Reappraise device validity given above | Reframe validity claims to proof-of-concept | §1, item E |
| R3-1 | 3 | N too small even for a pilot | Acknowledge; ideally add subjects | §9, L |
| R3-2 | 3 | Methods before Results; clearer/reproducible | Reorder per venue; tighten Methods | S |
| R3-3 | 3 | Provide AUC | Add ROC-AUC + PR-AUC + figure | F |
| R3-4 | 3 | Figures good (esp. Fig. 7) | — (keep) | — |
| R3-5 | 3 | References/style fine | — | — |
| R4-1 | 4 | Class choice inflates metrics (chewing); why split clench/grind? | Report w/o chewing; merged bruxism class; justify or collapse split | A, H |
| R4-2 | 4 | Scores are classification not detection; no resting class | Add rest; report detection framing | B |
| R4-3 | 4 | Compare to 90% EMG-only [10] | Add comparison + explain gap (LOSO, classes, hardware) | G |
| R4-4 | 4 | LOSO over-generalization sentence | Delete/rewrite; align abstract+conclusion | D |
| R4-5 | 4 | Events emulated, possibly overplayed | State plainly; discuss bias direction | C |
| NEW-1 | — | Audio benefit may be mostly chewing (eating) detection | Recompute audio benefit excluding chewing | E |
| NEW-2 | — | Table 1 baseline input parity unclear | Confirm/equalize fusion inputs | I |
| NEW-3 | — | Augmentation/normalization leakage risk | Verify + state training-only | J |
| NEW-4 | — | "Latency" conflates compute vs. detection | Distinguish the two | K |
| NEW-5 | — | Data/code "on request" is weak | Deposit publicly if IRB allows | T |

---

## 11. Final pre-submission checklist (run this last)

- [ ] Every number in the abstract and conclusion is consistent with the Limitations section.
- [ ] The headline metric reflects the clinically relevant task (post-chewing / with rest), not the 4-class number.
- [ ] Rest class is included and the lab-rest-vs-deployment mismatch is stated.
- [ ] "Emulated by confirmed bruxers" is stated; no "authentic/genuine, not simulated" language remains.
- [ ] The LOSO over-generalization sentence is gone.
- [ ] Audio-fusion benefit is reported on the clinically relevant classes, not just overall.
- [ ] AUC/PR reported with a figure.
- [ ] Sonmezocak & Kurt gap is addressed head-on.
- [ ] Clench/grind: split justified *and* merged bruxism class reported; Methods/Limitations agree.
- [ ] Baselines confirmed to use identical inputs; augmentation confirmed training-only (both stated in text).
- [ ] Title/abstract/intro specify awake, teeth-contact bruxism; definition updated to 2025 consensus.
- [ ] Causal bruxism→damage/TMD language softened.
- [ ] Every added citation checked against its DOI.
- [ ] Section order matches target venue; Methods reproducible without code.
- [ ] Data/code availability strengthened as far as IRB permits.
