# REFERENCES — Prior work for the V1 running-modulation project

Curated primary literature for natural-scene processing in Allen mouse V1 and how running
modulates it, grouped by the project's three analysis angles. Findings below were gathered
by a multi-source search and **adversarially fact-checked** (23 claims confirmed / 2 refuted
across 21 primary sources). Each entry says **what to cite it for**.

> **The one-line takeaway:** natural-scene coding and running modulation are each well-studied,
> but *almost every quantitative running-modulation result is grating-only*. **No published work
> builds a running-modulated natural-scene encoding model on the Allen 2P data** — that gap is
> this project's likely novel contribution.

---

## 1. Natural-scene coding & the Allen dataset (Analysis 1, characterization)

- **de Vries et al. 2020, *Nat Neurosci* — "A large-scale standardized physiological survey reveals functional organization of the mouse visual cortex."**
  [nature.com/articles/s41593-019-0550-9](https://www.nature.com/articles/s41593-019-0550-9) · preprint [biorxiv.org/content/10.1101/359513v1](https://www.biorxiv.org/content/10.1101/359513v1)
  **THE dataset paper — cite it for your data.** ~60k neurons, 6 areas, 4 layers, 12 Cre lines, 243 mice; the exact stimulus battery (drifting/static gratings, natural scenes, natural movies, sparse noise, spontaneous). Key results: natural-scene responses are **highly sparse** (median lifetime sparseness **0.77**, excitatory); **cross-stimulus responsiveness is only weakly correlated** (natural-scene responsiveness barely predicts grating responsiveness → analyze each stimulus separately); ~34% of cells unresponsive to everything; running gives **~1.9× median evoked increase, only ~13% of cells significantly modulated**.
- **de Vries, Siegle & Koch 2023, *eLife* — Allen Brain Observatory data/overview.**
  [elifesciences.org/articles/85550](https://elifesciences.org/articles/85550)
  Cite for **methods**: documents the stimulus set and the awake, head-fixed **spinning-disk running** + pupil recording setup.
- **Siegle et al. 2021, *Nature* — "Survey of spiking in the mouse visual system reveals functional hierarchy."**
  [nature.com/articles/s41586-020-03171-x](https://www.nature.com/articles/s41586-020-03171-x)
  The **Neuropixels** companion to the 2P survey (~100k units). Cite as a spiking cross-check; note 2P calcium imaging under-reports responses vs ephys.
- **Stringer, Pachitariu, Steinmetz, Carandini & Harris 2019, *Nature* — "High-dimensional geometry of population responses in visual cortex."**
  [nature.com/articles/s41586-019-1346-5](https://www.nature.com/articles/s41586-019-1346-5)
  Cite for **population geometry** of natural-image responses: eigenspectrum is a power law with exponent **≈1.04**, near the max compatible with a smooth code. (Caveat: a 2024–25 cvPCA critique refines the tail, not the qualitative conclusion.)
- **Froudarakis, Berens, Ecker, Cotton, Sinz, Bethge & Tolias 2014, *Nat Neurosci* — "Population code in mouse V1 facilitates readout of natural scenes through increased sparseness."**
  [nature.com/articles/nn.3707](https://www.nature.com/articles/nn.3707)
  **Co-authored by our PI (Berens).** Natural scenes drive a **sparser, more readable** code, and crucially the sparsening is **state-dependent — present in active/aroused (running/whisking) but NOT quiet wakefulness.** This directly predicts a *natural-scene-specific* running signature. (Used natural **movies**, not static images.)
- **Rikhye & Sur 2015, *J Neurosci* — "Spatial correlations in natural scenes modulate response reliability in mouse visual cortex."**
  [jneurosci.org/content/35/43/14661](https://www.jneurosci.org/content/35/43/14661)
  Cite for **reliability**: natural scenes are **more trial-reliable than gratings** (driven by within-image spatial correlations), organized into ensembles — favorable for per-image encoding. (Natural movies.)
- **"Natural images are reliably represented by sparse and variable populations of neurons in visual cortex," 2020, *Nat Commun*.**
  [nature.com/articles/s41467-020-14645-x](https://www.nature.com/articles/s41467-020-14645-x)
  Per-image natural-scene coding is **sparse and trial-variable** — the empirical basis for a per-image-mean `f(S)`.

## 2. Locomotion / running modulation (Analysis 2 — mostly grating-only)

- **Niell & Stryker 2010, *Neuron* — "Modulation of visual responses by behavioral state in mouse visual cortex."**
  [pmc.ncbi.nlm.nih.gov/articles/PMC3184003](https://pmc.ncbi.nlm.nih.gov/articles/PMC3184003/)
  Foundational: running **>2× evoked rate** (2.9→8.2 sp/s), acting as a **gain modulator that preserves orientation tuning/selectivity**. Gratings, ephys.
- **Dadarlat & Stryker 2017, *J Neurosci* — "Locomotion enhances neural encoding of visual stimuli in mouse V1."**
  [jneurosci.org/content/37/14/3764](https://www.jneurosci.org/content/37/14/3764)
  **The reference for the additive-vs-multiplicative decomposition.** Modulation is **mixed**: ~**38%** of cells multiplicative (gain ≈1.5, strongest in L2/3) + ~**27%** additive; single-cell MI +47%; grating decoding error −32% (direction) / −44% (orientation). Gratings, ephys.
- **Vinck, Batista-Brito, Knoblich & Cardin 2015, *Neuron* — "Arousal and locomotion make distinct contributions to cortical activity patterns and visual encoding."**
  [cell.com/neuron/fulltext/S0896-6273(15)00252-4](https://www.cell.com/neuron/fulltext/S0896-6273(15)00252-4)
  Cite when discussing arousal vs locomotion — they are **dissociable** (relevant since Allen also gives pupil).
- **Christensen & Pillow 2022, *Nat Commun* — "Reduced neural activity but improved coding in rodent higher-order visual cortex during locomotion."**
  [nature.com/articles/s41467-022-29200-z](https://www.nature.com/articles/s41467-022-29200-z)
  The paper behind the **AIBSmouselocomotion** repo; **uses the Allen 2P dataset**. Cite for running-modulation on our exact data.
- **Liska, Rowley et al. (Yates), *eLife* reviewed preprint — running modulates primate & mouse V1.**
  [elifesciences.org/reviewed-preprints/87736](https://elifesciences.org/reviewed-preprints/87736)
  The paper behind the **V1Locomotion** repo; additive + multiplicative running modulation and the gain-model machinery Analysis 3 ports.
- **iScience 2025 — feature-specific locomotion enhancement across visual areas.**
  [sciencedirect.com/science/article/pii/S2589004225026562](https://www.sciencedirect.com/science/article/pii/S2589004225026562)
  Recent: locomotion enhancement is **stimulus-feature- and area-specific**, not a uniform cortex-wide gain.

## 3. Encoding-model methodology for natural images (Analysis 3)

- **Li et al. 2023 — "V1T: large-scale mouse V1 response prediction using a Vision Transformer."**
  [arxiv.org/pdf/2302.03023](https://arxiv.org/pdf/2302.03023)
  **The closest template:** predicts per-neuron mouse-V1 **natural-image** responses and folds **running speed** (+ pupil) in as a **behavioral input**. This is the design pattern for a running-modulated natural-scene model.
- **Cadena, Denfield, Walker, Gatys, Tolias, Bethge & Ecker 2019, *PLoS Comput Biol* — "Deep convolutional models improve predictions of V1 responses to natural images."**
  [journals.plos.org/ploscompbiol/…pcbi.1006897](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1006897)
  The **feature-based** alternative to a per-image-mean `f(S)` (CNN features). Macaque, but methodologically the template; same Tübingen-lineage authors.
- **Large-scale system identification with behavioral-state inputs.**
  [pmc.ncbi.nlm.nih.gov/articles/PMC6948932](https://pmc.ncbi.nlm.nih.gov/articles/PMC6948932/)
  Additional methods reference for predicting population natural-image responses with behavioral modulators.

---

## Cite-X-for-Y quick map

| For… | Cite |
|---|---|
| The dataset itself | de Vries 2020; de Vries/Siegle/Koch 2023 (methods) |
| Natural scenes are sparse / selective (Analysis 1) | de Vries 2020; Froudarakis 2014 |
| Analyze stimuli separately (weak cross-stimulus correlation) | de Vries 2020 |
| Natural scenes more reliable than gratings | Rikhye & Sur 2015 |
| Population geometry / dimensionality | Stringer 2019 |
| Running ≈ additive + multiplicative gain (Analysis 2/3) | Dadarlat & Stryker 2017; Niell & Stryker 2010 |
| Running preserves tuning/selectivity | Niell & Stryker 2010; Dadarlat & Stryker 2017 |
| Running modulation on the Allen 2P data | Christensen & Pillow 2022; de Vries 2020 |
| Encoding model with running as an input (Analysis 3) | Li et al. 2023 (V1T); Cadena 2019 |
| Arousal vs locomotion | Vinck 2015 |

## Caveats (verified during fact-checking)

1. **Grating-only literature.** Niell & Stryker, Dadarlat & Stryker, and the Allen 1.9×/13% figure are all **gratings** (mostly ephys). Their 38%/27%, gain 1.5, MI +47%, and 32–44% decoding numbers must **not** be presented as natural-scene results.
2. **Movies vs static images.** Froudarakis 2014 and Rikhye & Sur 2015 used natural **movies**; Allen `natural_scenes` are **static images** — a real stimulus mismatch when generalizing.
3. **2P under-reporting.** Calcium imaging sparsifies responses vs ephys, so absolute responsive fractions (34% "none", 13% running-modulated) are modality-dependent; comparative conclusions are more robust.
4. **Two claims were refuted:** running is **not purely multiplicative** (it is mixed additive+multiplicative), and cross-stimulus reliability is **weakly correlated, not independent**.

## The gap = the project's contribution

No located reference measures running modulation for **natural scenes vs gratings on the same neurons**, nor builds a **running-modulated natural-scene encoding model** on the Allen 2P data. Froudarakis's *state-dependent* sparsening even predicts running might modulate natural-scene coding differently from gratings. With ~47 matched cells (and ~34% baseline-unresponsive, ~13% running-modulated), restrict to stimulus-responsive cells (`p_ns`/`reliability_ns` in `neurons_metadata.csv`) and expect effects in a minority.
