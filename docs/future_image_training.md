# Future work: image-driven perception training

> **Status:** design note only. NOT implemented. Written down so we
> don't forget the plan — revisit when the Phase 1 CNN slot is
> actually ready to be filled.

## The idea (plain language)

Instead of hand-labelling obstacle datasets, the perception module
will eventually **teach itself** by:

1. **Search the open web for images** of a named concept (e.g.
   "parked car", "stop sign", "power line", "pedestrian crossing",
   "fire hydrant", "treetop").
2. **Infer the label from the search query itself** — the name the
   user typed is the class label. "parked car" → class `parked_car`.
3. **Skim the image titles / surrounding alt-text** to filter out
   images that are clearly not the named concept (CAPTCHA, memes,
   screenshots, etc.).
4. **Hand the cleaned images (plus the label) to the AI** as a mini
   dataset. The perception CNN trains on them exactly like any other
   supervised batch.

In short: user names a class → system finds images → figures out
which ones are really that thing → trains on them. No manual
bounding-box annotation.

This is the step that turns Layer 3 from "noise-simulated
perception" into a real CNN, without the weeks of labelling a
custom dataset would normally need.

## Why this belongs in the future, not today

- Phase 1 is explicitly the **noise-simulated** perception path
  (see [`modules/perception.md`](modules/perception.md) and
  [`lessons_learned.md`](lessons_learned.md)). Grade-parameterised
  noise is the whole point of the current module — it lets us
  experiment with A/B/C-grade perception *without* training a CNN.
- The project runs **fully offline** once deployed (see
  [`comms.md`](comms.md)). Scraping the web is a **build-time**
  activity — done on the developer's workstation to produce a
  dataset, never in flight.
- A CNN swap-in is the documented "still undone" item in
  [`lessons_learned.md`](lessons_learned.md). This doc is the bridge
  between "note the gap" and "here's how we'll cross it when we get
  there."

## Plan (when we pick this back up)

1. **Class list.** Start small — 8-12 classes the drone actually
   needs to recognise (the current world has obstacles, landing
   pads, no-fly markers, other drones). The list lives in
   `docs/perception_classes.md` (not yet written).
2. **Image fetch step.** A developer-only `drone-ai scrape` command
   that takes the class list, queries the web (Google Images API,
   Bing API, or an equivalent that respects ToS), and downloads N
   candidates per class into `datasets/perception/raw/<class>/`.
   Store the original URL + filename + alt text alongside each
   image so the provenance is auditable.
3. **Name-based filtering.** Use a multimodal model (Claude API or
   a local CLIP model) to answer a single question per image: "Is
   this image really an example of `<class>`?" Anything that scores
   low is moved to `datasets/perception/rejected/` with the model's
   reasoning saved next to it. **Keep the rejected folder** — we'll
   want to re-review if the model is too strict.
4. **Split.** Standard train/val/test split, stratified by class.
   Write the split manifest to `datasets/perception/manifest.json`
   so the CNN training run is reproducible.
5. **Train.** The CNN replaces the noise model in
   `modules/perception/detector.py::PerceptionAI`. Interface stays
   identical — `detect(drone_pos, world) -> List[Detection]` — so
   the rest of the stack needs zero changes. Grade the trained CNN
   on the same benchmark that currently grades the noise model
   (`PerceptionMetrics` → `score_perception`).
6. **Commit the *weights*, not the raw images.** The raw dataset
   stays out of git (big, noisy, possibly copyrighted). The trained
   `.pt` + its manifest hash is what lives under `models/perception/`.

## Hard rules that can't move

- **Nothing about this runs in flight.** Scraping + labelling +
  training is all offline, developer-workstation-only. The field
  drone only ever loads the already-trained `.pt`. This is a Phase-1
  completion gate (see [`sensors.md`](sensors.md): "NO cloud, NO
  internet after takeoff").
- **Label provenance must be auditable.** For every image in the
  final training set, `datasets/perception/manifest.json` records:
  query string, source URL, download time, the filter model's
  yes/no decision + reasoning. If a class ends up mis-learned, we
  can trace which images drove it.
- **No auto-scraping on CI.** The scrape step is a manual developer
  action. CI runs training from the committed manifest, not from a
  fresh fetch — fresh fetches across CI runs would make runs
  non-reproducible.
- **Respect robots.txt + source ToS.** If a source forbids scraping,
  we pay for an API key or pick a different source. The scrape
  helper must surface any failure to the developer, not silently
  skip.

## Open questions (decide when we pick this up)

- Which image source? Google Images API is rate-limited; Bing has
  a cleaner search API. DuckDuckGo scrapes Google but is fragile.
  Possibly all three, behind a common fetcher interface, with
  source chosen per-class.
- Which filter model? Claude API with vision is the simplest ("is
  this a <class>?"). Local CLIP is free but less reliable. Start
  with Claude for the first pass, backfill with CLIP once we know
  which classes are hard.
- How big per class? 200 images is probably enough for a first
  CNN on 8-12 classes. If accuracy plateaus below the current
  noise-simulated P-grade, scale up to 1000+.
- Do we also generate synthetic images (procedurally-rendered
  obstacles from the simulation) to augment the web-scraped set?
  Probably yes — cheap, label-perfect, closes the sim-to-real
  gap for obstacles the drone sees often.

## Not in scope here

- Self-supervised / unsupervised training from drone cam footage.
  Different problem — the drone generates unlabelled frames in
  flight, not a human typing class names. Worth a separate doc
  when the time comes.
- LLM-generated image datasets. Same concern (quality + license)
  as scraping but with worse provenance. Skip.
