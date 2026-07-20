# Frames, Decoders, and Clapperboards: Sketching the Architecture of a Strive Move Analyzer

I've written before about data collection being the hard part of my Guilty Gear Strive move classifier. This post is about what happens after that — how a pile of classified frames becomes something that can actually analyze a match. Six ideas came together recently: the image classifier, temporal models, a decoder stage, a CLI-first design, a JSON timeline as the universal interface, and a sync trick borrowed from film production.

## The image classifier: video is just a loop

The core model is an image classifier: pixels in, move probabilities out ("Ventosa: 0.91"). Running it against video is less exotic than it sounds — video is just frames, so inference is a loop. Decode the VOD, batch 64 frames into a tensor, one GPU pass per batch. An RTX 4060 chews through a 15-minute VOD at hundreds of frames per second, which is why I'm comfortable designing this as offline batch processing rather than real-time.

The output of that loop is a per-frame prediction stream. And that stream is noisy: `neutral, neutral, ventosa, ventosa, dash?!, ventosa, neutral...` — a one-frame flicker in the middle of a move is the classic failure mode. Which is where the next two ideas come in.

## Temporal models: shelved, not binned

The obvious fix for fast, blurry moves is a temporal model — feed the network a stack of 8–16 frames so motion lives inside the model. Ventosa's forward lunge is unmistakable as a *trajectory* even when individual frames are ambiguous.

I'm shelving this for now, deliberately. A lot of what looked like "the game is too fast to classify" was actually label noise: hand-labeling video means fuzzy move boundaries, and a model trained on fuzzy boundaries learns fuzzy decision surfaces. With frame-perfect labels (more on that below), the single-frame classifier deserves a fair re-test before I pay the complexity cost of 3D CNNs or CNN+LSTM stacks. Change one variable at a time.

Temporal models stay on the roadmap for the cases that are *genuinely* ambiguous at the single-frame level. But there's a cheaper tool to try first.

## The decoder: the classifier knows what things look like, the decoder knows how the game works

The prediction stream needs to become discrete events — this move, started here, ended there. That's a second stage, but it doesn't need to be another neural network. It's a **decoder**, and it can be as simple or as principled as needed:

1. **Smoothing** — majority vote over a sliding 5–7 frame window. Kills single-frame flickers. Ten lines of code.
2. **Hysteresis rules** — a move "starts" after N consecutive agreeing frames, "ends" after M frames of something else, plus sanity constraints (no 2-frame moves).
3. **HMM/Viterbi decoding** — treat per-frame probabilities as emissions and move-to-move transitions as a state machine. This is where Dustloop frame data becomes *structural*: gatling tables literally define which move sequences are possible, so illegal transitions get probability zero.

If this sounds familiar, it's the same pattern as speech recognition — an acoustic model emits per-timestep probabilities, then a decoder uses a language model to rule out impossible word sequences. I'm doing the identical thing with gatling tables instead of grammar.

The division of labor is the satisfying part: pixels on one side, rules on the other. The classifier never needs to know frame data exists; the decoder never sees a pixel.

## The JSON timeline: the medium is the message

Everything above communicates through one artifact — a JSON timeline:

```json
{
  "schema": "0.1",
  "events": [
    {"start": 340, "end": 358, "move": "Ventosa", "result": "hit"},
    {"start": 412, "end": 429, "move": "5K", "result": "blocked"},
    {"start": 501, "end": 533, "move": "Erpressung", "result": "whiff"}
  ]
}
```

This format is doing more work than it looks like. It's the contract between every stage: classifier → decoder → rules layer → coaching layer → renderer, all independently swappable because they only agree on the timeline. It's also the evaluation target — ground truth is a timeline, model output is a timeline, and evaluation is a diff between the two, scored at the *event* level (precision, recall, F1 per move, onset latency) rather than raw frame accuracy.

There's a McLuhan angle I keep thinking about: whatever fields this schema encodes becomes the vocabulary of everything downstream. PGN made chess computable, and it also shaped a century of chess thinking around what notation captures. My timeline format is me deciding, right now, what Strive analysis *can be about*. So the schema gets a version field from file one, evolves additively where possible, and gets a small migration script when it must break. The skill isn't clairvoyant design — it's cheap migrations.

## The CLI: the eval harness is the product skeleton

No desktop app yet, and maybe not for a while. The v0 is a CLI where each pipeline stage is a subcommand that reads and writes the timeline:

```
strivecoach extract vod.mp4
strivecoach classify --model cnn_v3
strivecoach events
strivecoach render --open
strivecoach eval --model cnn_v3
```

Composable stages mean cached intermediates: when I tune the decoder, I rerun one stage on a saved probability stream — no GPU, no re-decoding fifteen minutes of video. And the `eval` subcommand lives in the same tool from day one, because the evaluation harness *is* an early version of the product. Both ingest footage, produce a timeline, and render a human-readable report — the only difference is whether the comparison target is ground truth or "what you should have done."

## What the reports are actually for: three layers of feedback

All of this pipeline exists to produce reports, so it's worth being clear about what a report should *say*. Someone pointed out to me that highlighting an opponent's weaknesses is easier than offering improvement suggestions — the former feels deterministic, the latter drifts into a philosophical question about what it even means to have a "better" reaction. Strive isn't chess; there's no Stockfish handing you one number that says this move was correct.

But the dichotomy softened once I noticed two things. First, opponent-weakness detection and self-improvement are the *same analysis pointed at different players* — "blocks low after every jump-in 87% of the time" is exploitation intel about them and a leak report about me, same computation. Second, the real split isn't opponent-vs-self, it's three layers of feedback that differ by how much objective ground truth they have. The whole aim of reporting is to lead with the certain layer and be honest about the fuzzy one.

**Layer 1 — provably suboptimal (chess-like, deterministic).** Frame data is a partial oracle. A dropped punish is a mathematical fact: their move is −15 on block, I blocked it, I pressed nothing — that's a missed guaranteed punish, as objective as a hanging queen. Same for suboptimal combos (140 damage where the BnB gets 210, straight from the Dustloop database), invalid gatlings, unsafe moves left unpunished. No philosophy required, and this layer alone is a huge chunk of what separates floors. Reports should lead here: *"you missed 6 guaranteed punishes on Nagoriyuki's beyblade."*

**Layer 2 — statistically exploitable (poker-like, not chess-like).** No single decision is wrong, but a *distribution* can be: wake-up DP 60% of the time is a leak regardless of philosophy, because a competent opponent baits it. The metric isn't "correct move," it's **exploitability** — how much EV an opponent gains by best-responding to my patterns. That's computable from the timelines, it's the same math poker solvers use, and mixing up my options *is* the improvement. This layer runs identically on opponents, which is where the weakness-detection framing came from.

**Layer 3 — execution and reaction (the fuzzy one, made concrete).** "Improve your reactions" is mush until you decompose it. Punish latency — frames between an opponent's whiff and my button — is measurable per instance and trendable over time. I don't have to philosophize about what a better reaction *is*; I plot median punish latency across sessions and watch it drop. Reports expose this as trend graphs, not verdicts.

The "ultimately the goal is to win" framing is what ties them together: improvement just means raising EV, and all three layers are EV measured at different levels of certainty. Chess needed one oracle; this gets three partial ones that together cover most of the gap. That's the aim every report is built around — and it's why the JSON timeline has to be generous enough to support all three, since Layer 2 needs habit frequencies and Layer 3 needs precise timings that a bare move-list wouldn't carry.

## The clapperboard: syncing pixels to ground truth

The frame-perfect labels everything above depends on come from reading game state via mod tooling — the same family of tools behind Strive's hitbox viewers and frame data overlays. The game knows exactly which move is active on which animation frame; the mod can log it.

But there's a trap: the game simulates at 60fps, and OBS capture is a *separate* pipeline that drops frames, duplicates frames, and drifts. "Video frame 1000" is not "game frame 1000" after a few minutes, and a naive 1:1 assumption silently poisons the labels — defeating the entire frame-perfect premise.

Film production solved this problem a century ago with the clapperboard: picture and sound recorded on separate devices, synced by one sharp event visible in both — the board snapping shut on film, the crack on the audio track. My version is continuous rather than one-shot: the mod renders the game's frame counter *into* the frame, as a small pixel pattern in a corner. Every captured video frame now carries its own ground-truth frame ID. Preprocessing reads the number back (the easiest CV task in the whole project), joins against the mod's state log, and dropped or duplicated capture frames become self-evident as gaps or repeats in the counter. Crop the corner before training so the classifier can't cheat off it.

One evening of mod-side code, and the lockstep problem reduces to reading a number off a frame.

## Exploiting mod data: what game state actually buys you

It's worth spelling out exactly how memory-extracted state improves the classifier, because "better labels" undersells it. And none of this requires writing a memory scanner from scratch: the Strive community has free, open-source, battle-tested tooling for exactly this. Altimor's hitbox viewer (`strivehitboxes` on GitHub) is an injected DLL — run the injector after the game starts, and the DLL lives inside the game process with direct access to hitboxes, hurtboxes, and counterhit state, drawing them over the running game. StriveFrameViewer does the same for per-frame player state, rendering color-coded startup/active/recovery/hitstun bars for both players in Training Mode. These projects are proof that the exact state I need is readable every frame, the source is auditable on GitHub rather than a mystery binary, and forking one to log state to a file (or draw the frame counter for the clapperboard) is a far smaller job than starting cold. The mod turns labeling from the project's bottleneck into a free byproduct of playing, and it changes what the model can learn in at least five distinct ways.

**1. Frame-perfect boundaries.** Hand-labeling video means guessing where a 20-frame move begins and ends. Every guess that's off by three frames teaches the model that some neutral frames are Ventosa and some Ventosa frames are neutral — fuzzy boundaries in, fuzzy decision surfaces out. The mod reports the exact move ID and animation frame for every game tick, so the training signal is clean at the resolution the problem actually lives at. A meaningful chunk of what looked like "the game is too fast to classify" was this label noise all along.

**2. Sub-move phase labels.** Because the mod exposes the animation frame *within* a move, I can label startup, active, and recovery phases separately. That's what makes early detection possible: a model that has explicitly learned "these three frames are Ventosa startup" can commit to a call before the move finishes, instead of needing to see the whole animation. For a coaching product this matters less; for anything approaching live analysis it's the difference between feasible and not.

**3. Volume without annotation.** Hand-labeling caps the dataset at whatever my patience allows. With the mod logging state during real matches, every session becomes training data — hours of footage, thousands of labeled examples, zero manual work. Fast and blurry moves benefit most, because what they need is coverage of every messy in-between frame, and coverage is exactly what volume provides.

**4. Realistic visual conditions for free.** My original pipeline used sterile Training Mode recordings — no hitsparks, no counter callouts, no RC slowdown tinting, no "Counter!" popups over the sprite. A model trained on lab footage stumbles the first time a real match throws screen effects at it. Mod-labeled *match* footage bakes those conditions into the training set in their natural distributions, and crucially, the labels stay perfect even during moments so visually chaotic that a human annotator couldn't tell what's happening underneath the effects. Those are precisely the frames where hand-labeling fails and memory-reading doesn't care.

**5. Pose as an auxiliary signal.** The game holds Giovanna's full skeleton in memory — Unreal Engine animates from it every frame. The same memory-reading approach could extract joint positions, giving a compact, costume-invariant representation of every move. Even if the shipped model stays pixels-based (it has to — see below), pose data can serve as an auxiliary training signal that teaches the pixel model which parts of the image matter.

The critical constraint framing all of this: **the mod is a training-time tool only.** It runs on my PC, on my game version, and ships nowhere. The product analyzes arbitrary footage — YouTube VODs, tournament streams, a friend's console capture — where no memory access exists. So the division is clean: mod data is the grandmaster annotating millions of positions for free; the classifier is what actually plays. One practical caveat comes with that: these mods break on game patches, so a data collection run should pin the game version and treat each patch as a dataset boundary.

## Where this lands

The shape of the whole thing: frames → classifier (per-frame probabilities) → decoder (events) → JSON timeline → reports. Labels come from game memory, synced by a burned-in clapperboard. Temporal models wait until clean data proves they're needed. And the whole pipeline lives in a CLI whose eval mode and product mode are the same code path.

None of the pieces is big. That's the part I keep relearning: you never build "the system," you build the next subcommand.
