---
layout: post
title: "One Fighter, One Timeline: A Deterministic Plan for Bootstrapping the Strive Classifier"
date: 2026-07-21
description: "The previous post sketched the architecture. This one commits to a plan: pick a single fighter, borrow ground-truth labels from community mods, build the smallest end-to-end pipeline that runs, and make the whole project falsifiable with one litmus test — does the model's timeline match the mod's timeline? Includes Giovanna's complete movelist, the clapperboard sync trick, mod install steps, and a full brainstorming session on the JSON schema."
---

The last post sketched the architecture: frames → classifier → decoder → JSON timeline → reports, with labels coming from game memory and synced by a burned-in frame counter. Sketches are cheap. This post is the opposite — a plan I could actually start executing this week, deterministic enough that "what do I do next" is never a matter of mood.

The spine of the whole thing is a single decision I'll keep coming back to: **one fighter, one timeline, one litmus test.** Pick Giovanna. Get her moves labeled for free from the mods. Build the thinnest pipeline that produces a timeline end to end. And judge the entire enterprise by one question — does the timeline my model produces match the timeline the mod produces? Everything below serves that.

## The target vocabulary: Giovanna's complete movelist

Before any of this, the classifier needs a class list, and the class list is just Giovanna's movelist. Here it is in full, pulled from Dustloop and the Guilty Gear Wiki command list. These names are the ground truth — the mod reports move IDs, and this is what those IDs mean in human terms.

**Normals**

| Notation | Input | Note |
| --- | --- | --- |
| 5P | Punch | Fast jab |
| 5K | Kick | Standing kick |
| c.S | Slash (close) | Close standing slash |
| f.S | Slash (far) | Far standing slash, key poke |
| 5H | Heavy Slash | Standing heavy |
| 2P | Down + Punch | Crouch jab |
| 2K | Down + Kick | Low kick |
| 2S | Down + Slash | Long disjointed low poke |
| 2H | Down + Heavy Slash | Crouch heavy |
| 2D | Down + Dust | Sweep / knockdown |
| 5D | Dust | Overhead Dust attack |
| j.P | Air Punch | Air jab |
| j.K | Air Kick | Air kick |
| j.S | Air Slash | Air slash |
| j.H | Air Heavy Slash | Air heavy |
| j.D | Air Dust | Air Dust attack |

**Command normals**

| Notation | Input | Note |
| --- | --- | --- |
| 6P | Forward + Punch | Upper-body invuln anti-air |
| 6K | Forward + Kick | Advancing command normal |
| 6H | Forward + Heavy Slash | Command heavy |

**System / universal mechanics**

| Notation | Input | Note |
| --- | --- | --- |
| Wild Assault | 236D | Universal Wild Assault (character-specific animation) |
| Ground Throw | 4/6 + Dust (close) | Forward or back throw |
| Air Throw | j.4/6 + Dust (close) | Air throw |
| Meter Boost | (passive install) | REI glows green at 50% Tension, Giovanna at 100%; attack/defense ramp up, normals deal chip |

**Special moves**

| Name | Notation | Description |
| --- | --- | --- |
| Sepultura | 214K | Slides forward for a high-angle back kick; REI follows the curve to spin the opponent out. |
| Trovão | 236K | Handstand, then flings herself forward feet-first, pushing the opponent back. |
| Sol Nascente | 623S | Reverse cartwheel kick on the spot; REI circles behind, pushing the opponent back. |
| Sol Poente | 214S (air OK) | Leaps high, reverse kick into vertical splits, mild ground bounce. |
| Chave | 214H | Low-profile command dash that can transition into the other specials, enhancing them. |
| Chave → Sepultura | 4K during Chave | Enhanced Sepultura; opponent spins out closer, more damage. |
| Chave → Trovão | 6K during Chave | Enhanced Trovão; instant stun on hit. |
| Chave → Sol Nascente | 6S during Chave | Enhanced Sol Nascente; launches the opponent. |
| Chave → Sol Poente | 4S during Chave | Enhanced Sol Poente; Giovanna lands closer after the ground bounce. |

**Overdrives**

| Name | Notation | Description |
| --- | --- | --- |
| Ventania | 632146H | Rushing helicopter kick; invincible on startup, knocks down on the last hit, Area Shift in the corner. |
| Tempestade | 236236H (air only) | Diagonal dive kick; on connect, an 8-hit volleyball barrage ending in Sol Nascente. Area Shift in the corner. |

That's the full class set. Counting normals, command normals, the Chave enhancements, throws, and overdrives, it's roughly three dozen labels — but note how many are visually distinct trajectories rather than single poses. That's the argument the mod is going to settle: how many of these a single-frame classifier can actually separate before I reach for temporal models.

(One correction to my own earlier notes: the placeholder move names I've been scribbling in drafts don't match the real kit. The canonical names are the ones above — Sepultura, Trovão, Sol Nascente, Sol Poente, Chave, Ventania, Tempestade. I'm standardizing on these so the label vocabulary matches what the mod and Dustloop actually emit.)

## The bootstrapping mechanism, restated: the clapperboard

None of the labels above are worth anything if they're misaligned with the pixels. So the load-bearing trick, restated cleanly.

Two independent timelines are the enemy. The game simulates at a clean 60Hz; each tick has a monotonic game-frame number, and the mod can read, for that tick, exactly which move each player is in and which animation frame within the move. Meanwhile OBS capture is a *separate* pipeline that drops frames, duplicates frames, and drifts. "Video frame 1000" is not "game frame 1000" after a few minutes, and any code that assumes a fixed 1:1 mapping silently poisons every label.

The fix refuses the alignment problem instead of solving it. The mod renders the game's frame counter *into* the frame itself, so each captured image carries its own ground-truth identity. Concretely:

<u>The key is a small patch of pixels in a fixed corner — a little grid of light/dark cells that encodes the current game-frame number in binary — and that patch is what links each video frame to its exact entry in the mod's state log.</u>

Read the number back off the patch, index the mod's log at that number, and you have a perfect label. It's a barcode, not a single dot — one pixel can only carry one bit, and you need many bits to name a frame, so the number is spread across fat cells (16×16 px, luma-encoded, with a checksum) so it survives H.264 compression. Because this is a training-time-only tool on my own PC, I can also just capture at a very high bitrate and make the read trivial.

The payoff is that every capture pathology degrades gracefully. A dropped frame makes the counter jump — I simply have no image for those ticks. A duplicated frame repeats the counter — dedupe it. A garbled read fails the checksum — drop that one frame. **Every failure mode costs one training example; none of them produces a wrong one.** And unlike a film clapperboard, which syncs once at the head of the take and trusts constant rates thereafter, this counter is drawn every frame — a per-frame heartbeat that re-establishes lockstep 60 times a second, so drift never accumulates.

## Installing the mods (and forking one to log state)

The plan assumes I'm reading game state from community tooling rather than writing a memory scanner. Two projects matter, both free and open-source.

**strivehitboxes (Altimor).** An injected DLL hitbox viewer. Installation is deliberately simple: download the release, put `strivehitboxes.dll` next to `striveinjector.exe`, start the game, then run `striveinjector.exe`. The DLL lives inside the game process and draws hurtboxes (green), counterhit-state hurtboxes (cyan), hitboxes (red), and pushboxes (yellow) over the running game. The value for me isn't the boxes — it's the proof that hitbox, hurtbox, and counterhit state are all readable every frame from a source I can audit on GitHub.

**StriveFrameViewer (Procdox, plus active forks).** An SF6-style frame-data display for Training/Replay mode. Standalone install: download `Standalone.zip` from releases and extract into the game's `win64` directory — typically `...\steamapps\common\GUILTY GEAR STRIVE\RED\Binaries\Win64`. On upgrade, delete the old `cache` folder or it'll misbehave. It renders per-frame startup/active/recovery/hitstun bars for both players, which is exactly the per-frame player state I want to log.

**The fork I actually need.** Neither mod, out of the box, writes a labeled dataset. The one evening of work is forking whichever project exposes the cleanest state (StriveFrameViewer's per-frame state is the closer fit) to do two things: (1) append one row per game tick to a log file — `{game_frame, p1_move_id, p1_anim_frame, p1_flags, p2_move_id, p2_anim_frame, p2_flags}` — and (2) draw the frame-counter patch in the corner. Both happen inside the same frame callback, which is the whole reason the counter is a perfect key into the log.

**One hard operational rule:** these mods break on game patches, and the move IDs underneath them can shift. So every collection run pins the game version, and each patch is a hard dataset boundary. A sync scheme this precise is worthless if the label vocabulary silently reindexes under it.

## The deterministic getting-started plan

Here's the sequence I can execute without re-deciding scope every morning. Each step has a concrete output, and no step starts until the previous one produces its artifact.

**Step 0 — Pin the environment.** Record the exact GGST version and the exact mod commit hashes. Write them into a `run.json`. This is the boundary marker for the whole dataset.

**Step 1 — Instrument.** Fork the mod to (a) log per-tick state and (b) draw the frame-counter patch. Verify by eye that the number in the corner increments cleanly and the log rows line up with what you're doing in Training Mode.

**Step 2 — Capture one clean session.** Giovanna mirror, Training Mode, high-bitrate OBS capture, a few minutes. One video file, one state log, one `run.json`. This is the first raw dataset.

**Step 3 — Build the join.** Write the preprocessing that reads the counter off each video frame, dedupes repeats, drops checksum failures, and emits `(image, game_frame)` pairs joined against the log to `(image, move_id, anim_frame)`. Crop the counter patch out of the image before it becomes a training example so the classifier can't cheat off it. Output: a folder of labeled frames.

**Step 4 — Narrow the vocabulary for v0.** Do not train all three dozen classes yet. Pick five plus a `neutral/other` bucket — e.g. `f.S`, `2S`, `Sepultura`, `Sol Nascente`, `Ventania`, `other`. Five visually distinct moves, chosen to be easy wins so the pipeline gets proven before the hard separations.

**Step 5 — Train the smallest classifier that runs.** Single-frame CNN, pixels in, six-way softmax out. Accuracy is not the goal here; a checkpoint that runs inference is.

**Step 6 — Run the full pipeline once, end to end.** `extract → classify → decode (5-frame majority vote only) → timeline.json`. The moment this produces a timeline file, the architecture exists. Everything after is deepening, not building.

**Step 7 — Wire up eval (see the litmus test below).** This is the last piece of v0, and it's what turns "is it good?" from a feeling into a number.

The discipline that keeps this honest: **keep exactly one working pipeline alive at all times, and change one variable against it.** Never let three half-finished features coexist. The scarcest resource for a solo project isn't skill or time — it's a working baseline to measure against, and the entire method is refusing to break it. After v0, the next task is never chosen by ambition; it's read off the eval as the largest, cheapest-to-fix error source.

## Analyzing results: reading the eval instead of guessing

When the pipeline runs, the temptation is to watch it render a pretty timeline and feel good. Resist that — the useful signal is in the diff against ground truth, and it comes in a few specific shapes:

- **Confusion between two moves.** If `f.S` and `2S` swap constantly, that's a data/label problem for those two classes, not a decoder problem. Fix: more examples of the confused pair, or check whether the mod's move IDs for them are what you think.
- **Onset lag.** If moves are labeled correctly but start three frames late, that's the decoder's hysteresis, not the model. Tune the "starts after N agreeing frames" threshold on the *cached* probability stream — no GPU, no re-decoding.
- **Flicker.** Single-frame wrong predictions in the middle of a move mean the smoothing window is too short. Widen it.
- **Missing short moves.** If quick jabs vanish, the "no 2-frame moves" sanity rule is too aggressive.

Each symptom points at exactly one stage. That's the payoff of the pipeline being composable — a bad number is diagnosable rather than mysterious, and the fix touches one subcommand.

## Brainstorming session: designing the JSON timeline

This deserves its own session because the schema is the most expensive thing to get wrong. It's the contract between every stage and the evaluation target, and — the McLuhan worry from the last post — whatever fields it encodes silently become the vocabulary of everything downstream. So let me actually think it through rather than pattern-match to the obvious.

**Question 1: what is the canonical time axis — video frames or game frames?**

Tempting to use video frames since that's what the classifier sees. Wrong. The clapperboard exists precisely to convert video frames into game frames, and game frames are the axis the mod, the frame data, and every future analysis live in. So the timeline's canonical clock is the **game frame** (60Hz, monotonic). I'll optionally carry a `video_frame` on each event for debugging the join, but nothing downstream should depend on it.

**Question 2: one events list or two (one per player)?**

Two lists feels natural, but a single flat `events` list with a `player` field is better. Every interesting analysis — punishes, whiff-punishes, exploitability — is about the *relationship* between the two players' events on a shared clock. A single sorted list makes "what was P2 doing while P1 recovered" a windowing operation instead of a merge. So: one list, `player` field, sorted by `start`.

**Question 3: how do I represent neutral / nothing-happening?**

Two options: explicit `neutral` events filling every gap, or sparse events with neutral implied by absence. Explicit neutral doubles the file size and bakes a specific idea of "neutral" into the schema forever. Sparse wins — events are discrete actions, and the gaps between them *are* neutral. If I later need "time spent in neutral" it's derivable.

**Question 4: how much of the move's internal structure do I store?**

The bare version is `{move, start, end}`. But Layer 3 (reactions) needs precise phase timings, and early detection needs startup/active/recovery boundaries. Storing full per-frame phase data is overkill in the event object. Compromise: each event carries optional phase boundary frames — `startup_end`, `active_end` — so the recovery window is derivable (`active_end → end`) without a per-frame blob. Present when the mod knows them, absent when a model guesses.

**Question 5: what does an event need to serve all three feedback layers?**

- Layer 1 (provably suboptimal) needs move identity, result, and enough timing to check "was this punishable and did I punish it" — covered by `move`, `result`, and phase boundaries.
- Layer 2 (exploitability) needs habit *frequencies*, which are aggregates over many events — so the timeline doesn't store distributions, it stores the raw events that distributions are computed from. Keep the timeline raw; compute frequencies downstream.
- Layer 3 (reactions) needs precise onsets, which is just `start` at frame resolution. Covered.

The lesson: the timeline stays *raw and per-event*. Derived quantities (punish counts, DP frequency, median latency) are computed by the rules and coaching layers, never stored in the timeline. That keeps the schema small and keeps the derivations swappable.

**Question 6: how does the same schema serve both ground truth and predictions?**

This is the one that makes the litmus test possible. A mod timeline and a model timeline must be the *same shape* so evaluation is a diff. So every event carries a `confidence` (1.0 for mod ground truth, the softmax value for predictions), and the top-level metadata carries a `source` (`"mod"` or `"model:cnn_v3"`). Same schema, two producers, one comparator.

**Landing on v0.1.** Metadata block for provenance and versioning, plus a flat sorted event list:

```json
{
  "schema": "0.1",
  "meta": {
    "source": "mod",
    "game_version": "1.44",
    "character": "Giovanna",
    "fps": 60,
    "video": "mirror_session_01.mp4",
    "run_id": "2026-07-21-run-03"
  },
  "events": [
    {
      "player": 1,
      "move": "f.S",
      "start": 340,
      "end": 358,
      "startup_end": 348,
      "active_end": 351,
      "result": "whiff",
      "confidence": 1.0
    },
    {
      "player": 2,
      "move": "Sepultura",
      "start": 352,
      "end": 384,
      "startup_end": 364,
      "active_end": 370,
      "result": "hit",
      "confidence": 1.0
    }
  ]
}
```

Design commitments carried out of this session: canonical clock is the game frame; events are sparse and flat with a `player` field; the timeline stores raw events only, never derived stats; phase boundaries are optional fields; and `source` + `confidence` let one schema represent both ground truth and prediction. The `schema` version field is there from file one, evolution is additive where possible, and a breaking change gets a small migration script rather than clairvoyant up-front design. The skill isn't predicting the future — it's cheap migrations.

## The litmus test: model timeline vs. mod timeline

Here's the finish line that makes the whole project falsifiable, and it falls out of Question 6 for free. Because the mod produces a timeline and the model produces a timeline *in the same schema*, evaluation is a diff between two timelines.

The mod timeline is ground truth: the game told it exactly which move was active on which frame, so it's correct by construction. The model timeline is the pipeline's best guess from pixels alone. **Eval is: run the model over footage the mod also labeled, and diff the two.**

Scored at the event level, not raw frame accuracy:

- **Matching rule.** A predicted event matches a ground-truth event when it's the same `player`, the same `move`, and overlaps in time past an IoU threshold. Matched pair = true positive; unmatched prediction = false positive; unmatched ground-truth = false negative.
- **Per-move precision / recall / F1**, so I can see *which* moves the model nails and which it fumbles.
- **Onset latency** on matched pairs (`predicted.start − truth.start`), which is both a model-quality metric and, later, the exact machinery Layer 3 uses to measure a player's reactions.
- **Boundary IoU**, how tightly predicted start/end hug the truth.

Two things make this powerful. First, it's a single number I can watch move as I change one variable — the anti-overpromising instrument. Second, `eval` and the product are the same code path: both ingest footage, produce a timeline, and diff it against a reference. The only difference is whether the reference is the mod's ground truth (eval) or "what you should have done" (coaching). The evaluation harness *is* an early version of the product.

So the definition of done for v0 is one falsifiable sentence: *the pipeline produces an event-level timeline for a Giovanna mirror on five moves, and the diff against the mod timeline scores F1 > 0.8 with median onset error under 3 frames.* If it clears that bar, the architecture is real and the next move is adding one class or one variable. If it doesn't, the eval tells me exactly which stage to fix.

## Where this lands

The plan in one breath: pick Giovanna, borrow her labels from the mods, sync pixels to labels with a corner counter, build the five-class pipeline end to end, and judge all of it by diffing my timeline against the mod's. The mod is the grandmaster annotating for free; the classifier is what actually has to play; and the litmus test is the referee that keeps me honest. None of the pieces is big. You never build "the system" — you build the next subcommand, and then you check it against ground truth.
