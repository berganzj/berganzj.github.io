 The Data Gathering Plan

Before training anything, I need footage. Here's the exact plan.

---

## The five classes

Quick correction from a previous post: the move names I had before (Ventosa, Erpressung, Seppi) were wrong — hallucinated by an AI and not Giovanna's actual moves. I cross-checked against Dustloop, the community wiki, and here are the real ones.

For this POC I'm targeting four Giovanna special moves plus a neutral class:

- **Sepultura** (214S) — forward ground lunge, visually distinct low-profile motion
- **Trovão** (214K) — rushing kick forward
- **Sol Nascente** (623S) — rising anti-air, upward arc
- **Ventania** (632146H) — reversal, explosive startup
- **Neutral** — walking, blocking, idle

Punches, kicks, and normals are deferred. Too fast and too similar to distinguish from a single frame.

---

## Step 1: Command list bootstrap

GGS has a command list that plays a clean animation for every move. No opponent, no health bars, no noise — just the move.

I'm screen recording each of the four specials from the command list on loop for 2–3 minutes each. This gets me a labeled dataset for free before touching Training Mode.

---

## Step 2: Training Mode sessions

For anything the command list doesn't cover cleanly, I'll use Training Mode. Set the dummy to record/playback a specific move on loop, hit record in OBS, walk away for five minutes.

Neutral class comes from here — just record walking and blocking with no moves.

---

## Step 3: Extract frames

Drop all videos into `data/` and run the extraction script. It detects motion automatically and strips idle frames, splitting output into train and val folders. Target is 400+ frames per class.

---

## Step 4: First model

Train a binary classifier first — Sepultura vs. neutral only. Two classes, one afternoon. If the model fires correctly on a real Giovanna clip, the pipeline works and we move on to all five classes.

That's the plan. Next post is the first training run.
