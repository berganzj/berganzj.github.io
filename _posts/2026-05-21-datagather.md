# Data Gathering for a Fighting Game AI: What I Got Wrong First

Building the model is the fun part. Getting clean data to train it is where projects go to die.

Here's what I ran into, what I'm doing about it, and the minimum path to a working model.

---

## The LosslessCut trap

My first instinct was to record match footage and use LosslessCut to manually cut out clips of each move. It works in theory. In practice it's tedious enough to kill your momentum before you ever train anything.

The bigger problem: I don't need to cut clips at all. The data pipeline already uses motion detection to strip out idle frames automatically. What I actually need is clean, single-class recordings — one video per move. If the video only contains one move being repeated, the extractor does the rest.

LosslessCut is the wrong tool for this job. The right tool is Training Mode.

---

## Training Mode as a data factory

In GGS Training Mode, you can set the dummy to loop a specific move on repeat. Record five minutes in OBS and you get hundreds of labeled examples of that move — no cutting, no sorting, no LosslessCut.

This is the core data strategy. One OBS session per class. Let the motion filter handle extraction.

---

## A hidden data source: the in-game command list

GGS has a command list in the menu that plays a short animation for every move. This is free, clean, labeled video of each move in isolation — no gameplay noise, no opponent on screen, no health bars flashing.

It's not a replacement for real gameplay footage, but as a way to bootstrap a small dataset fast or validate that the pipeline works, it's genuinely useful. Worth recording before touching Training Mode.

---

## The PNG quality concern

After running the extractor for the first time, the output frames look noticeably compressed. This is worth taking seriously.

The issue is the resize step: the script shrinks every frame to 224×224 pixels regardless of your original recording resolution. At 1080p source footage, that's a significant downscale. The PNG format itself is lossless — so the quality loss is happening at the resize, not the save.

For a POC this is acceptable. ResNet18 was designed for 224×224. But for a production tool that needs to distinguish subtle move differences, higher resolution (384×384 or 512×512) will matter. Filing that as a future problem. For now, the model will learn what it needs to learn at 224×224.

---

## What moves to actually target

A previous suggestion was to include the full breadth: punches, kicks, slashes, heavy slashes, and dusts. I thought about this more carefully.

Punches and kicks are a problem. They're fast, small, and visually similar to each other and to neutral movement. A frame-level classifier looking at a single image has almost no signal to distinguish a 5P from a 2K. You'd need temporal context — sequences of frames — to do it reliably. That's a harder problem to build and a harder problem to validate.

The moves worth targeting first are the ones that are visually unmistakable from a single frame:

**In for POC:**
- **Ventosa** — full-body command grab lunge, nothing else looks like it
- **Erpressung** — airborne overhead flip, distinctive arc
- **Seppi** — forward rushing slash with clear horizontal motion
- **Dust** — launching D move, very distinct animation
- **Neutral** — idle, walking, blocking (the baseline class)

**Deferred:**
- Punches, kicks, normals — too similar to each other, need temporal context
- Sol e Vento (super) — useful eventually, but rare enough in matches that the dataset will be tiny

Special moves are where the coaching value is highest anyway. These are the high-commitment actions that win and lose rounds. Knowing when your opponent throws Ventosa — or knowing you never used Erpressung in a set — is directly actionable.

---

## Other workarounds worth knowing about

A few alternatives to brute-force recording, logged here for future reference:

**Community VODs.** High-level Giovanna gameplay exists on YouTube. In principle you could download it with yt-dlp and run the extractor on it. The catch: footage contains both players, varying camera behavior during supers, and no clean class labels. Useful for testing a trained model on real footage. Not great for training data.

**Semi-supervised labeling.** Label 50–100 frames per class manually, train a rough model, then use that model's predictions to soft-label a larger unlabeled set. Only worth it at a later stage when you need volume and hate the idea of more recording sessions.

**Synthetic data.** Generating fake training images with AI tools is possible but probably overkill here. The game itself is the best synthetic data source you have.

---

## The minimum path to a first working model

Here's the smallest sequence of steps that produces something real:

1. Open GGS command list. Screen record Ventosa animation for 2–3 minutes on loop.
2. Open Training Mode. Record yourself doing nothing — walking, blocking — for 3 minutes. This is your neutral class.
3. Drop both videos in `data/`. Run the extractor.
4. Train ResNet18 on those two classes. This is a binary classifier: Ventosa vs. not Ventosa.
5. Find a Giovanna clip online. Run predict.py on it. See the model fire on Ventosa attempts.

That's the taste of it working. Two classes, one afternoon, one result that feels real.

Everything after that — more classes, better resolution, temporal context, coaching output — is an expansion of something that already runs.

---

Next post: what the first training run actually looked like.
