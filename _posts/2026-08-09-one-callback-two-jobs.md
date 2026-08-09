---
layout: post
title: "One Callback, Two Jobs: Forking a Strive Mod into HitLab's Data Logger"
date: 2026-08-09
description: The plan said the whole labeling problem reduces to one evening of mod work — fork the frame-data overlay so that, inside a single frame callback, it writes one row per game tick and draws a counter into the corner. This post is that evening: which mod to fork, what to log field by field, what to draw, how to verify it before trusting a single frame, and the two-pass review I run on the data that comes out. The project has a name now: HitLab.
---

# One Callback, Two Jobs: Forking a Strive Mod into HitLab's Data Logger

Aug 9, 2026

The last two posts were architecture and plan — frames to classifier to decoder
to timeline, labels borrowed from community mods, pixels synced to ground truth
by a counter burned into the corner of every frame. Plans are cheap. This post
is the first place the plan touches a compiler. And it's the first post with the
product's name in it: **HitLab**. The CLI is `hitlab`, the timeline schema lives
in the repo, and the thing I'm building toward is a tool that reads a match and
tells you what you got wrong.

But none of that exists until the labels do. So the whole job this week reduces
to one deliberately small target the plan set: make a mod write its own dataset,
then look hard at what it wrote before I trust any of it. Two halves — the fork,
and the review — and this post is both.

## Why a fork at all

Worth restating what I'm *not* doing, because it's the part that used to scare
me off. I am not writing a memory scanner. The Strive community already did that,
twice, out in the open.

Altimor's `strivehitboxes` injects a DLL that draws hurtboxes and hitboxes over
the running game — install is deliberately simple, drop `strivehitboxes.dll` next
to `striveinjector.exe`, start the game, run the injector, and the DLL lives
inside the game process drawing boxes over it. The value to me isn't the boxes;
it's proof that hitbox, hurtbox, and counterhit state are readable every frame
from a source I can audit on GitHub rather than a mystery binary.

`StriveFrameViewer` (Procdox, plus active forks) does the same for per-frame
*player* state, rendering the startup/active/recovery/hitstun bars you'd see in
SF6's training mode. Standalone install is extract `Standalone.zip` into the
game's `...\RED\Binaries\Win64`, and delete the old `cache` folder on upgrade or
it misbehaves.

That second one is the grandmaster I want annotating for me, and it's the one I
fork. It has already solved the hard problem — finding and reading, every tick,
exactly which move each player is in and which animation frame within that move.
My fork doesn't add intelligence. It redirects an existing read from the screen
to a file. `strivehitboxes` stays installed as a second opinion: if a field I'm
logging ever looks wrong, its box rendering is an independent read I can eyeball
against.

## The one callback

Here's the whole trick, and it's smaller than it sounds. Somewhere in the viewer
is a callback that fires once per game tick — 60 times a second, on the clean
game clock — and reads both players' state to draw its bars. I make that callback
do two extra things:

1. Append one row to a log file with both players' state for this tick.
2. Draw the frame-counter patch — a little grid of light and dark cells in a
   fixed corner that encodes *this tick's* `game_frame` in binary.

The reason both live in the same callback is the entire reason the scheme works.
The number I draw into the corner and the number I write into the log row are the
same integer, produced on the same tick, with nothing allowed to change between
them. That's what makes the counter a perfect key back into the log. Split them
across two callbacks and let the clock tick in between, and I've reintroduced the
exact off-by-one the counter exists to kill.

## What to log, field by field

One JSON object per line — `.jsonl`, append-only, flushed often. A row looks like
this:

```json
{"game_frame": 91442, "p1": {"move_id": 214, "anim_frame": 7, "flags": ["startup"]}, "p2": {"move_id": 0, "anim_frame": 0, "flags": ["neutral"]}}
```

- **`game_frame`** — the mod's monotonic tick counter. This is the canonical
  clock for the entire project and the key the clapperboard reads back. It must
  be the exact same integer I draw into the corner patch on this tick.
- **`move_id`** — the raw move ID the game reports, logged as a raw integer, not
  a human name. Name resolution (214 becomes `Sepultura`) happens later, in
  preprocessing, against a mapping table pinned to the game version. Resolve
  names inside the mod and I bake today's guesses into the data forever; a dumb,
  reversible log is one I can re-interpret when I'm smarter next month.
- **`anim_frame`** — the animation frame *within* the move. This is what makes
  sub-move phase labels possible — startup vs. active vs. recovery — which is
  what later lets a model commit to a call before the move finishes. Cheap to log
  now, expensive to reconstruct later.
- **`flags`** — whatever phase state the viewer already computes to color its
  bars: startup, active, recovery, hitstun, blockstun, counterhit. I'm already
  calculating these; I just serialize them.

Both players, every tick, even when nothing is happening. Neutral ticks are
`move_id: 0` — I do **not** skip them. The gaps are load-bearing: they're where
the move boundaries and the neutral class both come from.

Write discipline that saves me later: open the log once at session start, append
and flush each tick, close on session end. Do not buffer the whole session in
memory and dump at the end — a crash then costs the entire run.

## What to draw: the clapperboard, stated once and left alone

I'm keeping the counter itself at arm's length here, because the bit layout and
checksum belong to the preprocessing spec, not the mod. What matters mod-side is
three properties, and only these three:

**Fixed position and fixed cell geometry.** Preprocessing needs to know exactly
where to look. Pick a corner, pick a cell size — fat cells, think ~16px, so they
survive H.264 — and never let them move. Record the geometry in `run.json` so the
decoder is configured, not hard-coded.

**Same integer, same tick.** The number in the patch on tick N equals the
`game_frame` in the log row for tick N. Draw before I read, or read before I draw,
with the clock able to tick between, and the whole scheme is dead.

**Always drawn.** Even on menu frames, pause, whatever. A frame with a patch is
always placeable; a frame without one is a frame I can't trust. And because this
is a training-time tool on my own PC, I can just crank the OBS bitrate until the
read is trivial. I'll leave room in the encoding for a checksum so preprocessing
can drop a garbled read rather than believe it — but the layout is a later
problem.

The property I keep coming back to: every capture pathology degrades gracefully.
A dropped frame makes the counter jump — I simply have no image for those ticks.
A duplicate repeats the counter — dedupe it. A garbled read fails the checksum —
drop it. Every failure costs one training example; none of them produces a wrong
one. That sentence is the whole reason I trust an automated labeler I never
manually check.

## Verifying the fork before I trust it

The temptation after the fork compiles is to immediately record an hour of
footage and feel productive. That's exactly how you generate a thousand corrupt
labels. So before capturing anything real, an by-eye check.

Load the game with the fork, go to training mode, and watch two things at once.
The corner counter should tick up cleanly and never freeze while the game runs.
And I tail the log: mash `f.S`, and I should see a run of rows with that move's ID
and `anim_frame` climbing 0, 1, 2… then falling back to neutral. Do a move with
known startup and count the `startup` flags — if the frame data says 9f startup,
I should see about nine startup rows; if I see three, the flag read is wrong or
I'm reading the wrong player.

Then the single check the whole thing rides on: pick one tick, read the number
off the corner by eye, and confirm the log has a row with that exact `game_frame`
showing the move I was doing at that instant. That's the entire clapperboard
promise in one manual test. If that holds, the automated join holds.

It didn't hold the first time, which is the useful part. My first pass logged
neutral as *nothing* — I skipped idle ticks to save space. Wrong instinct, and
the by-eye check caught it in five minutes where a confusion matrix would have
hidden it for a week. Every tick gets a row, idle included.

## The first honest look at the data

One clean session out of the fork is three files: `session.mp4`, a
`state_log.jsonl`, and a `run.json` that pins the game version and the mod commit
hash. Before any of it goes near training, a review pass — because no human ever
looked at these labels one by one, so a silent corruption poisons thousands at
once and I find out only when the model is mysteriously bad.

The mindset isn't "is the data perfect." It's "are the failure modes the safe
kind" — every pathology costing one example, never producing a wrong one. Most of
the review is just confirming that property held. I run it in two passes.

**Pass one is the raw log, no video needed.** Is `game_frame` strictly
increasing, with a delta of 1 almost everywhere? I histogram the frame deltas and
want a spike at 1; a step of 0 (duplicate) or a negative step is a logging bug to
fix in the mod, not a capture artifact to patch in the data. Does every `move_id`
resolve against the version-pinned mapping? An ID I can't name is the scariest
finding in the whole review, because it usually means the game patched and the
vocabulary reindexed under me — stop, this run is on the wrong side of a boundary.
Do move durations and phases look believable: a 20-frame move showing up as ~20
rows, `anim_frame` climbing and resetting cleanly rather than jumping around? Does
every row have both a `p1` and a `p2` block even during neutral? And is neutral
actually the majority, and does it look like neutral?

**Pass two is the joined frames.** Preprocessing reads the counter off each video
frame, dedupes repeats, drops checksum failures, crops the patch out, and joins
the rest against the log. I have it emit counts — frames seen, duplicates dropped,
checksum failures dropped, counter values with no matching row, rows with no
matching frame — and the thing I'm confirming is that these are all *drops* and
never *mismatches*. There's no such thing as a joined-but-wrong label if the
counter is the key; if the kept total is way lower than expected, the read is
failing too often and I raise the bitrate.

Then the highest-value five minutes in the whole project: open twenty random
frames from each class folder and just *look*. A `Sepultura` folder full of
neutral poses is unmissable to the eye and invisible to every automated check I
could write. That's the review that catches the corruption I didn't think to look
for. While I'm in there: confirm the counter patch actually got cropped out of
every frame, or the classifier cheats off it and posts fake accuracy.

Last, the counts that make the run usable. Examples per class — I want a few
hundred usable frames per move for the v0 five-plus-`other` vocabulary. The
`other`/neutral bucket will dominate raw counts, so I decide the down-sample ratio
now and record it, because it's a variable I'll hold fixed while changing others.
And rare moves like `Ventania` are allowed to be thin — I flag them for a targeted
training-mode capture rather than letting the model learn them from ten examples.

A run passes review when the clock is monotonic with a delta-1 spike, every ID
resolves, durations look right, the join is all drops and no mismatches, a random
visual sample per class shows the right move, the patch is cropped, and I know my
per-class counts and neutral ratio. I write those numbers down — they're the
baseline the next run gets diffed against. It's the same "diff against ground
truth instead of guessing" discipline the eval uses later, just pointed at the
data itself. The payoff: when the model underperforms, I've already ruled out the
data, so a bad number points cleanly at the model or decoder.

## The rule sitting over all of it

These mods break on game patches, and the move IDs underneath them can shift. So
every collection run pins the game version, records the mod commit hash in
`run.json`, and each patch is a hard dataset boundary. A sync scheme this precise
is worthless if the label vocabulary silently reindexes under it. When Strive
updates, I re-verify the ID mapping before trusting a single new frame, and never
mix pre- and post-patch data in one training set.

## Where this lands

The plan promised the whole labeling bottleneck collapses into one evening of mod
work, and — for once — it roughly did. One callback, two jobs: write a row, draw
a counter. The grandmaster was already reading the board every frame; I just
handed it a pen, then spent the next hour making sure it was writing the truth.
What's left before the first model is the preprocessing join and the actual
counter decode, which is the easiest computer-vision task in the whole project.

None of the pieces is big. That's the thing I keep relearning: you don't build
"the data pipeline," you fork one overlay, make one callback do one more thing,
and then you look at what came out before you believe it.

---

*The two working documents behind this post, reproduced in full below as
reference — the checklist I actually run at the terminal.*

---

# Appendix A — Mod fork guide: turning a Strive overlay into a data logger

The goal of this step is small and specific. Neither community mod, out of the
box, writes a labeled dataset. This is one evening of work: fork whichever
project exposes the cleanest per-frame state and make it do two things inside
the same frame callback — append one row per game tick to a log file, and draw
the frame-counter patch in a corner. Both happening in the same callback is the
whole reason the counter is a perfect key into the log.

Everything here is a training-time-only tool. It runs on your PC, on your
pinned game version, and ships nowhere. Keep that framing; it decides a lot of
the trade-offs below (you can be sloppy about performance and generous about
bitrate, because nothing about this has to survive contact with a real user).

## Which mod to fork

Two projects are in play, both free and open-source:

**StriveFrameViewer** (Procdox, plus active forks) is the one to fork. It's an
SF6-style frame-data display for Training/Replay mode that already renders
per-frame startup/active/recovery/hitstun bars for both players. That means it
has already solved the hard part — locating and reading per-frame player state
every tick. You are borrowing that read and redirecting it to a file instead of
(or in addition to) the screen.

**strivehitboxes** (Altimor) is an injected DLL hitbox viewer. Keep it around as
a cross-check: if a field you're logging looks wrong, its box rendering is an
independent read of hitbox/hurtbox/counterhit state you can eyeball against.
You probably don't fork this one; it's your second opinion.

The install paths, for reference:
- StriveFrameViewer standalone: extract `Standalone.zip` into
  `...\steamapps\common\GUILTY GEAR STRIVE\RED\Binaries\Win64`. On upgrade,
  delete the old `cache` folder or it misbehaves.
- strivehitboxes: put `strivehitboxes.dll` next to `striveinjector.exe`, start
  the game, then run `striveinjector.exe`.

## What to log: one row per game tick

Find the per-frame callback where the viewer already reads both players' state
to draw its bars. That callback fires once per game tick at a clean 60Hz, and
each tick has a monotonic game-frame number. Append one row there. Target
schema, one JSON object per line (`.jsonl`, append-only, flush often):

```jsonl
{"game_frame": 91442, "p1": {"move_id": 214, "anim_frame": 7, "flags": ["startup"]}, "p2": {"move_id": 0, "anim_frame": 0, "flags": ["neutral"]}}
```

Field by field:

- **`game_frame`** — the mod's monotonic tick counter. This is the canonical
  clock for the entire project and the key the clapperboard reads back. It must
  be the exact same integer you draw into the corner patch on this tick.
- **`move_id`** — the raw move ID the game reports for that player this tick.
  Log the raw integer, not a human name. Name resolution (ID -> `Sepultura`)
  happens later, in preprocessing, against a version-pinned mapping table. If
  you resolve names inside the mod you bake today's guesses into the data
  forever; keep the log dumb and reversible.
- **`anim_frame`** — the animation frame *within* the move. This is what makes
  sub-move phase labels possible (startup vs. active vs. recovery), which is
  what later enables early detection. Cheap to log now, expensive to reconstruct
  later.
- **`flags`** — whatever phase/state the viewer already knows: startup, active,
  recovery, hitstun, blockstun, counterhit. You're already computing these to
  color the bars; just serialize them.

Two players, every tick, even when nothing is happening. Neutral ticks are
`move_id: 0` (or whatever the game's idle ID is) — don't skip them; the gaps
matter for the join and for computing boundaries.

Write discipline that saves you later: open the log once at session start,
append and flush each tick (or every N ticks), close on session end. Put the
`run_id` in the filename. Do **not** buffer the whole session in memory and
dump at the end — a crash then costs you the entire run.

## What to draw: the frame-counter patch

Kept high-level here on purpose (the encode/decode spec is a separate task when
you get to preprocessing). The essential properties the drawing code must have:

The patch is a small grid of light/dark cells in a fixed screen corner that
encodes *this tick's* `game_frame` in binary, drawn every single frame. It is a
per-frame heartbeat, not a one-time sync — a film clapperboard syncs once at the
head of a take and trusts constant rates after; this re-establishes lockstep 60
times a second, so drift never accumulates. Draw it in the same callback that
writes the log row, using the same `game_frame` integer, so the two can never
disagree.

Three things to get right in the mod, and only these three:

1. **Fixed position and fixed cell geometry.** Preprocessing needs to know
   exactly where to look. Pick a corner, pick a cell size (fat cells survive
   H.264 — think ~16px), and never let them move. Record the geometry in
   `run.json` so the decoder is configured, not hard-coded.
2. **Same integer, same tick.** The number in the patch on tick N must equal
   the `game_frame` in the log row for tick N. If you draw before you read (or
   vice versa) and the counter can tick between them, you've reintroduced the
   off-by-one the whole scheme exists to kill.
3. **Draw it always.** Even on menu frames, pause, whatever. A frame with no
   patch is a frame you can't place; a frame with a patch is always placeable.

Leave a little room in the encoding for a checksum so preprocessing can drop a
garbled read rather than trust it — but the actual bit layout and checksum
choice belong to the preprocessing spec, not the mod. The mod's only job is to
draw the agreed pattern.

## Verifying the fork before you collect anything

Do not capture a real session until the fork passes a by-eye check, because
every downstream label depends on this being right.

Start the game with the fork loaded, go to Training Mode, and watch two things
at once. First, the corner counter should increment cleanly and never freeze or
jump while the game is running normally. Second, tail the log file and confirm
the rows line up with what you're doing: mash `f.S`, and you should see a run of
rows with that move's ID and `anim_frame` climbing 0,1,2,... then falling back
to neutral. Do a known move with known startup and count the `startup` flags —
if the frame data says 9f startup, you should see about nine startup rows.

The single most important check: pick one tick, read the number off the corner
patch by eye, and confirm the log has a row with that exact `game_frame` showing
the move you were doing at that instant. That's the whole clapperboard promise in
one manual test. If that holds, the automated join will hold.

## The operational rule that keeps the data honest

These mods break on game patches, and the move IDs underneath them can shift.
So every collection run pins the game version and records the mod commit hash in
`run.json` (see `../run.example.json`), and each patch is a **hard dataset
boundary**. A sync scheme this precise is worthless if the label vocabulary
silently reindexes under it. When Strive updates: re-verify the ID mapping
before trusting a single new frame, and never mix pre- and post-patch data in
one training set.

## Definition of done for this step

You have a fork that, in one frame callback, appends a correct per-tick row and
draws a correct corner counter; you've eyeballed that a hand-read counter value
indexes to the right row; and you've captured one clean high-bitrate session
(Giovanna mirror, Training Mode, a few minutes) producing exactly three files:
`session.mp4`, `state_log.jsonl`, and `run.json`. That trio is the first raw
dataset, and it's the input to the review below.

---

# Appendix B — High-level data review: what to check before training

The mod turns labeling from the project's bottleneck into a free byproduct of
playing. That's the upside. The risk is the mirror image: because no human ever
looks at each label, a silent corruption — a wrong ID mapping, a broken join, a
duplicated-frame bug — poisons thousands of examples at once and you find out
only when the model is mysteriously bad. This review is the cheap insurance. It
runs in two passes: the raw state log, then the joined labeled frames. Do it
once per run before that run's data is allowed anywhere near training.

The mindset: you are not looking for the data to be perfect, you are looking for
the *failure modes to be the safe kind*. Every capture pathology should cost you
one training example, never produce a wrong one. Most of this review is
confirming that property held.

## Pass 1 — the raw state log (`state_log.jsonl`)

This pass needs no video. It's checking that the mod logged sane, monotonic,
believable state.

**Monotonic, gapless-where-it-should-be clock.** `game_frame` must strictly
increase. Diff consecutive rows: the step should be 1 almost everywhere.
Occasional larger steps are fine if the game genuinely paused (menus, loading),
but a step of 0 (duplicate) or a negative step means a logging bug, not a
capture artifact — fix the mod, don't patch the data. Plot or histogram the
frame deltas; you want a spike at 1 and nothing weird.

**Move-ID sanity against the pinned mapping.** Every `move_id` in the log should
resolve to a known move in the version-pinned mapping table. Count distinct IDs
and eyeball the list: an ID you can't name is the single scariest finding,
because it usually means the game patched and the vocabulary reindexed under
you. If that happens, stop — this run is on the wrong side of a dataset
boundary. IDs are raw integers on purpose so this check is possible; resolve
names only here, never in the mod.

**Believable move durations and phase structure.** Group consecutive rows by
move and look at the run lengths. A 20-frame move should show up as ~20 rows.
`anim_frame` should climb 0,1,2,... within a move and reset at the next one, not
jump around. The `startup` flag count for a known move should roughly match its
published frame data — if `f.S` is 9f startup and you see three startup rows,
either the flag read is wrong or you're reading the wrong player.

**Both players present every tick.** Confirm every row has a `p1` and a `p2`
block even during neutral. Missing-player rows mean the read failed silently for
one side; those ticks are unusable and you want to know the rate.

**Neutral is the majority and looks like neutral.** In a normal session most
ticks are idle/walk/block. If neutral is rare, you're probably mislabeling idle
as something, or the idle ID isn't what you think.

## Pass 2 — the joined labeled frames (`data/labeled/<run_id>/`)

This pass checks the clapperboard join: that the pixels actually match the label
the log assigned them.

**The join accounting adds up.** Preprocessing reads the counter off each video
frame, dedupes repeats, drops checksum failures, and joins the rest against the
log. Have it emit counts: frames seen, duplicates dropped, checksum failures
dropped, counter values with no matching log row, log rows with no matching
frame. None of these should be *wrong labels* — they should all be *dropped
examples*. Confirm the dropped-fraction is small and, more importantly, that the
"joined but mismatched" bucket is zero by construction (there's no such thing if
the counter is the key). If the total kept is wildly lower than expected, the
counter read is failing more than it should — raise the capture bitrate.

**Eyeball a random sample per class.** The highest-value five minutes in the
whole review: open ~20 random frames from each class folder and confirm they
show that move. This is where a swapped-ID mapping or an off-by-one join reveals
itself instantly — a `Sepultura` folder full of neutral poses is unmissable to
the eye and invisible to any automated check. Do this every run; it's the thing
that catches the corruptions you didn't anticipate.

**Spot-check onsets against the counter.** Pick a couple of moves, read the
corner counter on the first visibly-active frame, and confirm the log marks that
`game_frame` as the move's onset (allowing for startup). This is the manual
version of the eval's onset-latency metric, done by hand on a few examples
before you trust the automated pipeline.

**Confirm the counter patch got cropped out.** Every labeled frame must have the
corner patch removed before it becomes a training example, or the classifier
cheats off it and posts fake accuracy. Look at the corner in a few frames from
each class; it should be background, not a grid of cells.

## Class balance and vocabulary readiness (across the run)

**Count examples per class.** For the v0 five-plus-`other` vocabulary, you want
a few hundred usable frames per move (the earlier POC targeted 400+). Print the
per-class counts. Heavy imbalance isn't fatal but you should know about it before
training, not diagnose it from a confusion matrix afterward.

**Watch the `other` bucket.** `other`/neutral will dominate raw frame counts.
Decide the cap now (down-sample neutral so it doesn't swamp the five real
classes) and record the ratio you used — that's a variable you'll want to hold
fixed while changing others.

**Rare moves are allowed to be thin — flag them, don't force them.** Overdrives
like `Ventania` are naturally rare in play. If a class is starved, note it and
plan a targeted Training-Mode capture for it rather than letting the model learn
it from ten examples.

## What "passed review" means, and where it feeds

A run passes when: the clock is monotonic with a delta-1 spike, every `move_id`
resolves against the pinned mapping, durations and phases look believable, the
join accounting is all *drops* and no *mismatches*, a random visual sample per
class shows the right move, the patch is cropped out, and you know your per-class
counts and neutral ratio. Record those numbers in the run's notes — they're the
baseline you compare the next run against.

This is the same "diff against ground truth instead of guessing" discipline the
eval uses later, just applied by hand to the data itself. The payoff is that when
the model underperforms, you've already ruled out the data as the cause — so a
bad number points at the model or decoder, which is exactly the composability the
whole pipeline is built for.
