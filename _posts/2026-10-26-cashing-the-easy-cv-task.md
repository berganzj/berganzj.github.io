---
layout: post
title: "Cashing the Easy CV Task: Decoder, Join, and One Honest Folder"
date: 2026-10-26
---

Last post ended on a promise I made in passing, the kind you only notice later:
the preprocessing join is "the easiest computer-vision task in the whole
project." Plans are cheap, and so are throwaway lines at the end of a post that's
already feeling good about itself. This is the post where I cash it — where the
counter stops being a drawing I described and becomes a number a program reads
back, and where the mod's output turns into a folder of labeled frames I'd
actually train on. Three pieces fell into place today: the detailed fork guide,
the decoder that closes the counter contract, and the join itself. None of them
was hard. All three had exactly one place they could silently lie to me, and the
work was mostly building the thing that catches the lie.

## The gap between a good plan and a compiler

First I went back and read my own fork post like a stranger. It holds up at
altitude: which mod to fork and why, the log schema field by field, the
write-once-flush-often discipline, the by-eye verification, the definition of
done. If I already knew the codebase I could execute from it. But I don't yet,
and reading it with that honesty, the post stops exactly where the work gets
real. "Somewhere in the viewer is a callback." "The bit layout belongs to the
preprocessing spec." Fine sentences for a plan; useless at 11pm with Visual
Studio open.

So I wrote the layer underneath it — the implementation companion. Build the
stock viewer first and change nothing, because if I can't reproduce the plain
overlay then every later failure is ambiguous. Then find the one per-tick
callback by grepping for where the startup/active/recovery colors get chosen,
because the phase decision lives right next to the state read I want. Then the
actual JSONL writer: open once, `fprintf` a row, `fflush` every tick, close on
teardown — small enough to paste. The thing I made myself do was mark every
spot that depends on the live upstream source with `[LOCATE]` instead of
inventing a function name that looks authoritative and is wrong. A guessed
symbol is worse than a blank, because a blank makes me go look and a guess makes
me trust.

## The decoder, or: an encode is worthless alone

Here's the part the old post waved away, and the part that turned out to matter
most. I'd described the clapperboard as "a grid of light and dark cells that
encodes `game_frame` in binary." That's a drawing. It is not a contract until
something can read it back and *disagree* when the read is bad.

So I pinned it down: 24 payload bits carrying the frame number, an 8-bit CRC,
LSB first, near-black and near-white cells fat enough to survive H.264. And then
the only thing that makes any of it trustworthy — the decoder recomputes the CRC
and returns *nothing* on a mismatch. Not a best guess. Not the nearest frame.
Nothing. A garbled read is dropped, never believed.

I wrote the encoder and decoder as a matched pair in the same file and made a
test draw a frame and read it back for a handful of numbers including zero, one,
and the 24-bit max. It failed immediately, which was the useful part: the patch
was being drawn at the top-left corner but the decoder was sampling at the
origin offset I'd written into `run.json`, so it was reading black and decoding
zero for everything. Off by eight pixels. The exact class of bug the whole
counter scheme exists to kill, caught in a unit test in thirty seconds instead
of in a confusion matrix in a week. I fixed the renderer to respect the origin,
and now a corrupted cell round-trips to "dropped" and a clean one round-trips to
the integer I drew. That property — *corruption becomes a drop, never a wrong
answer* — is the single sentence the rest of the project leans on.

## The join: all drops, no mismatches

With a decoder that tells the truth, the join is almost boring, which is how I
knew the earlier "easiest task" line was right for once. Walk the video frame by
frame. Decode the counter. If it won't decode, drop it. If I've already seen
that `game_frame`, it's a duplicated frame — drop it. Otherwise look up the log
row for that exact integer, resolve the player's raw `move_id` against the
version-pinned mapping, crop the counter patch out of the corner so the model
can't cheat off it, and write the frame into `data/labeled/<run_id>/<move>/`.

The design rule I held the whole time: every failure is a *drop*, never a
*mismatch*. Because the counter is the only key, there is no such thing as a
frame that joined to the wrong label — a frame either matches its row or it
doesn't get written. The join emits counts for exactly this reason: frames seen,
checksum drops, duplicate drops, rows with no frame, frames with no row,
unresolved move IDs, and kept-per-class. That block of numbers *is* the review
scorecard from the last post, produced for free. An unresolved `move_id` gets
its own counter and its own drop, because that one is the scary finding — it
usually means the game patched and the vocabulary reindexed under me, and the
right response is to stop, not to guess a name.

The test that proves it feeds in seven synthetic frames on purpose rigged to
break: a good one, a neutral one, one with a move ID I left out of the mapping,
a duplicate, one with a deliberately corrupted patch, one whose counter points
at a row that doesn't exist. Three survive, with the labels their rows dictate,
and the corrupted and unknown ones land in the drop buckets where they belong.
Writing that test taught me something small about my own stats: a log row whose
frame got dropped for an unresolved ID *also* shows up as a row-without-frame,
which is correct but surprised me until I said it out loud. Better to be
surprised by a passing test than by a model.

## Where the code actually lives

The other thing I settled today was repos, and it's a two-repo answer. The mod
is C++ and it's someone else's project, so the clean home for the modding work
is a *fork* of StriveFrameViewer — fork it on GitHub, branch it, keep my changes
diffable against upstream so I can pull their fixes when Strive patches and
breaks everything. Mixing an overlay's C++ into my training repo would just
braid two toolchains together for no reason.

The Python stays in `action-recognition`, which is basically this project
already wearing a more general name: record in OBS, extract frames, train with
fastai, save a model. The GGST twist is only *how* the labels get made — the
counter join replaces the old "extract motion frames" step. And because this is
a fastai course exercise first and a product second, I threw away the stock
torchvision training loop I'd started; fastai owns the model. The join writes
exactly the `data/labeled/<class>/` tree that `ImageDataLoaders.from_folder`
expects, so the two halves meet with no adapter. The move-ID mapping and the v0
vocabulary — five moves plus `other` — live in `configs/`, pinned to the game
version like everything else.

## Where this lands

The labeler is real now. Not "designed," not "planned" — there is a function
that eats a video and a log and lays down labeled frames, and a test that proves
it fails safe. What's left before the first model is the boring, wonderful part:
fork the actual viewer and fill in the `[LOCATE]` holes, capture one clean
Giovanna-mirror session, run it through the join, and open twenty random frames
per folder to confirm by eye that a `Sepultura` folder is full of Sepultura. Then
it's just a fastai notebook pointed at a folder — which, after all this, really
is the easy part.

One callback drew a number. One decoder read it back and learned to say "I don't
know." One join turned the whole thing into a folder I trust. You don't build the
data pipeline; you build the three small honest pieces and let them compose.
