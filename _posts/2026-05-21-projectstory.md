# I'm Building an AI to Help Me Win a Guilty Gear World Championship

I play Guilty Gear Strive competitively. My goal is to qualify for the Japan world championship. And I'm building a computer vision tool to help me get there.

This is the first post in a series documenting how I'm building it — as someone who is still learning Python.

---

## The idea

Fighting games at a high level are a lot like chess. Every match is a series of reads — guessing what your opponent will do, and punishing it. The problem is that humans are bad at tracking patterns under pressure. In the middle of a match you're not thinking "this player has thrown out Ventosa four times from this exact range" — you're just reacting.

What if I had a tool that watched the video for me?

The goal is a program that can watch match footage and tell me: what moves is my opponent doing, how often, and in what situations. A scouting report, generated automatically.

I play Giovanna — a close-range rushdown character with a few high-commitment special moves. Those moves are exactly the kind of thing that's readable and punishable if you know they're coming.

---

## What I'm building

A move recognition pipeline using computer vision and machine learning:

1. Record Giovanna gameplay footage in OBS
2. Extract motion frames automatically from the video
3. Train an image classifier to recognize which move is happening
4. Run it on match replays to generate a move frequency report

The whole thing runs locally on my Windows PC with an RTX 4060 GPU. I'm using Python, PyTorch, and a library called fastai that makes training image classifiers manageable for someone at my level.

---

## The first challenge: getting data

Machine learning needs examples. To train a classifier that recognizes Giovanna's Ventosa grab, I need hundreds of images of Ventosa actually happening.

My first instinct was to record real matches and label the clips manually. That's slow and painful.

The smarter approach: use Training Mode as a data factory. I set the game's training dummy to loop a single move on repeat, hit record in OBS, and let it run for five minutes. That gives me hundreds of clean, perfectly labeled examples of that move with almost no manual work. Repeat for each move I want to detect.

The data collection that felt like it would take days actually takes about two hours.

---

## The plan

I'm breaking this into three stages:

**Stage 1 — Does a move happen or not?** A simple two-class model: "action" vs "neutral." This validates the whole pipeline and is already useful for measuring how active a player is.

**Stage 2 — Which move is it?** Starting with four of Giovanna's most visually distinct specials: Ventosa (command grab), Erpressung (overhead), Seppi (rushing slash), and Sol e Vento (super). These are the high-commitment moves — if you know one is coming, you can punish it.

**Stage 3 — Match analysis tool.** Run the model on a full replay and output a timeline. How often did my opponent throw Ventosa? In what situations? That's the actual coaching insight.

---

## One more thing: the setup

I'm running this across two machines. The Windows PC handles the heavy work — recording, training, running inference. My Mac Mini is where I do the planning and code work with Claude (an AI assistant) helping me build and debug.

The repo lives on GitHub. When something needs updating, it gets pushed from the Mac and pulled to the PC. Simple sync, no drama.

---

I'll post updates as each stage comes together. Stage 1 target: a working model by end of week.

If you play Strive and want to follow along, or if you're also learning Python through a project like this — I'd love to hear from you.
