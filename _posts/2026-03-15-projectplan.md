#Project plan from opus
Your pipeline, the fast.ai way
1. Install fast.ai (you'll need this):


pip install fastai
2. Data loading — fast.ai's DataBlock:


from fastai.vision.all import *path = Path("dataset")dls = DataBlock(    blocks=(ImageBlock, CategoryBlock),    get_items=get_image_files,    splitter=GrandparentSplitter(train_name="train", valid_name="val"),    get_y=parent_label,  # Folder name = label    item_tfms=Resize(224),    batch_tfms=aug_transforms()).dataloaders(path, bs=32)dls.show_batch()  # Preview your data
3. Train — two lines:


learn = vision_learner(dls, resnet18, metrics=accuracy)learn.fine_tune(5)
That's it. fast.ai handles the training loop, CUDA, learning rate, etc.

4. Check results:


learn.show_results()interp = ClassificationInterpretation.from_learner(learn)interp.plot_confusion_matrix()
5. Save the model:


learn.export("action_model.pkl")
6. Use the model later:


learn_inf = load_learner("action_model.pkl")prediction = learn_inf.predict("new_image.png")print(prediction)  # ('kick', tensor(1), tensor([0.02, 0.95, 0.01, 0.02]))
What about video / action recognition?
The fast.ai book (Chapters 1–6) focuses on image classification first. This is actually the right starting point for your MVP:

Recommended fast.ai aligned approach:

Start with single-frame classification (exactly what the book teaches)

Extract key frames from your clips
Label them: punch / kick / throw / jump
Train with vision_learner + fine_tune
This may actually work surprisingly well
Only move to video sequences if single frames aren't enough

The book covers sequence models in later chapters
By then you'll understand the fundamentals
Match your learning to the book chapters
fast.ai Chapter	What you learn	How it applies
1	Image classifier in minutes	Classify action frames
2	Putting model in production	Deploy your action model
3	Data ethics	—
4	Deeper into training	Understand what fine_tune does
5	Image classification details	Improve your model
6	Multi-label & regression	Combine actions, confidence scores
Later	Sequences, tabular, NLP	Video sequences if needed
My recommendation
Follow the fast.ai book. Don't mix in raw PyTorch yet — it'll make things confusing. The book's approach is:

Get something working fast (Chapter 1)
Understand why it works (Chapters 2–6)
Customize when you need to (later chapters)
Start by doing Chapter 1 with your own action images instead of the book's example dataset. That gives you a working model on day one using the tools the course teaches.
