# Trouble at the Moon & Spoon cafe

## Name and purpose

Trouble at the Moon & Spoon cafe is a small dishware classifier. Help Pip, a cupboard sprite, sort a photo into one of three categories: bowl, cup, or plate. The page shows a prediction and estimated probabilities so visitors can explore how a classifier makes guesses and mistakes.

## How to open and use the page

1. Open `dishware-classifier.html` in a browser such as Chrome or Edge. No installation, server, or internet connection is needed.
2. Select **Choose photo**, drag a photo onto the tray, or try a mystery dish. Uploads can be JPG, PNG, or WebP, up to 15 MB.
3. Look at Pip's prediction and the three probabilities in box 2.
4. Expand the questions below the boxes for explanations, a test of all 19 test photos, saved-photo exports, or a replay of recorded training.
5. After uploading your own photo, you can select its true category and press **Save photo + label**. This saves the photo in your browser for future review; it does not retrain the current model.

## How it makes a prediction

Pip learned from 80 labeled dish photos. During training, she repeatedly guessed their categories and adjusted numerical weights to reduce mistakes.

When you choose a photo, the page shrinks the whole image into a 48-by-48 square, keeping its proportions and adding white padding. It measures color, shading, and edges, then uses the learned weights to calculate a score for each category. These scores become three estimated probabilities that add up to 100%. The highest is Pip's prediction.

Pip does not use the filename to guess. Her expression reflects the model's confidence, which does not guarantee that the answer is correct.

## One limitation

The training set is small, and shrinking images can hide useful details such as handles and rims. Pip can confuse bowls with plates or rely on backgrounds instead of dish shapes. The current model correctly classified 13 of 19 held-out test photos (68%); this small test does not establish accuracy on new photos.

## Short development log

This log was drafted from the project files and the changes recorded in this conversation. It is not a replacement for the student's own written responses.

- Built a standalone page for classifying bowls, cups, and plates, with Pip illustrations and probability bars.
- Added learned weights, a recorded training replay, and a way to save and export labeled photos for future training.
- Compared image sizes using validation photos. The selected 48-by-48 model improved the test result from 11 of 19 to 13 of 19 correct compared with the earlier 24-by-24 model.
- Changed the page title to **Trouble at the Moon & Spoon cafe**, removed the repeated top wordmark, and shortened the wording.
- Organized extra information after boxes 1 and 2 into nine expandable questions.
- Discussed grayscale training as a possible future experiment. It has not been implemented.

## Other documents and sources

- [Original course instructions](COURSE-INSTRUCTIONS.md). Preserved from the original README; its course-root paths refer to the full course repository.
- [Detailed model notes](PIP-MODEL-NOTES.md).
- [Dishware dataset on Kaggle](https://www.kaggle.com/datasets/adhamabdelazeem/dishware-classification-dataset).

README drafted with Codex from the existing project and conversation.
## Additional training experiment

Added 9 reviewed photos from the September 21 export to a separate 89-image training dataset; one ambiguous dish was excluded with the user's confirmation. The retrained model kept validation accuracy at 22/25 but had worse validation loss, so it was not promoted. Its test accuracy was 11/19, compared with 13/19 for the main model.

The main page still uses the original model. Try [the retrained experiment](dishware-classifier-retrained.html), or read the [training report](pip-art/retraining-2026-09-21/README.md).