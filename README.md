# Review Intelligence

**What are customers actually complaining about, how often is a free local language model right about it, and which complaints cost the most?**

This project reads about 41,000 written customer reviews from Olist, a real Brazilian marketplace (100k orders, 2016 to 2018), and tags each one with the complaints it contains: not delivered, late, missing items, wrong item, damaged, not as described, poor quality, seller support, refunds, packaging, price and shipping. The model is an open 7B model running on my own GPU through Ollama, so the whole project costs nothing to run and no review text leaves the machine. Because a small model makes mistakes, I measure how often it is right before using any of its answers.

| Layer | Question | How |
|---|---|---|
| **1. Classify** | Which complaints does each review contain? | Qwen 2.5 7B served locally by Ollama, answers forced into a JSON schema of 12 themes, every answer cached so reruns are free |
| **2. Measure** | How often is the model right? | Per theme precision, recall and F1 on a dev set used to improve the prompt, and a separate test set the prompt never sees |
| **3. Size** | Which complaints matter most to the business? | Each theme linked to order value, star rating and whether the customer bought again |
| **4. Decide** | What should be fixed first? | A fix list ranked by the money attached to each complaint |

<!-- FINDINGS:START -->

## Status

The full classification of all 40,950 reviews is running. Findings, charts and the fix list will appear here when it finishes.

Early result on the dev set: writing down the rules for ambiguous cases and putting them in the prompt (v1 to v2) raised exact match from 77% to 81% and micro F1 from 0.78 to 0.81, and the model now spots whether a review is a complaint at all 99% of the time.

<!-- FINDINGS:END -->

## Why I built it this way

**A local open model instead of a paid API.** Anyone can rerun this on a consumer GPU for free, and the review text never leaves the machine. The trade-off is accuracy, which is why measuring it is part of the project rather than an afterthought.

**The model can only answer in a fixed format.** Ollama's structured output forces every answer into a JSON list drawn from the 12 theme names. Anything outside the list is dropped, so a creative answer cannot slip a made-up theme into the results.

**I wrote the rules down before judging the model.** The first labels showed that one line theme definitions leave common cases open. Is a customer still waiting after the deadline "late" or "not delivered"? Is "bought two, got one" a missing item or a delivery problem? Those rules live in [docs/LABELLING_GUIDE.md](docs/LABELLING_GUIDE.md), and the prompt follows the same rules, so the model is judged against the definitions it was given.

**The prompt is tuned on one set and scored on another.** The sample is stratified by star rating (a random sample would be 58% five star reviews with almost no complaints). A dev split is used to improve the prompt. The test split is scored once, at the end, so the reported accuracy is not flattered by tuning.

**Translation is kept separate from classification.** I do not read Portuguese, so labelling uses an offline Portuguese to English model that has nothing to do with the classifier. If the classifier translated for me, its reading of a review would leak into the labels used to judge it.

**Long runs survive interruptions.** Eight requests run in parallel on the GPU, every answer is written to a cache as it arrives, and a failed request is retried on the next run instead of being recorded as "no complaint".

The full reasoning is in [docs/DECISIONS.md](docs/DECISIONS.md).

## Running it

Needs Python 3.10+ and [Ollama](https://ollama.com).

```
ollama pull qwen2.5:7b
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Download the [Olist dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) and extract the CSVs into `data/raw/`. Then:

```
python -m reviews check          confirm the data and the model are ready
python -m reviews evaluate       score the prompt on the dev split
python -m reviews classify       classify every review (resumable)
pytest                           run the tests
```

Every judgement call, from the theme list to the sample sizes, is in `config.toml`.

## Project layout

```
reviews/          the pipeline: data loading, classifier, translation, labelling tool, scoring
docs/             decisions log and labelling guide
data/labels/      the labelling sample and labels
tests/            unit tests
```

## Data

Brazilian E-Commerce Public Dataset by Olist, released on Kaggle under CC BY-NC-SA 4.0. The raw files are not included in this repository.
