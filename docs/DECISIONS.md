# Decisions log

Every judgement call in this project, why I made it, and what would change my mind.

## 1. Dataset: Olist reviews

Olist reviews are tied to orders and customers, so a complaint theme can be linked to what the customer did next (bought again or not) and to how much the order was worth. Review only datasets cannot do that.

## 2. Model: a local open LLM through Ollama

Anyone can rerun the project for free on a consumer GPU, and no review text leaves the machine. The cost is accuracy, which is why I measure it against my own hand labels instead of assuming it.

## 3. Money is measured mainly through order value and rating, repeat purchase is a side check

Only 3.1% of Olist customers ever place a second order (2,997 of 96,096). That is enough to test whether a theme lowers the repeat rate, but the estimate will be noisy, so the main business sizing uses the order value and star rating attached to each theme.

## 4. The hand labelled sample is stratified by star rating

A plain random sample would be about 58% five star reviews, which rarely complain, and themes like packaging would get only a handful of examples. I take 100 reviews per star rating (500 in total) and reweight to the real rating mix when I report overall accuracy.

## 5. Dev and test sets are kept apart

150 labelled reviews are a dev set I may look at while improving the prompt. The other 350 are a test set I score only once, at the end. Tuning on the same reviews used to report accuracy would make the number look better than it is.

## 6. I label from an independent translation

I do not read Portuguese, so the labelling tool shows each review with an English translation from Argos Translate, an offline model unrelated to the classifier. If the classifier translated for me, its reading of a review would leak into the labels used to judge it. Labelling is blind: the tool never shows the model's answer.
