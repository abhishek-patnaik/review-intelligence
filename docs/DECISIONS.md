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

## 7. A written labelling guide, shared by the labels and the prompt

Labelling the first reviews showed that the one line theme definitions did not settle common cases. Is a customer still waiting after the deadline "late" or "not delivered"? Is "bought two, got one" a missing item or a delivery problem? The translation also misled me once: "suporte" means a mount or stand, not customer support. I wrote docs/LABELLING_GUIDE.md to settle these, and the classifier prompt follows the same rules, so the model is judged against the definitions it was given.

## 8. The headline accuracy comes from a hand labelled test set

I hand label a fixed set of 100 test reviews (20 per star rating), blind to the model's answers, and report the model's accuracy against those. A separate dev split is used only to iterate on the prompt, so the test set never shapes the prompt it is scoring.

## 9. Problems are ranked by the orders and ratings they touch, not by churn

I planned to size each complaint by the repeat purchases it costs. The data mostly does not support that. Customers with no complaint buy again about 4% of the time, and for all but one theme the repeat rate's 95% interval overlaps that. The exception is late delivery, where customers came back clearly less often. Even there the number of lost returning customers is small, because repeat buying is rare to begin with, so a "revenue lost to churn" ranking would mostly rank noise. So the fix list ranks themes by the value of the orders behind them, and shows how many stars each takes off the average rating. A review that mentions two problems has its rating shortfall split between them, so overlapping themes are not counted twice.

## 10. An outside check on the classifier that needs no labels

The model never sees delivery dates. If it reads delivery complaints correctly, reviews it tags as late should belong to orders Olist actually delivered late. They do, about ten times as often as reviews with no complaint. This does not replace the labelled accuracy check, but it cannot be biased by how the labels were written.

## 11. Sellers are compared by complaint rate, not complaint count

The sellers with the most complaints are mostly the sellers with the most orders. Ranking by count would flag them for being big. I compare complaint rates among sellers with at least 30 written reviews.
