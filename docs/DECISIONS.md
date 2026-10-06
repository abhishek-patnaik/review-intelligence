# Decisions log

Every judgement call in this project, why I made it, and what would change my mind.

## 1. Dataset: Olist reviews

Olist reviews are tied to orders and customers, so a complaint theme can be linked to what the customer did next (bought again or not) and to how much the order was worth. Review only datasets cannot do that.

## 2. Model: a local open LLM through Ollama

Anyone can rerun the project for free on a consumer GPU, and no review text leaves the machine. The cost is accuracy, which is why I measure it against my own hand labels instead of assuming it.
