# Labelling guide

Rules for tagging a review with complaint themes. The short theme definitions
in config.toml left common cases open, so these rules settle them. The same
rules go into the classifier prompt, so the model and the labels it is scored
against share one definition.

## General

- Tag every theme the customer actually complains about. A review can have none, one or several.
- Praise is not a complaint. A neutral remark ("ok", a product name, "3") gets no theme.
- A vague negative with no reason ("não recomendo", "muito ruim" about the store) gets no theme. If it names the product as bad ("produto muito ruim"), it is `poor_quality`.
- Wanting a refund or return is not `refund_return` by itself. Use it only when the customer reports trouble getting one.

## Delivery

| Situation | Theme |
|---|---|
| Customer is still waiting, nothing has arrived (even if past the deadline) | `not_delivered` |
| Tracking says delivered, customer has nothing | `not_delivered` |
| It arrived, but late or slowly | `late_delivery` |
| Part of the order arrived, some units or items are still missing or never came | `missing_items` |
| Order arrived in several shipments, nothing is missing | `other_complaint` |
| Customer had to collect the parcel at the post office | `other_complaint` |

## Product

| Situation | Theme |
|---|---|
| A different product, colour, size, voltage or model was sent | `wrong_item` |
| Broken, scratched, torn, leaking, or does not work | `damaged_defective` |
| Differs from the listing, photo or specs; fake or "not original" | `not_as_described` |
| Correct product, but flimsy, thin, cheap or badly finished | `poor_quality` |
| Correct product, customer simply finds it small or does not like it | `other_complaint` |

## Service and cost

| Situation | Theme |
|---|---|
| No reply, no contact, no tracking information, unreachable support | `seller_support` |
| Badly packed, box opened or unsealed, no protection | `packaging` |
| Product or shipping too expensive, charged shipping twice | `price_shipping` |

## Words the translation gets wrong

- "suporte" usually means a mount or stand (a TV mount), not customer support.
- "frete" is shipping cost.
- "correio" / "correios" is the Brazilian postal service.
- "estorno" is a refund to the card.
