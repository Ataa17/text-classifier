# Annotation Guidelines — Tunisian Derja Sentiment (binary)

We label the **sentiment the comment expresses**, as pos vs neg. This is a **binary** task:
neutral comments are **discarded**, not labeled.

## Labels

- **positive** — the comment expresses approval, praise, affection, joy, support,
  encouragement, gratitude, or satisfaction.
  - *behi barcha ❤️* / *يعطيك الصحة* / *3aychek champion*
- **negative** — the comment expresses disapproval, criticism, anger, disgust, mockery,
  disappointment, insult, or sadness.
  - *خايب برشا* / *7acha 3lik* / *waqt dhayaa*
- **discard (neutral / not applicable)** — label as discard, it will be dropped from the dataset:
  - purely factual / informational: *الماتش الساعة 8* ("the match is at 8")
  - a genuine question with no stance: *من وين الاغنية؟* ("where's the song from?")
  - tagging / mentions only, "first", pure greetings: *سلام* , *@ami chouf*
  - advertising / spam / off-topic
  - unintelligible, or not Tunisian (pure MSA news copy, pure foreign language)
  - sentiment is truly **mixed and balanced** with no dominant side (see below)

## Decision rules (apply in order)

1. **Can you tell it's pos or neg at all?** If not → **discard**.
2. **Dominant sentiment wins.** If a comment is mostly positive with a small caveat
   (*film behi ama tawil chwaya*, "good film but a bit long") → **positive**. Only discard
   as "mixed" when the two sides are genuinely balanced with no clear lean.
3. **Sarcasm / irony → label the intended meaning, not the literal words.**
   *يا سلام ع الاداء الرائع 👏* said about a loss → **negative**.
4. **Emojis count.** A thumbs-up/heart/😍 with no text → positive; 🤮/💩/😡 → negative.
   Laughing 😂 depends on context (mocking = negative, enjoying = positive); if unclear → discard.
5. **Target doesn't matter.** Sentiment toward the video, a player, a politician, a person
   in the comments — all count. We label sentiment *expressed*, regardless of who it's aimed at.
6. **Profanity is not automatically negative** — some is affectionate/joking in Derja.
   Judge the overall stance.
7. **Script is irrelevant to the label.** Arabizi, Arabic, French, mixed — label the meaning.

## Hard cases — default behavior

- Can't decide after ~10 seconds → **discard**. Don't agonize; consistency beats coverage.
- Negation matters: *mech behi* ("not good") → negative; *mech khayeb* ("not bad") → positive.
- Quoted/reported sentiment you don't endorse → label the sentiment the author themselves expresses.

## Process

- The **κ-overlap set (200 comments)** is labeled **independently and blind** by two annotators
  (no LLM suggestions, no discussion) to measure Cohen's kappa. Reconcile only *after* both finish.
- The **main set** may use LLM pre-labels to speed up, but every label is confirmed by a human.
- Target ≈ **2000** labeled pos/neg comments after discards.
