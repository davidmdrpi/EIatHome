// What happened to the result cards computed in this browser: computed here, saved to a file,
// opened as a GitHub issue, or accepted into the community results. Nothing here is sent
// anywhere; the counts are shown to the volunteer.

export const STATUSES = {
  computed: 'Computed here',
  saved: 'Saved to a file',
  opened: 'Opened on GitHub',
  accepted: 'Accepted',
};

// One card's identity in this browser: its unit and when it started.
export const cardKey = (card) => `${card.spec}#${card.unit}@${card.started}`;

// The community results' cards, by their exact text: ingestion records the card that was
// submitted, so a card computed here was accepted exactly when its text is among these.
export const acceptedCards = (results) => new Set(results.map((r) => JSON.stringify(r.card)));

// marks: { [cardKey]: { saved?, opened? } }, the times a card was saved or opened for submission.
export function cardStatus(card, marks, accepted) {
  if (accepted.has(JSON.stringify(card))) return 'accepted';
  const mark = marks[cardKey(card)] ?? {};
  if (mark.opened) return 'opened';
  if (mark.saved) return 'saved';
  return 'computed';
}

// A card can be dropped to make room only if it exists somewhere else: in the community
// results, or in a file the volunteer saved.
export const safeToDrop = (status) => status === 'accepted' || status === 'saved';

// Adds a new card, newest first, keeping at most `cap`. Over the cap, the oldest cards that are
// safe to drop go first; a card that exists nowhere else is never dropped. replaces(card) marks
// kept cards the new one supersedes. Returns { cards, full }: full when the cards are over the
// cap because none could be dropped.
export function keepCard(cards, card, cap, isSafe, replaces = () => false) {
  const kept = [card, ...cards.filter((c) => !replaces(c))];
  for (let i = kept.length - 1; kept.length > cap && i > 0; i--) {
    if (isSafe(kept[i])) kept.splice(i, 1);
  }
  return { cards: kept, full: kept.length > cap };
}

// How many cards are in each state, and how many still need submitting (computed or saved, but
// neither opened nor accepted).
export function tally(cards, marks, accepted) {
  const counts = { computed: 0, saved: 0, opened: 0, accepted: 0 };
  for (const card of cards) counts[cardStatus(card, marks, accepted)]++;
  return { ...counts, total: cards.length, awaiting: counts.computed + counts.saved };
}

// Marks only for cards still kept, so the record doesn't outgrow them.
export const pruneMarks = (marks, cardLists) => {
  const keys = new Set(cardLists.flat().map(cardKey));
  return Object.fromEntries(Object.entries(marks).filter(([key]) => keys.has(key)));
};
