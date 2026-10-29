// What became of the result cards computed in a browser (web/js/contributions.js).
import assert from 'node:assert/strict';
import { test } from 'node:test';

import { acceptedCards, cardKey, cardStatus, keepCard, pruneMarks, safeToDrop, tally } from '../web/js/contributions.js';

const card = (unit, minute = unit) => ({ spec: 'recovery/v0', unit, started: `2026-10-28T00:${String(minute).padStart(2, '0')}:00.000Z`, results: { unit } });

test('a card is accepted only when its exact text is in the community results', () => {
  const [a, b] = [card(1), card(2)];
  const accepted = acceptedCards([{ issue: 5, card: JSON.parse(JSON.stringify(a)) }]);
  const marks = { [cardKey(b)]: { opened: '2026-10-28T01:00:00Z' } };
  assert.equal(cardStatus(a, marks, accepted), 'accepted');
  assert.equal(cardStatus(b, marks, accepted), 'opened');
  assert.equal(cardStatus({ ...a, seconds: 1 }, marks, accepted), 'computed', 'the same unit run again is a different card');
  assert.equal(cardStatus(card(3), { [cardKey(card(3))]: { saved: 'x' } }, accepted), 'saved');
  assert.equal(cardStatus(card(3), { [cardKey(card(3))]: { saved: 'x', opened: 'y' } }, accepted), 'opened');

  // Dropping is safe when the card is accepted or saved, whatever the status shown.
  const both = card(4);
  const savedOpened = { [cardKey(both)]: { saved: 'x', opened: 'y' } };
  assert.equal(cardStatus(both, savedOpened, accepted), 'opened');
  assert.equal(safeToDrop(both, savedOpened, accepted), true, 'saved, then opened');
  assert.equal(safeToDrop(both, { [cardKey(both)]: { opened: 'y', saved: 'z' } }, accepted), true, 'opened, then saved');
  assert.equal(safeToDrop(b, marks, accepted), false, 'opened only');
  assert.equal(safeToDrop(a, {}, accepted), true, 'accepted');
  assert.equal(safeToDrop(card(5), {}, accepted), false, 'computed only');
});

test('saved cards make room even after their issue was opened', () => {
  const cards = [card(3), card(2), card(1)];
  const marks = { [cardKey(cards[2])]: { opened: 'a', saved: 'b' }, [cardKey(cards[1])]: { saved: 'c', opened: 'd' }, [cardKey(cards[0])]: { opened: 'e' } };
  const kept = keepCard(cards, card(4), 2, (c) => safeToDrop(c, marks, new Set()));
  assert.deepEqual(kept.cards.map((c) => c.unit), [4, 3]);
  assert.equal(kept.full, false);
});

test('kept cards make room only by dropping cards that exist somewhere else', () => {
  const cards = [card(3), card(2), card(1)]; // newest first
  const safe = new Set([1]);
  const isSafe = (c) => safe.has(c.unit);

  let kept = keepCard(cards, card(4), 3, isSafe);
  assert.deepEqual(kept.cards.map((c) => c.unit), [4, 3, 2], 'the oldest safe card goes');
  assert.equal(kept.full, false);

  kept = keepCard(cards, card(4), 2, isSafe);
  assert.deepEqual(kept.cards.map((c) => c.unit), [4, 3, 2], 'unsafe cards stay, over the limit');
  assert.equal(kept.full, true);

  safe.add(3);
  kept = keepCard(cards, card(4), 2, isSafe);
  assert.deepEqual(kept.cards.map((c) => c.unit), [4, 2], 'safe cards go oldest first, whatever their place');

  kept = keepCard(cards, card(2, 59), 3, () => false, (c) => c.unit === 2);
  assert.deepEqual(kept.cards.map((c) => [c.unit, c.started.slice(14, 16)]), [[2, '59'], [3, '03'], [1, '01']], 'a superseded card is replaced');
  assert.equal(kept.full, false);

  assert.equal(keepCard([], card(1), 0, () => true).cards.length, 1, 'the new card itself is never dropped');
});

test('the tally counts each state, and what still awaits submission', () => {
  const cards = [card(1), card(2), card(3), card(4)];
  const marks = { [cardKey(cards[1])]: { saved: 'x' }, [cardKey(cards[2])]: { opened: 'y' }, stale: { saved: 'z' } };
  const accepted = acceptedCards([{ card: cards[3] }]);
  assert.deepEqual(tally(cards, marks, accepted), { computed: 1, saved: 1, opened: 1, accepted: 1, total: 4, awaiting: 2 });
  assert.deepEqual(Object.keys(pruneMarks(marks, [cards.slice(0, 2), []])), [cardKey(cards[1])], 'marks of cards no longer kept go');
});
