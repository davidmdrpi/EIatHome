// The Live Lab's shared number and name formatting (web/js/format.js), and the split names Track A
// and Track B share (web/js/sample.js).
import assert from 'node:assert/strict';
import { test } from 'node:test';

import { fixed, possessive } from '../web/js/format.js';
import { SPLIT_NOTES, splitNote, trackASplit } from '../web/js/sample.js';

test('a small negative value that rounds to zero is written as zero, never -0.00', () => {
  assert.equal(fixed(-0.001), '0.00');
  assert.equal(fixed(-0.004999), '0.00');
  assert.equal(fixed(-0), '0.00');
  assert.equal(fixed(-0.0004, 3), '0.000');
  assert.equal(fixed(-0.4, 0), '0');
  // a value that doesn't round to zero keeps its sign
  assert.equal(fixed(-0.005), '-0.01');
  assert.equal(fixed(-0.051), '-0.05');
  assert.equal(fixed(-1.2, 0), '-1');
  assert.equal(fixed(0.004), '0.00');
  assert.equal(fixed(0.123), '0.12');
  assert.equal(fixed(NaN), '–');
  assert.equal(fixed(Infinity), '–');
  assert.equal(fixed(null), '–');
});

test('a plural family name ending in s takes the apostrophe alone', () => {
  assert.equal(possessive('Blind windows'), 'Blind windows’');
  assert.equal(possessive('SETI targets'), 'SETI targets’');
  assert.equal(possessive('Voyager 1'), 'Voyager 1’s');
  assert.equal(possessive('Off-source'), 'Off-source’s');
  assert.equal(possessive('Wow! signal (1977)'), 'Wow! signal (1977)’s');
});

test('Track A and Track B name a sample’s split the same way', () => {
  assert.deepEqual(SPLIT_NOTES, { train: 'in the training split', validation: 'in the validation split', test: 'held out in the test split' });
  assert.equal(splitNote(trackASplit({ heldOut: true })), splitNote('test'), 'Track A’s held-out sessions are its test split');
  assert.equal(splitNote(trackASplit({ heldOut: false })), splitNote('train'));
  assert.equal(splitNote(undefined), 'not in the split');
  // Track A has no validation split, so it never says "held out" for one.
  for (const split of ['train', 'validation']) assert.doesNotMatch(splitNote(split), /held out/);
});
