#!/usr/bin/env node
// Runs recovery work units (web/js/recovery.js) on the published windows, outside a browser.
//
//   node tools/recovery_unit.mjs 12             # unit 12's results
//   node tools/recovery_unit.mjs 12 --details   # and each dealt window's spec and scores
//   node tools/recovery_unit.mjs 0 999          # units 0 to 999, their counts added up
//
// tests/test_recovery_browser.py uses --details to check the Python port against this.

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { RECOVERY_SPEC, combine, parseRecoveryWindows, runRecoveryUnit } from '../web/js/recovery.js';

const args = process.argv.slice(2);
const details = args.includes('--details');
const [first, last = first] = args.filter((a) => a !== '--details').map(Number);
if (!Number.isInteger(first) || !Number.isInteger(last) || last < first) {
  console.error('Usage: node tools/recovery_unit.mjs <first unit> [<last unit>] [--details]');
  process.exit(2);
}

const bytes = readFileSync(fileURLToPath(new URL('../web/data/recovery/test.i8', import.meta.url)));
const data = await parseRecoveryWindows(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));

if (first === last) {
  const { results, key, scores } = runRecoveryUnit(first, data);
  const out = details ? { unit: first, results, key, scores: Object.fromEntries(Object.entries(scores).map(([k, v]) => [k, [...v]])) } : { unit: first, results };
  console.log(JSON.stringify(out));
} else {
  const cards = [];
  for (let unit = first; unit <= last; unit++) cards.push({ spec: RECOVERY_SPEC, unit, results: runRecoveryUnit(unit, data).results });
  console.log(JSON.stringify({ first, last, ...combine(cards) }));
}
