// Recovery work units (recovery.js) in a worker, so the page stays responsive. Loads and
// verifies the published windows once, then runs units on request.

import { RECOVERY_DATA, emptyWindowFlags, parseRecoveryWindows, recoveryCard, runRecoveryUnit, windowOutcomes } from './recovery.js';

let data = null;

const loading = (async () => {
  const response = await fetch(new URL(`../${RECOVERY_DATA.url}`, import.meta.url));
  if (!response.ok) throw new Error(`couldn't fetch the windows (HTTP ${response.status})`);
  data = await parseRecoveryWindows(await response.arrayBuffer());
  postMessage({ type: 'ready', windows: data.n, empty: emptyWindowFlags(data) });
})().catch((err) => postMessage({ type: 'error', message: err.message }));

self.onmessage = async ({ data: msg }) => {
  if (msg.type !== 'unit') return;
  await loading;
  if (!data) return;
  const started = new Date().toISOString();
  const t0 = performance.now();
  const onProgress = (fraction, cells) => postMessage({ type: 'progress', unit: msg.unit, fraction, cells });
  const { results, key, scores } = runRecoveryUnit(msg.unit, data, { onProgress });
  const seconds = Number(((performance.now() - t0) / 1000).toFixed(2));
  // For the viewport: each window's outcome under de-Doppler at 1%, by position along the band.
  const outcomes = Array.from(windowOutcomes(key, scores));
  postMessage({ type: 'result', card: recoveryCard(msg.unit, results, { seconds, started }), outcomes });
};
