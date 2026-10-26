// Test fixtures: synthetic micro-slices in the published format. Not real data.
import { createHash } from 'node:crypto';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

export const ROWS = 16;
export const WIDTH = 256;

export function sliceBytes(valueAt) {
  const view = new DataView(new ArrayBuffer(ROWS * WIDTH * 4));
  for (let i = 0; i < ROWS * WIDTH; i++) view.setFloat32(i * 4, valueAt(i), true);
  return view.buffer;
}

// A drifting line on a deterministic background, so every channel offset gives a different window.
export const lineSlice = (drift, seed = 1) =>
  sliceBytes((i) => {
    const t = Math.floor(i / WIDTH);
    const c = i % WIDTH;
    const line = Math.abs(c - (WIDTH / 2 - drift * (ROWS / 2 - t))) < 1 ? 40 : 0;
    return line + Math.sin(i * 0.37 * seed) + ((i * seed) % 7) / 7;
  });

export const sha256 = (buffer) => createHash('sha256').update(new Uint8Array(buffer)).digest('hex');

export function entry({ id, group = 'carrier', file = `${id}.bin`, digest, observationFile = 'obs-a.h5', tstart = 57650.78, drift = 2.4 }) {
  return {
    id,
    family: 'voyager',
    group,
    file,
    dtype: 'float32-le',
    shape: [ROWS, WIDTH],
    ...(digest === undefined ? {} : { sha256: digest }),
    source: { file: observationFile, tstart_mjd: tstart, foff_hz: -2.7939677238464355, tsamp_s: 18.253611008, drift_channels_per_step: drift, snr: 50 },
  };
}

// Write a manifest plus .bin files to a temporary folder; returns the file paths.
export function writeSliceFolder(slices) {
  const dir = mkdtempSync(join(tmpdir(), 'eiathome-slices-'));
  const manifest = { format: 'eiathome-slices', version: 1, slices: [] };
  const paths = [];
  slices.forEach(({ meta, bytes }, i) => {
    const file = `slice-${i}.bin`;
    writeFileSync(join(dir, file), new Uint8Array(bytes));
    manifest.slices.push({ ...entry(meta), file, sha256: sha256(bytes) });
    paths.push(join(dir, file));
  });
  writeFileSync(join(dir, 'index.json'), JSON.stringify(manifest));
  return [join(dir, 'index.json'), ...paths];
}
