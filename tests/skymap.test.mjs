// web/js/skymap.js: the Observatory Viewport's coordinates, projection and slews, and the sky data
// it draws (web/data/sky/sky.json, tools/build_sky.py) and the pointings it reads from the slices.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

import { RECOVERY_DATA, runRecoveryUnit, parseRecoveryWindows, tally, windowOutcomes, SIGNALS, STRENGTHS } from '../web/js/recovery.js';
import {
  BEAM_FOV, OUTCOMES, WIDE_FOV, angleBetween, beamArcmin, bvToRgb, camera, computeFov, constellationAt, equatorialToGalactic,
  formatDec, formatRA, galacticToEquatorial, horizonToEquatorial, radec, siderealHours, slerp, slew, vec,
} from '../web/js/skymap.js';

const sky = JSON.parse(readFileSync(new URL('../web/data/sky/sky.json', import.meta.url)));
const slices = JSON.parse(readFileSync(new URL('../web/data/slices/index.json', import.meta.url)));
const close = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg ?? ''} ${a} vs ${b}`);

test('positions convert to vectors and back, and great circles interpolate between them', () => {
  for (const [ra, dec] of [[0, 0], [258.17, 12.4], [101.29, -16.72], [359.9, 89.5]]) {
    const [r, d] = radec(vec(ra, dec));
    close(r, ra, 1e-9);
    close(d, dec, 1e-9);
  }
  const a = vec(258.17, 12.4);
  const b = vec(14.49 * 15, -62.68);
  const angle = angleBetween(a, b);
  for (const t of [0, 0.25, 0.5, 1]) close(angleBetween(a, slerp(a, b, t)), t * angle, 1e-9, `slerp ${t}`);
});

test('galactic coordinates: the galactic centre and north pole land where they should', () => {
  const [ra, dec] = galacticToEquatorial(0, 0);
  close(ra, 266.405, 0.01, 'RA');
  close(dec, -28.936, 0.01, 'Dec');
  const [pra, pdec] = galacticToEquatorial(123, 90);
  close(pra, 192.859, 0.01);
  close(pdec, 27.128, 0.01);
  const [l, b] = equatorialToGalactic(ra, dec);
  close(Math.min(l, 360 - l), 0, 1e-6);
  close(b, 0, 1e-6);
});

test('the camera puts its centre in the middle, north up and east to the left, and inverts exactly', () => {
  const center = vec(258.17, 12.4);
  const cam = camera(center, 20, 800, 400);
  assert.deepEqual(cam.project(center).map((x) => Math.round(x * 1e6) / 1e6), [400, 200]);
  const north = cam.project(vec(258.17, 14.4));
  const east = cam.project(vec(260.17, 12.4));
  assert.ok(north[1] < 200 && Math.abs(north[0] - 400) < 1, 'north is up');
  assert.ok(east[0] < 400, 'east is to the left, as the sky looks from below');
  for (const [x, y] of [[0, 0], [123, 321], [799, 1]]) {
    const back = cam.project(cam.unproject(x, y));
    close(back[0], x, 1e-6);
    close(back[1], y, 1e-6);
  }
  // The vertical field of view spans the canvas height.
  const top = cam.project(vec(258.17, 22.4));
  close(top[1], 0, 1e-6, 'a point half the field above the centre is at the top edge');
});

test('a slew starts and ends at its views, follows the great circle, and zooms out for long moves', () => {
  const from = vec(30, -20);
  const to = vec(258.17, 12.4);
  const plan = slew(from, 20, to, 2);
  close(angleBetween(plan.at(0).center, from), 0, 1e-9);
  close(angleBetween(plan.at(1).center, to), 0, 1e-9);
  close(plan.at(0).fov, 20, 1e-9);
  close(plan.at(1).fov, 2, 1e-9);
  assert.ok(plan.at(0.5).fov > 20, 'zooms out on the way');
  const mid = plan.at(0.5).center;
  close(angleBetween(from, mid) + angleBetween(mid, to), angleBetween(from, to), 1e-6, 'on the great circle');
  assert.ok(plan.duration >= 0.7 && plan.duration <= 3.2);
  close(slew(to, 10, to, 10).at(0.5).fov, 10, 1e-9, 'no move, no zoom');
});

test('computations zoom from the wide view to the telescope beam with progress', () => {
  assert.equal(computeFov(0), WIDE_FOV);
  close(computeFov(1), BEAM_FOV, 1e-12);
  close(computeFov(0.5), Math.sqrt(WIDE_FOV * BEAM_FOV), 1e-9, 'evenly in log scale');
  assert.equal(computeFov(2), computeFov(1));
  close(beamArcmin(8.42), 1.5, 0.01, "the GBT's beam at Voyager's X band");
});

test('sidereal time and the horizon: the zenith is at the latitude and the local sidereal time', () => {
  close(siderealHours(new Date('2000-01-01T12:00:00Z'), 0), 18.697374558, 1e-6, 'GMST at J2000.0');
  const lst = 7.25;
  const [ra, dec] = horizonToEquatorial(90, 0, 38.4331, lst);
  close(dec, 38.4331, 1e-9);
  close(ra, lst * 15, 1e-6);
  const [, south] = horizonToEquatorial(0, 180, 38.4331, lst);
  close(south, 38.4331 - 90, 1e-9, 'due south on the horizon');
});

test('formatting and colours', () => {
  assert.equal(formatRA(17.211244), '17h 12m 40.5s');
  assert.equal(formatDec(11.403773), '+11° 24′ 14″');
  assert.equal(formatDec(-62.679489), '−62° 40′ 46″');
  const [r1, , b1] = bvToRgb(-0.3);
  const [r2, , b2] = bvToRgb(1.8);
  assert.ok(b1 > r1, 'hot stars are blue');
  assert.ok(r2 > b2, 'cool stars are red');
});

test('the sky data: the brightest stars, real fields around the targets, and every constellation', () => {
  assert.equal(sky.format, 'eiathome-sky');
  assert.equal(sky.stars.length, 1500);
  for (let i = 1; i < sky.stars.length; i++) assert.ok(sky.stars[i][2] >= sky.stars[i - 1][2], 'brightest first');
  assert.ok(sky.stars.every(([ra, dec, mag]) => ra >= 0 && ra < 360 && Math.abs(dec) <= 90 && mag < 5));
  for (const field of sky.fields) {
    const centre = vec(field.ra_deg, field.dec_deg);
    assert.ok(field.stars.length > 50, field.name);
    assert.ok(field.stars.every(([ra, dec]) => angleBetween(vec(ra, dec), centre) <= field.radius_deg + 1e-3), `${field.name}: inside its radius`);
  }
  // The 88 IAU constellations, with Serpens in its two parts.
  assert.equal(sky.constellations.length, 89);
  assert.equal(sky.boundaries.length, 89);
  assert.ok(sky.boundaries.some((b) => b.name === 'Serpens Cauda') && sky.boundaries.some((b) => b.name === 'Serpens Caput'));
  assert.ok(sky.names.some(([, , , name]) => name === 'Vega'));
  assert.match(sky.source, /Hipparcos/);
  assert.match(readFileSync(new URL('../web/data/sky/LICENSE.d3-celestial.txt', import.meta.url), 'utf8'), /Olaf Frohn/);
});

test('constellations come from the IAU boundaries, as tools/skycoords.py finds them', () => {
  const at = (raHours, dec) => constellationAt(raHours * 15, dec, sky.boundaries);
  assert.equal(at(17.211244, 12.403782), 'Ophiuchus', 'Voyager 1 in 2020');
  assert.equal(at(17.167773, 12.183), 'Ophiuchus', 'Voyager 1 in 2016');
  assert.equal(at(17.211245, 13.403784), 'Hercules', 'the off-source position 1° north');
  assert.equal(at(2.5303, 89.2641), 'Ursa Minor', 'Polaris, around the pole');
  assert.equal(at(21.1463, -88.956), 'Octans', 'around the south pole');
  for (const r of sky.references) assert.equal(at(r.ra_hours, r.dec_deg), r.constellation, r.label);
});

test('the v2 manifest records where every slice pointed, for both Voyager sessions', () => {
  assert.equal(slices.version, 2);
  const dates = new Set();
  for (const s of slices.slices) {
    const k = s.source.sky;
    assert.ok(k && Number.isFinite(k.ra_hours) && Number.isFinite(k.dec_deg), `${s.id} has a pointing`);
    assert.equal(constellationAt(k.ra_hours * 15, k.dec_deg, sky.boundaries), k.constellation, s.id);
    assert.match(k.epoch, /^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$/);
    if (s.family === 'voyager') {
      assert.match(k.distance_label, /AU from the Sun/);
      dates.add(k.epoch.slice(0, 4));
    } else {
      const on = slices.slices.find((o) => o.id === s.source.window_of).source.sky;
      const off = angleBetween(vec(k.ra_hours * 15, k.dec_deg), vec(on.ra_hours * 15, on.dec_deg));
      close(off, 1, 0.01, `${s.id} is 1° from Voyager 1`);
      assert.match(k.distance_label, /^Empty sky, 1\.0° (north|south|east|west) of Voyager 1$/);
    }
  }
  assert.deepEqual([...dates].sort(), ['2016', '2020']);
});

test('the recovery windows point where their scan did, and each window has one outcome', async () => {
  const scan = slices.slices.filter((s) => s.source.url.includes('59046_81628'));
  assert.ok(scan.length);
  for (const s of scan) {
    for (const key of ['ra_hours', 'dec_deg', 'epoch', 'constellation', 'distance_label', 'telescope']) assert.equal(RECOVERY_DATA.sky[key], s.source.sky[key], key);
  }
  const bytes = readFileSync(new URL('../web/data/recovery/test.i8', import.meta.url));
  const data = await parseRecoveryWindows(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
  const { key, scores, results } = runRecoveryUnit(5, data, { onProgress: () => {} });
  const codes = windowOutcomes(key, scores);
  const count = (name) => codes.filter((c) => OUTCOMES[c] === name).length;
  const found = SIGNALS.reduce((n, k) => n + results.recovery.dedoppler['0.01'][k].reduce((a, b) => a + b, 0), 0);
  assert.equal(count('found'), found);
  assert.equal(count('found') + count('missed'), SIGNALS.length * STRENGTHS.length * 18);
  assert.equal(count('false alarm'), results.false_alarms.dedoppler['0.01']);
  assert.deepEqual(tally(key, scores), results);
});

test('a unit reports its progress as it scores, with every window once', async () => {
  const bytes = readFileSync(new URL('../web/data/recovery/test.i8', import.meta.url));
  const data = await parseRecoveryWindows(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
  const reports = [];
  const plain = runRecoveryUnit(9, data);
  const watched = runRecoveryUnit(9, data, { onProgress: (fraction, cells) => reports.push([fraction, cells]) });
  assert.deepEqual(watched.results, plain.results, 'watching changes nothing');
  assert.equal(reports.length, 32);
  assert.equal(reports.at(-1)[0], 1);
  const cells = reports.flatMap(([, c]) => c);
  assert.deepEqual([...cells].sort((a, b) => a - b), [...Array(1024).keys()], 'each window once');
  assert.deepEqual(cells, watched.key.map((s) => s.background), 'in the order they were dealt');
});
