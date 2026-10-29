// Train/evaluation splits for real data.
//
// Real-data snippets (Voyager, off-source) are shifted windows of a few slices, so snippets
// from the same observing session overlap heavily or share an interference environment. When
// the model trains on a real family, evaluation must use other sessions of that family. Sessions are found from the data itself, never
// from file names, which change when a file is renamed or re-exported:
//
// - slices whose scans started within SESSION_GAP_DAYS of each other (directly or through a
//   chain of scans) belong to one session;
// - slices with identical bytes always belong to the same session;
// - if any slice has no start time, independence can't be established, so nothing is held out.
//
// Whenever nothing is held out, the evaluation measures reconstruction of the training data,
// not generalization, and the Live Lab says so.

export const SESSION_GAP_DAYS = 0.5;
// Scans: slices cut from the same scan share its start time exactly.
export const SCAN_GAP_DAYS = 1e-8;

// The UTC date (YYYY-MM-DD) of a modified Julian date.
export const mjdDate = (mjd) => new Date(Math.round((mjd - 40587) * 86400000)).toISOString().slice(0, 10);

// Group slices into observing sessions. Returns one array of slices per session, in a
// canonical order (earliest start first), so the result doesn't depend on the order slices
// were listed.
export const sessions = (slices) => groups(slices, SESSION_GAP_DAYS);

// Slices whose start times chain within gapDays, or whose bytes are identical, form a group.
export function groups(slices, gapDays) {
  const parent = slices.map((_, i) => i);
  const find = (i) => (parent[i] === i ? i : (parent[i] = find(parent[i])));
  const join = (a, b) => {
    const [ra, rb] = [find(a), find(b)];
    if (ra !== rb) parent[Math.max(ra, rb)] = Math.min(ra, rb);
  };

  const byTime = slices.map((s, i) => [s.startMjd, i]).sort((a, b) => a[0] - b[0]);
  for (let k = 1; k < byTime.length; k++) {
    if (byTime[k][0] - byTime[k - 1][0] <= gapDays) join(byTime[k - 1][1], byTime[k][1]);
  }
  const firstWithHash = new Map();
  slices.forEach((s, i) => {
    if (firstWithHash.has(s.sha256)) join(firstWithHash.get(s.sha256), i);
    else firstWithHash.set(s.sha256, i);
  });

  const groups = new Map();
  slices.forEach((s, i) => {
    const root = find(i);
    if (!groups.has(root)) groups.set(root, []);
    groups.get(root).push(s);
  });
  const earliest = (group) => Math.min(...group.map((s) => s.startMjd));
  const firstHash = (group) => group.map((s) => s.sha256).sort()[0];
  return [...groups.values()].sort((a, b) => earliest(a) - earliest(b) || (firstHash(a) < firstHash(b) ? -1 : 1));
}

// Training and evaluation slices for training on one real family (pass that family's slices).
export function sessionPools(slices, evalFraction = 0.3) {
  const reconstruction = (reason, count) => ({ train: slices, eval: slices, heldOut: false, sessions: count, reason });
  if (slices.some((s) => !Number.isFinite(s.startMjd))) return reconstruction('A slice has no start time, so independence can’t be established.', null);
  const groups = sessions(slices);
  if (groups.length < 2) return reconstruction('All loaded slices come from one observing session.', groups.length);

  const nEval = Math.max(1, Math.round(groups.length * evalFraction));
  const evalSet = new Set(groups.slice(-nEval).flat());
  return {
    train: slices.filter((s) => !evalSet.has(s)),
    eval: slices.filter((s) => evalSet.has(s)),
    heldOut: true,
    sessions: groups.length,
  };
}

// Train/validation/test slices of one real family for held-out token statistics (Track B).
// Whole sessions are held out when the family has at least three; otherwise whole scans, so
// a held-out scan can share an observing session with training scans (a within-session
// holdout, labelled as such). Units are found from every real slice, so another family's scan
// can link two sessions. Chronological 60/20/20 split, with at least one unit in each part; a
// copy never counts twice. Also reports the family's sessions (their start times) and which of
// them each part draws on.
export function holdoutSplits(slices, family) {
  const own = slices.filter((s) => s.family === family);
  if (own.some((s) => !Number.isFinite(s.startMjd))) return { error: 'A slice has no start time, so whole scans can’t be held out.' };
  const dated = slices.filter((s) => Number.isFinite(s.startMjd));
  const unitsOf = (gap) => groups(dated, gap).map((g) => g.filter((s) => s.family === family)).filter((g) => g.length);
  const bySession = unitsOf(SESSION_GAP_DAYS);
  let unit = 'session';
  let units = bySession;
  if (units.length < 3) {
    unit = 'scan';
    units = unitsOf(SCAN_GAP_DAYS);
  }
  if (units.length < 3) return { error: `Held-out statistics need slices from at least 3 scans; ${units.length} loaded.` };
  const nEval = Math.max(1, Math.round(units.length * 0.2));
  const unique = (part) => {
    const seen = new Set();
    return part.flat().filter((s) => !seen.has(s.sha256) && seen.add(s.sha256));
  };
  const parts = {
    train: unique(units.slice(0, -2 * nEval)),
    validation: unique(units.slice(-2 * nEval, -nEval)),
    test: unique(units.slice(-nEval)),
  };
  const sessionOf = new Map(bySession.flatMap((g, i) => g.map((s) => [s.sha256, i])));
  const sessionsIn = (part) => [...new Set(part.map((s) => sessionOf.get(s.sha256)))].sort((a, b) => a - b);
  return {
    unit,
    independent: unit === 'session',
    units: units.length,
    ...parts,
    sessions: bySession.map((g) => Math.min(...g.map((s) => s.startMjd))),
    sessionsOf: Object.fromEntries(Object.entries(parts).map(([name, part]) => [name, sessionsIn(part)])),
  };
}
