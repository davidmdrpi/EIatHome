// Train/evaluation splits for real data.
//
// Voyager snippets are shifted windows of a few signals, so snippets from the same observing
// session overlap heavily or share an interference environment. When the model trains on
// Voyager, evaluation must use other sessions. Sessions are found from the data itself, never
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

// Group slices into sessions. Returns one array of slices per session, in a canonical order
// (earliest start first), so the result doesn't depend on the order slices were listed.
export function sessions(slices) {
  const parent = slices.map((_, i) => i);
  const find = (i) => (parent[i] === i ? i : (parent[i] = find(parent[i])));
  const join = (a, b) => {
    const [ra, rb] = [find(a), find(b)];
    if (ra !== rb) parent[Math.max(ra, rb)] = Math.min(ra, rb);
  };

  const byTime = slices.map((s, i) => [s.startMjd, i]).sort((a, b) => a[0] - b[0]);
  for (let k = 1; k < byTime.length; k++) {
    if (byTime[k][0] - byTime[k - 1][0] <= SESSION_GAP_DAYS) join(byTime[k - 1][1], byTime[k][1]);
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

export function voyagerPools(slices, evalFraction = 0.3) {
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
