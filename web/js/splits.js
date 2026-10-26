// Train/evaluation splits for real data.
//
// Voyager snippets are shifted windows of a few signals, so snippets from the same
// observation overlap heavily. When the model trains on Voyager, evaluation must use other
// observations. With only one observation loaded there is nothing independent to hold out:
// the evaluation then measures reconstruction of the training data, not generalization.

export function voyagerPools(slices, evalFraction = 0.3) {
  const observations = [...new Set(slices.map((s) => s.observation))].sort();
  if (observations.length < 2) return { train: slices, eval: slices, heldOut: false, observations: observations.length };
  const nEval = Math.max(1, Math.round(observations.length * evalFraction));
  const evalObservations = new Set(observations.slice(-nEval));
  return {
    train: slices.filter((s) => !evalObservations.has(s.observation)),
    eval: slices.filter((s) => evalObservations.has(s.observation)),
    heldOut: true,
    observations: observations.length,
  };
}
