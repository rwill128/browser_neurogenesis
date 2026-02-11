export function captureRngSnapshot(randomFn) {
  if (!randomFn || typeof randomFn !== 'function') return null;

  const hasGetState = typeof randomFn.getState === 'function';
  if (!hasGetState) return null;

  return {
    type: 'stateful-rng',
    seed: typeof randomFn.getSeed === 'function' ? (randomFn.getSeed() >>> 0) : null,
    state: randomFn.getState() >>> 0
  };
}

export function applyRngSnapshot(randomFn, snapshot) {
  if (!randomFn || typeof randomFn !== 'function') return false;
  if (!snapshot || snapshot.type !== 'stateful-rng') return false;
  if (typeof randomFn.setState !== 'function') return false;

  randomFn.setState(snapshot.state >>> 0);
  return true;
}
