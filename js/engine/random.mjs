export function mulberry32(seed) {
  const initialSeed = seed >>> 0;
  let t = initialSeed;

  const rng = function () {
    t += 0x6D2B79F5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };

  rng.getSeed = () => initialSeed;
  rng.getState = () => (t >>> 0);
  rng.setState = (nextState) => {
    t = (nextState >>> 0);
  };

  return rng;
}
