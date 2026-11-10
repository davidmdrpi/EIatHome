// Number and name formatting shared by the Live Lab's panels. No DOM, so the tests import it.

// v to a fixed number of digits, never "-0.00": a small negative value that rounds to zero is
// written as zero, as a gain of -0.001 is no worse than 0 at two digits. Not finite: "–".
export function fixed(v, digits = 2) {
  if (!Number.isFinite(v)) return '–';
  const text = v.toFixed(digits);
  return /^-0(\.0*)?$/.test(text) ? text.slice(1) : text;
}

// A family's possessive: "Voyager 1’s", but a plural name ending in s takes the apostrophe alone
// ("SETI targets’", "Blind windows’").
export const possessive = (name) => (/s$/.test(name) ? `${name}’` : `${name}’s`);
