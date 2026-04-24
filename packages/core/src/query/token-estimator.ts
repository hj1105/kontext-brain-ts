/**
 * Estimates token count for context budget management.
 * Inject the appropriate estimator for your content language.
 */
export interface TokenEstimator {
  estimate(text: string): number;
}

/** Default estimator — English/Latin text (~0.25 tokens per char). */
export const DefaultTokenEstimator: TokenEstimator = {
  estimate(text: string): number {
    return Math.max(1, Math.floor(text.length * 0.25));
  },
};

/** Korean-aware estimator — Korean ~1.5 tokens/char, Latin ~0.25 tokens/char. */
export const KoreanTokenEstimator: TokenEstimator = {
  estimate(text: string): number {
    let korean = 0;
    for (let i = 0; i < text.length; i++) {
      const code = text.charCodeAt(i);
      if (code >= 0xac00 && code <= 0xd7a3) korean++;
    }
    return Math.max(1, Math.floor(korean * 1.5 + (text.length - korean) * 0.25));
  },
};
