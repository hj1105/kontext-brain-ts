/**
 * Language-neutral tokenizer port. BM25/TF-IDF quality depends entirely on the
 * tokenizer. Swap implementations for different languages/domains; core logic
 * stays unchanged (OCP).
 *
 * Implementation guidelines:
 *   - Deterministic (pure): same input -> same output
 *   - Implementations decide case normalization
 */
export interface Tokenizer {
  tokenize(text: string): string[];
  /** Bag-of-tokens view (with duplicates), used by BM25. */
  tokenBag(text: string): string[];
  /** Unique set view for keyword matching. */
  tokenSet(text: string): Set<string>;
}

abstract class BaseTokenizer implements Tokenizer {
  abstract tokenize(text: string): string[];
  tokenBag(text: string): string[] {
    return this.tokenize(text);
  }
  tokenSet(text: string): Set<string> {
    return new Set(this.tokenize(text));
  }
}

/**
 * Whitespace + punctuation splitter + lowercase.
 * Suitable for English/Latin languages.
 */
export class WhitespaceTokenizer extends BaseTokenizer {
  private readonly splitter = /[\s\p{P}]+/u;

  constructor(private readonly minTokenLength = 2) {
    super();
  }

  tokenize(text: string): string[] {
    return text
      .toLowerCase()
      .split(this.splitter)
      .filter((t) => t.length >= this.minTokenLength);
  }
}

/**
 * CJK character N-gram tokenizer. For Korean, 2-gram is sensible baseline
 * without a morphological analyzer. Works for Japanese/Chinese too.
 */
export class CharNGramTokenizer extends BaseTokenizer {
  constructor(
    private readonly n = 2,
    private readonly rangeStart = 0xac00,
    private readonly rangeEnd = 0xd7a3,
  ) {
    super();
    if (n < 1) throw new Error("n must be >= 1");
  }

  tokenize(text: string): string[] {
    if (text.length < this.n) return [];
    let filtered = "";
    for (let i = 0; i < text.length; i++) {
      const code = text.charCodeAt(i);
      if (code >= this.rangeStart && code <= this.rangeEnd) {
        filtered += text[i];
      }
    }
    if (filtered.length < this.n) return [];
    const tokens: string[] = [];
    for (let i = 0; i <= filtered.length - this.n; i++) {
      tokens.push(filtered.substring(i, i + this.n));
    }
    return tokens;
  }
}

/** Combines multiple tokenizer outputs — useful for multilingual corpora. */
export class CompositeTokenizer extends BaseTokenizer {
  constructor(private readonly tokenizers: readonly Tokenizer[]) {
    super();
  }

  tokenize(text: string): string[] {
    return this.tokenizers.flatMap((t) => t.tokenize(text));
  }
}

/** Default tokenizer — whitespace only. Generic default for most languages. */
export const DefaultTokenizer: Tokenizer = new WhitespaceTokenizer(2);

/** Multi-language tokenizer — English whitespace + Korean bigram. */
export const MultiLanguageTokenizer: Tokenizer = new CompositeTokenizer([
  new WhitespaceTokenizer(2),
  new CharNGramTokenizer(2, 0xac00, 0xd7a3),
]);
