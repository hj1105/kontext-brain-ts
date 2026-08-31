# ADR 0005: Index code as Resources and semantic Code Symbols

- Status: Accepted

A source file is synchronized through the existing evidence-backed Resource path, while source-native semantic regions become Chunks and declarations receive stable Code Symbol identities based on Codebase, path, language, kind, qualified name, and signature rather than line number. Language providers emit Facts only for relationships they can establish deterministically; unresolved or syntactic-only relationships remain Hypotheses or inconclusive instead of being promoted to false certainty. Code extraction stays outside `@kontext-brain/core` and enters through `ResourceSnapshot`, preserving the derived-index boundary from ADR 0001.
