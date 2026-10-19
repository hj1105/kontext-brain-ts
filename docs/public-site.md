# Public project site

The `site/` directory is a dependency-free HTML introduction to kontext-brain-ts in English and Korean. It describes capabilities supported by the public repository and links to source, architecture decisions, setup, and evaluation limitations. It does not publish private corpora, credentials, or unverified benchmark superiority claims.

Run `python3 scripts/check-public-site.py` before publishing. Preview with `python3 -m http.server --directory site 3215`.

The Public project site workflow validates pull requests and deploys `site/` to GitHub Pages on main. In repository settings, Pages must use GitHub Actions as the build source. The intended canonical URL is `https://hj1105.github.io/kontext-brain-ts/`.

The sitemap is at `https://hj1105.github.io/kontext-brain-ts/sitemap.xml`. A project Pages site does not control the domain-root `/robots.txt`, so no misleading project-local robots file is provided. Submit the sitemap in verified Google Search Console and Bing Webmaster Tools properties when those accounts are available.

Repository About description and homepage should match the published identity. Keep English and Korean pages aligned whenever capabilities or limitations change.

For measurement, record a small stable question set such as “TypeScript knowledge graph RAG framework”, “MCP knowledge server with source evidence”, and “permission-aware self-hosted RAG TypeScript”. Log question, date, service, cited URL, brand mention, and actual referral visits separately. Crawling, indexing, citation, and recommendation are distinct outcomes; a successful deployment only establishes public availability.
