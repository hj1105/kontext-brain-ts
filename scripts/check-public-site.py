"""Validate crawlable static pages and source links without third-party dependencies."""
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
import json
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
BASE = "https://hj1105.github.io/kontext-brain-ts/"
REPO = "https://github.com/hj1105/kontext-brain-ts/blob/main/"


class Page(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.lang = None
        self.links = []
        self.meta = {}
        self.ids = set()
        self.h1 = 0
        self.row_cells = None
        self.structured_data = []
        self.json_buffer = None
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        data = dict(attrs)
        if tag == "tr":
            self.row_cells = 0
        if tag in ("th", "td") and self.row_cells is not None:
            self.row_cells += 1
        if tag == "html":
            self.lang = data.get("lang")
        if "id" in data:
            self.ids.add(data["id"])
        if tag in ("link", "a"):
            self.links.append(data)
        if tag == "meta":
            self.meta[data.get("name", data.get("property"))] = data.get("content")
        if tag == "h1":
            self.h1 += 1
        if tag == "script" and data.get("type") == "application/ld+json":
            self.json_buffer = ""

    def handle_data(self, data):
        if self.json_buffer is not None:
            self.json_buffer += data

    def handle_endtag(self, tag):
        if tag == "tr":
            assert self.row_cells == 2, "Use-case table rows must have two cells"
            self.row_cells = None
        if tag == "script" and self.json_buffer is not None:
            self.structured_data.append(json.loads(self.json_buffer))
            self.json_buffer = None


pages = {}
for path in SITE.rglob("*.html"):
    page = Page(path.read_text())
    relative = path.relative_to(SITE).as_posix().removesuffix("index.html")
    canonical = BASE + relative
    assert page.h1 == 1, f"{path}: expected one H1"
    assert page.lang in ("en", "ko"), f"{path}: missing language"
    assert page.meta.get("description"), f"{path}: missing description"
    assert page.meta.get("robots") == "index,follow", f"{path}: index blocked"
    assert [x.get("href") for x in page.links if x.get("rel") == "canonical"] == [canonical]
    assert page.meta.get("og:url") == canonical
    assert page.structured_data and page.structured_data[0]["@type"] == "SoftwareSourceCode"
    assert page.structured_data[0]["codeRepository"] == REPO.removesuffix("/blob/main/")
    assert {x.get("hreflang") for x in page.links if x.get("rel") == "alternate"} == {"en", "ko", "x-default"}
    for link in page.links:
        href = link.get("href", "")
        parsed = urlsplit(href)
        if href.startswith(REPO):
            source = unquote(href.removeprefix(REPO).split("#")[0])
            assert (ROOT / source).exists(), f"Missing source: {href}"
        elif href.startswith(BASE):
            target = SITE / unquote(parsed.path.removeprefix("/kontext-brain-ts/"))
            if target.is_dir():
                target /= "index.html"
            assert target.exists(), f"Missing local target: {href}"
        elif not parsed.scheme:
            if parsed.path:
                assert (path.parent / unquote(parsed.path)).exists(), f"Missing asset: {href}"
            if parsed.fragment:
                assert parsed.fragment in page.ids, f"Missing anchor: {href}"
    pages[canonical] = page

urls = {item.text for item in ET.parse(SITE / "sitemap.xml").getroot().iter("{http://www.sitemaps.org/schemas/sitemap/0.9}loc")}
assert urls == set(pages), "Sitemap and public pages differ"
assert len(pages) == 2, "Expected English and Korean project pages"
print(f"PASS: {len(pages)} static pages, canonical/hreflang, source links, structured data, and sitemap")
