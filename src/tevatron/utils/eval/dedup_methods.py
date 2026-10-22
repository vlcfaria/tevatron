"""Match dev passages against corpus passages to remove cross-source duplicates.

Each `map.json` entry may carry an optional `dedup` block:

    {
        "name": "arguana",
        ...
        "dedup": {
            "action": "drop_original",
            "methods": [
                {"method": "suffix", "min_chars": 100, "min_ratio": 0.5},
                {"method": "containment", "min_chars": 100, "min_ratio": 0.5},
                {"method": "jaccard", "threshold": 0.9, "min_tokens": 20}
            ]
        }
    }

Methods are tried in order and the first match wins. For `drop_original` the
corpus doc is removed and the injected dev copy is kept, so qrels do not need
rewriting.

Supported methods:

- exact: normalized text equality
- prefix: the shorter text is a prefix of the longer one
- suffix: the shorter text is a suffix of the longer one
- containment: the shorter text appears anywhere inside the longer one
- jaccard: token Jaccard similarity above a threshold

`min_chars` and `min_ratio` apply to prefix/suffix/containment. `threshold` and
`min_tokens` apply to jaccard.

Usage:
    from tevatron.utils.eval.dedup_methods import build_resolver

    resolver = build_resolver(dev_texts, {"method": "suffix"})
    resolver.matches(corpus_text)
"""

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field

METHODS = ("exact", "prefix", "suffix", "containment", "jaccard")

WORD_RE = re.compile(r"\w+")


def normalize_text(text: str) -> str:
    """NFKC normalize and collapse whitespace."""
    return " ".join(unicodedata.normalize("NFKC", text).split())


@dataclass
class MethodConfig:
    method: str = "exact"
    min_chars: int = 100
    min_ratio: float = 0.5
    threshold: float = 0.9
    min_tokens: int = 20

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError(
                f"Unknown dedup method {self.method!r}; choose one of {METHODS}"
            )
        if self.min_chars < 1:
            raise ValueError("dedup.min_chars must be >= 1")
        if not 0 < self.min_ratio <= 1:
            raise ValueError("dedup.min_ratio must be in (0, 1]")
        if not 0 <= self.threshold <= 1:
            raise ValueError("dedup.threshold must be in [0, 1]")
        if self.min_tokens < 1:
            raise ValueError("dedup.min_tokens must be >= 1")


@dataclass
class DedupConfig:
    methods: list[MethodConfig] = field(default_factory=lambda: [MethodConfig()])
    action: str = "drop_original"
    method: str = field(init=False)

    def __post_init__(self):
        if self.action != "drop_original":
            raise ValueError(
                f"dedup.action {self.action!r} is not implemented; use 'drop_original'"
            )

        self.methods = [
            method
            if isinstance(method, MethodConfig)
            else MethodConfig(**method)
            for method in self.methods
        ]

        if not self.methods:
            raise ValueError("dedup.methods must not be empty")

        seen = set()
        for method in self.methods:
            if method.method in seen:
                raise ValueError(f"Duplicate dedup method {method.method!r}")
            seen.add(method.method)

        self.method = "+".join(method.method for method in self.methods)

    @classmethod
    def from_dict(cls, data) -> "DedupConfig":
        if data is None:
            return cls()
        if isinstance(data, cls):
            return data
        if not isinstance(data, dict):
            raise TypeError(f"dedup config must be a dict, got {type(data)!r}")
        data = dict(data)
        action = data.pop("action", "drop_original")
        if "methods" not in data:
            data = {"methods": [data]}
        return cls(action=action, **data)


@dataclass
class Match:
    method: str
    score: float
    dev_index: int


class _MethodResolver:
    def __init__(self, dev_texts: list[str], config: MethodConfig):
        self.dev_texts = dev_texts
        self.config = config
        self._tokens_by_dev: dict[int, set[str]] = {}
        self._index = self._build_index()

    def match(self, text: str) -> Match | None:
        text = normalize_text(text)
        return getattr(self, f"_match_{self.config.method}")(text)

    def _build_index(self) -> dict:
        index = defaultdict(list)
        method = self.config.method

        for i, text in enumerate(self.dev_texts):
            if method in ("prefix", "suffix", "containment"):
                if len(text) < self.config.min_chars:
                    continue

            if method == "exact":
                index[text].append(i)
            elif method == "prefix":
                index[text[: self.config.min_chars]].append(i)
            elif method == "suffix":
                index[text[-self.config.min_chars :]].append(i)
            elif method == "containment":
                for shingle in self._shingles(self._tokens(text)):
                    index[shingle].append(i)
            elif method == "jaccard":
                tokens = set(self._tokens(text))
                if len(tokens) < self.config.min_tokens:
                    continue
                self._tokens_by_dev[i] = tokens
                for token in tokens:
                    index[token].append(i)

        return index

    def _match_exact(self, text: str) -> Match | None:
        candidates = self._index.get(text)
        return Match("exact", 1.0, candidates[0]) if candidates else None

    def _match_prefix(self, text: str) -> Match | None:
        if len(text) < self.config.min_chars:
            return None
        candidates = self._index.get(text[: self.config.min_chars], ())
        return self._match_affix(text, candidates, "prefix")

    def _match_suffix(self, text: str) -> Match | None:
        if len(text) < self.config.min_chars:
            return None
        candidates = self._index.get(text[-self.config.min_chars :], ())
        return self._match_affix(text, candidates, "suffix")

    def _match_containment(self, text: str) -> Match | None:
        if len(text) < self.config.min_chars:
            return None

        candidates = set()
        for shingle in self._shingles(self._tokens(text)):
            candidates.update(self._index.get(shingle, ()))

        for i in candidates:
            longer, shorter = self._ordered(text, self.dev_texts[i])
            if not self._lengths_ok(shorter, longer):
                continue
            if shorter in longer:
                return Match("containment", len(shorter) / len(longer), i)
        return None

    def _match_jaccard(self, text: str) -> Match | None:
        tokens = set(self._tokens(text))
        if len(tokens) < self.config.min_tokens:
            return None

        candidates = set()
        for token in tokens:
            candidates.update(self._index.get(token, ()))

        best = None
        for i in candidates:
            dev_tokens = self._tokens_by_dev[i]
            union = len(tokens | dev_tokens)
            if not union:
                continue
            score = len(tokens & dev_tokens) / union
            if score >= self.config.threshold and (best is None or score > best.score):
                best = Match("jaccard", score, i)
        return best

    def _match_affix(self, text: str, candidates, kind: str) -> Match | None:
        for i in candidates:
            longer, shorter = self._ordered(text, self.dev_texts[i])
            if not self._lengths_ok(shorter, longer):
                continue
            hit = (
                longer.startswith(shorter)
                if kind == "prefix"
                else longer.endswith(shorter)
            )
            if hit:
                return Match(kind, len(shorter) / len(longer), i)
        return None

    def _lengths_ok(self, shorter: str, longer: str) -> bool:
        if not shorter or len(shorter) < self.config.min_chars:
            return False
        return len(shorter) / len(longer) >= self.config.min_ratio

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return WORD_RE.findall(text.lower())

    @staticmethod
    def _shingles(tokens: list[str], n: int = 5) -> set[str]:
        return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}

    @staticmethod
    def _ordered(a: str, b: str) -> tuple[str, str]:
        return (a, b) if len(a) >= len(b) else (b, a)


@dataclass
class DedupResolver:
    dev_texts: list[str]
    config: DedupConfig
    matched: set[int] = field(default_factory=set)
    matched_by_method: dict[str, set[int]] = field(default_factory=dict)
    match_counts: dict[str, int] = field(default_factory=dict)
    _resolvers: list[_MethodResolver] = field(init=False, repr=False)

    def __post_init__(self):
        self.dev_texts = list(
            dict.fromkeys(normalize_text(text) for text in self.dev_texts)
        )
        self._resolvers = [
            _MethodResolver(self.dev_texts, method) for method in self.config.methods
        ]
        self.matched_by_method = {
            method.method: set() for method in self.config.methods
        }
        self.match_counts = {method.method: 0 for method in self.config.methods}

    def matches(self, text: str) -> bool:
        return self.match(text) is not None

    def match(self, text: str) -> Match | None:
        text = normalize_text(text)
        for resolver in self._resolvers:
            match = resolver.match(text)
            if match is not None:
                self.matched.add(match.dev_index)
                self.matched_by_method[match.method].add(match.dev_index)
                self.match_counts[match.method] += 1
                return match
        return None


def build_resolver(dev_texts, config=None) -> DedupResolver:
    return DedupResolver(dev_texts, DedupConfig.from_dict(config))
