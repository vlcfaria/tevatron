"""Deduplicate corpus passages against dev/query passages.

Each `map.json` entry may carry a `dedup` block:

    "dedup": {
        "action": "drop_original",
        "methods": [
            {"method": "suffix", "min_chars": 100, "min_ratio": 0.5},
            {"method": "jaccard", "threshold": 0.9, "min_tokens": 20}
        ]
    }

Methods are tried in order and the first match wins. `drop_original` removes
the corpus doc and keeps the injected dev copy. All string comparisons are
case-insensitive after NFKC normalization and whitespace collapsing.

Methods: exact, prefix, suffix, containment, coverage, jaccard.
`min_chars`/`min_ratio` apply to the text methods; `threshold`/`min_tokens` to
coverage/jaccard. `coverage_mode` is `token` or `shingle`.
"""

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field

METHODS = ("exact", "prefix", "suffix", "containment", "coverage", "jaccard")
TOKEN_METHODS = ("containment", "coverage", "jaccard")
WORD_RE = re.compile(r"\w+")


def normalize_text(text: str) -> str:
    """NFKC normalize and collapse whitespace."""
    return " ".join(unicodedata.normalize("NFKC", text).split())


def _tokens(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def _shingles(tokens: list[str], n: int = 5) -> set[str]:
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


@dataclass
class MethodConfig:
    method: str = "exact"
    min_chars: int = 100
    min_ratio: float = 0.5
    threshold: float = 0.9
    min_tokens: int = 20
    coverage_mode: str = "shingle"

    def __post_init__(self):
        if (
            self.method not in METHODS
            or self.coverage_mode not in ("token", "shingle")
            or self.min_chars < 1
            or not 0 < self.min_ratio <= 1
            or not 0 <= self.threshold <= 1
            or self.min_tokens < 1
        ):
            raise ValueError(f"Invalid dedup method config: {self}")


@dataclass
class DedupConfig:
    methods: list[MethodConfig] = field(default_factory=lambda: [MethodConfig()])
    action: str = "drop_original"

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
        if len({method.method for method in self.methods}) != len(self.methods):
            raise ValueError("Duplicate dedup method")

    @property
    def method(self) -> str:
        return "+".join(method.method for method in self.methods)

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


class DedupResolver:
    def __init__(self, dev_texts: list[str], config: DedupConfig):
        self.config = config
        self.dev_texts = list(dict.fromkeys(normalize_text(text) for text in dev_texts))
        self._texts = [text.lower() for text in self.dev_texts]
        self.matched: set[int] = set()
        self.matched_by_method = {
            method.method: set() for method in self.config.methods
        }
        self.match_counts = {method.method: 0 for method in self.config.methods}

        self._needs_tokens = any(
            method.method in TOKEN_METHODS for method in self.config.methods
        )
        self._needs_shingles = any(
            method.method == "containment"
            or (method.method == "coverage" and method.coverage_mode == "shingle")
            for method in self.config.methods
        )
        self._build_indexes()

    def _build_indexes(self):
        self._token_sets: list[set[str]] = []
        self._shingle_sets: list[set[str]] = []
        self._token_index: dict = {}
        self._shingle_index: dict = {}

        if self._needs_tokens:
            token_lists = [_tokens(text) for text in self.dev_texts]
            self._token_sets = [set(tokens) for tokens in token_lists]
            self._token_index = self._invert(self._token_sets)
            if self._needs_shingles:
                self._shingle_sets = [_shingles(tokens) for tokens in token_lists]
                self._shingle_index = self._invert(self._shingle_sets)

        self._short_devs = [
            i for i, tokens in enumerate(self._token_sets) if len(tokens) < 5
        ]

        self._text_indexes: dict[str, dict] = {}
        for config in self.config.methods:
            if config.method not in ("exact", "prefix", "suffix"):
                continue
            index = defaultdict(list)
            for i, text in enumerate(self._texts):
                index[self._text_key(config, text)].append(i)
            self._text_indexes[config.method] = index

    def matches(self, text: str) -> bool:
        return self.match(text) is not None

    def match(self, text: str) -> Match | None:
        text = normalize_text(text).lower()
        token_list = _tokens(text) if self._needs_tokens else []
        token_set = set(token_list)
        shingles = _shingles(token_list) if self._needs_shingles else set()

        for config in self.config.methods:
            method = config.method
            if method in ("exact", "prefix", "suffix"):
                candidates = self._text_indexes[method].get(
                    self._text_key(config, text), ()
                )
            elif method == "containment":
                if shingles:
                    candidates = self._union(self._shingle_index, shingles)
                    candidates.update(self._short_devs)
                else:
                    candidates = range(len(self._texts))
            elif method == "coverage" and config.coverage_mode == "shingle":
                candidates = self._union(self._shingle_index, shingles)
            else:
                candidates = self._union(self._token_index, token_set)

            best = None
            for i in candidates:
                score = self._score(config, text, token_set, shingles, i)
                if score is None:
                    continue
                if best is None or (score, -i) > (best.score, -best.dev_index):
                    best = Match(method, score, i)

            if best is not None:
                self.matched.add(best.dev_index)
                self.matched_by_method[best.method].add(best.dev_index)
                self.match_counts[best.method] += 1
                return best
        return None

    def _score(self, config, query: str, token_set, shingles, i: int):
        if config.method in ("exact", "prefix", "suffix", "containment"):
            return self._text_score(config, query, i)

        shingle = config.method == "coverage" and config.coverage_mode == "shingle"
        query_set = shingles if shingle else token_set
        dev_set = self._shingle_sets[i] if shingle else self._token_sets[i]

        if min(len(token_set), len(self._token_sets[i])) < config.min_tokens:
            return None
        if config.method == "coverage":
            if min(len(query), len(self._texts[i])) < config.min_chars or not query_set:
                return None
            score = len(query_set & dev_set) / len(query_set)
        elif query_set | dev_set:
            score = len(query_set & dev_set) / len(query_set | dev_set)
        else:
            return None
        return score if score >= config.threshold else None

    def _text_score(self, config, query: str, i: int):
        dev = self._texts[i]
        if config.method == "exact":
            return 1.0 if query == dev else None

        longer, shorter = (query, dev) if len(query) >= len(dev) else (dev, query)
        if (
            not shorter
            or len(shorter) < config.min_chars
            or len(shorter) / len(longer) < config.min_ratio
        ):
            return None

        if config.method == "prefix":
            matched = longer.startswith(shorter)
        elif config.method == "suffix":
            matched = longer.endswith(shorter)
        else:
            matched = shorter in longer
        return len(shorter) / len(longer) if matched else None

    @staticmethod
    def _text_key(config: MethodConfig, text: str) -> str:
        if config.method == "exact":
            return text
        if config.method == "prefix":
            return text[: config.min_chars]
        return text[-config.min_chars :]

    @staticmethod
    def _invert(sets: list[set[str]]) -> dict:
        index = defaultdict(list)
        for i, keys in enumerate(sets):
            for key in keys:
                index[key].append(i)
        return index

    @staticmethod
    def _union(index: dict, keys) -> set[int]:
        candidates = set()
        for key in keys:
            candidates.update(index.get(key, ()))
        return candidates


def build_resolver(dev_texts, config=None) -> DedupResolver:
    return DedupResolver(dev_texts, DedupConfig.from_dict(config))
