"""Check stable feature arithmetic and deterministic indexing behavior."""

import json
import os
import subprocess
import sys
from collections import Counter

from docfgp.fingerprint import _shingles
from docfgp.minhash import band_keys, sketch, stable_digest
from docfgp.normalize import strict_text, tolerant_text, word_tokens
from docfgp.winnowing import anchors


def test_normalization_preserves_numbers_and_joins_hyphenation() -> None:
    """Preserve numeric content while normalizing common layout artifacts."""
    text = strict_text("Cafe\u0301  42\ninter-\nnational\tCOST")
    assert text == "café 42 international cost"
    assert word_tokens(text) == ("café", "42", "international", "cost")
    assert tolerant_text(text) == "café42internationalcost"
    assert strict_text("inter\u00ad\nnational") == "international"
    assert strict_text("first-\n\nsecond") == "first- second"


def test_tokens_follow_the_word_spacing_of_each_script() -> None:
    """Keep combining marks within words and split unspaced scripts by character."""
    hindi = strict_text("हिन्दी में दस्तावेज")
    assert word_tokens(hindi) == ("हिन्दी", "में", "दस्तावेज")
    assert tolerant_text(hindi) == "हिन्दीमेंदस्तावेज"
    mixed = strict_text("Revenue 本季度 iPhone")
    assert word_tokens(mixed) == ("revenue", "本", "季", "度", "iphone")
    assert word_tokens("ภาษาไทย ๑๒") == ("ภ", "า", "ษ", "า", "ไ", "ท", "ย", "๑", "๒")
    assert word_tokens("ཀུན་མཁྱེན") == ("ཀུ", "ན", "མ", "ཁྱེ", "ན")


def test_minhash_and_band_keys_are_stable_across_processes() -> None:
    """Keep sketches independent of input order and Python hash seeds."""
    values = sketch(["alpha", "beta", "alpha", "gamma"])
    assert len(values) == 512
    assert values == sketch(["gamma", "alpha", "beta"])
    assert len(band_keys(values, "word")) == 32
    fixture = sketch(["alpha", "beta", "gamma"])
    assert fixture[:2] == ("044b9f0ab10bbc43", "0fc4f9ffb70408d3")
    assert band_keys(fixture, "word")[0] == "word:v2:00:8858b7595122d28212a3576e"
    code = (
        "import json; from docfgp.minhash import sketch,band_keys; "
        "print(json.dumps(band_keys(sketch(['gamma','beta','alpha']), 'word')))"
    )
    env = {**os.environ, "PYTHONHASHSEED": "random"}
    child = subprocess.check_output([sys.executable, "-c", code], env=env, text=True)
    assert tuple(json.loads(child)) == band_keys(values, "word")
    assert sketch([]) == ()
    assert band_keys((), "word") == ()


def test_winnowing_is_stable_and_detects_embedded_passage() -> None:
    """Retain passage anchors when a passage is embedded elsewhere."""
    passage = [stable_digest(str(index).encode()) for index in range(100)]
    embedded = [stable_digest(b"prefix")] * 70 + passage + [stable_digest(b"suffix")] * 70
    assert set(anchors(passage)) <= set(anchors(embedded))
    assert anchors(["b", "a", "a", "c"], 2) == ("a",)


def test_precise_multiset_overlap_arithmetic() -> None:
    """Count repeated shingles in directional overlap arithmetic."""
    original = tuple(f"word{index}" for index in range(104))
    edited = (*original[:50], "changed", *original[51:])
    left = Counter(_shingles(original))
    right = Counter(_shingles(edited))
    shared = sum(min(count, right[key]) for key, count in left.items())
    assert shared == 95
    assert shared / sum(left.values()) == 0.95
    duplicate = Counter(_shingles(original + original))
    shared_repeated = sum(min(count, duplicate[key]) for key, count in left.items())
    assert shared_repeated / sum(duplicate.values()) < 0.5
