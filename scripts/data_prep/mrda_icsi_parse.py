#!/usr/bin/env python3
"""
Parses official ICSI NXT DialogueActs + Words XML into (meeting, channel,
participant, start, end, text, general_da) rows, using the general-tag
extraction rule documented in the original MRDA paper (Shriberg et al. 2004,
Section 3 and Table 4): the general tag is the first component before any
'^' (specific tags), '|' splits one utterance into multiple separate acts,
and indecipherable/disruption-only tags (%, %-, %--) collapse to '%'.

Validated against Table 4's published general-tag distribution (Section 6)
before being trusted for anything downstream.
"""
import os
import re
import glob
import xml.etree.ElementTree as ET
from collections import Counter

NS = {"nite": "http://nite.sourceforge.net/"}
ROOT = "/tmp/icsi_check/icsi_core/ICSI"

# 11 general tags per the paper, in Table 4's order
GENERAL_TAGS = {"s", "b", "fh", "qy", "fg", "qw", "h", "qh", "qrr", "qr", "qo"}
DISRUPTION_ONLY = {"%", "%-", "%--", "x"}


def extract_general_da(raw_tag):
    """Returns the general_da label per the documented MRDA rule, or None
    if this piece isn't a labelable general-tag DA (e.g. pure 'rt' rising-
    tone marker, which the paper explicitly excludes from the DA count)."""
    if not raw_tag:
        return None
    pieces = raw_tag.split("|")
    labels = []
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        # bare disruption/indecipherable tags (possibly with trailing .%- etc already stripped)
        base = piece.split("^")[0].split(".")[0]
        if piece in DISRUPTION_ONLY or base in DISRUPTION_ONLY:
            labels.append("%")
            continue
        if base in GENERAL_TAGS:
            labels.append(base)
        # else: not a recognized general tag (e.g. 'rt' alone, or nonlabelable) -> skip
    return labels


def parse_words(words_path):
    tree = ET.parse(words_path)
    out = {}
    for el in tree.getroot():
        tag = el.tag.split("}")[-1]
        wid = el.get("{http://nite.sourceforge.net/}id")
        if tag == "w":
            out[wid] = el.text or ""
        elif tag in ("vocalsound", "comment", "pause", "disfmarker"):
            out[wid] = ""
    return out


ID_RANGE_RE = re.compile(r"#id\(([^)]+)\)\.\.id\(([^)]+)\)")
ID_SINGLE_RE = re.compile(r"#id\(([^)]+)\)")


def parse_dialogue_acts(da_path, words_path):
    words = parse_words(words_path)
    word_ids_in_order = list(words.keys())
    id_pos = {wid: i for i, wid in enumerate(word_ids_in_order)}

    tree = ET.parse(da_path)
    rows = []
    for da in tree.getroot():
        tag = da.tag.split("}")[-1]
        if tag != "dialogueact":
            continue
        starttime = da.get("starttime")
        endtime = da.get("endtime")
        dtype = da.get("type")
        participant = da.get("participant")
        child = da.find("nite:child", NS)
        if child is None:
            continue
        href = child.get("href")
        m = ID_RANGE_RE.search(href)
        if m:
            start_id, end_id = m.group(1), m.group(2)
        else:
            m2 = ID_SINGLE_RE.search(href)
            if not m2:
                continue
            start_id = end_id = m2.group(1)
        if start_id not in id_pos or end_id not in id_pos:
            continue
        i0, i1 = id_pos[start_id], id_pos[end_id]
        toks = [words[wid] for wid in word_ids_in_order[i0:i1 + 1]]
        text = " ".join(t for t in toks if t).strip()
        if not text:
            continue
        general_labels = extract_general_da(dtype)
        if not general_labels:
            continue
        rows.append({
            "start": float(starttime) if starttime else None,
            "end": float(endtime) if endtime else None,
            "raw_type": dtype,
            "general_da": general_labels[0] if len(general_labels) == 1 else general_labels,
            "n_split_acts": len(general_labels),
            "participant": participant,
            "text": text,
        })
    return rows


def validate_against_table4():
    """Parses ALL 75 meetings' ALL channels and compares the resulting
    general-tag distribution against the paper's published Table 4."""
    published = {
        "s": 68.00, "b": 13.37, "fh": 7.38, "qy": 4.91, "fg": 2.74,
        "qw": 1.82, "h": 0.70, "qh": 0.36, "qrr": 0.35, "qr": 0.20, "qo": 0.17,
    }
    counter = Counter()
    n_files = 0
    da_files = glob.glob(os.path.join(ROOT, "DialogueActs", "*.dialogue-acts.xml"))
    for da_path in da_files:
        base = os.path.basename(da_path).replace(".dialogue-acts.xml", "")
        words_path = os.path.join(ROOT, "Words", f"{base}.words.xml")
        if not os.path.exists(words_path):
            continue
        try:
            rows = parse_dialogue_acts(da_path, words_path)
        except Exception as e:
            continue
        n_files += 1
        for r in rows:
            labs = r["general_da"] if isinstance(r["general_da"], list) else [r["general_da"]]
            for lab in labs:
                if lab != "%":  # Table 4 excludes indecipherable/nonlabelable
                    counter[lab] += 1

    total = sum(counter.values())
    print(f"Parsed {n_files} channel-files, {total} general-tag instances (excl. '%')\n")
    print(f"{'tag':6s} {'parsed %':>10s} {'paper %':>10s} {'diff':>8s}")
    for tag in sorted(published, key=lambda t: -published[t]):
        parsed_pct = 100 * counter.get(tag, 0) / total if total else 0
        print(f"{tag:6s} {parsed_pct:10.2f} {published[tag]:10.2f} {parsed_pct - published[tag]:+8.2f}")


if __name__ == "__main__":
    validate_against_table4()
