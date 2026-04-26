#!/usr/bin/env python3
"""
Compare un SRT arbitré avec la référence pyannote des speaker turns.
Détecte :
1. Les segments SRT où le speaker change selon pyannote → à splitter
2. Les segments où le speaker SRT ne correspond à aucun speaker pyannote
3. Les segments où plusieurs speakers parlent selon pyannote
"""

import sys, os, re
from collections import defaultdict

def parse_srt(path):
    """Parse un fichier SRT → liste de {index, start_s, end_s, speaker, text}."""
    segments = []
    with open(path, encoding="utf-8") as f:
        content = f.read()
    
    pattern = re.compile(
        r'(\d+)\n'
        r'(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})\n'
        r'(SPEAKER_\d+):?\s*(.*?)(?=\n\d+\n|\Z)',
        re.DOTALL
    )
    
    for m in pattern.finditer(content):
        idx = int(m.group(1))
        start_s = int(m.group(2))*3600 + int(m.group(3))*60 + int(m.group(4)) + int(m.group(5))/1000
        end_s   = int(m.group(6))*3600 + int(m.group(7))*60 + int(m.group(8)) + int(m.group(9))/1000
        speaker = m.group(10)
        text = m.group(11).strip().replace('\n', ' ')
        segments.append({
            'index': idx, 'start': start_s, 'end': end_s,
            'speaker': speaker, 'text': text
        })
    return segments


def parse_speaker_turns(path):
    """Parse le fichier speaker turns → liste de {start, end, speaker}."""
    turns = []
    pattern = re.compile(
        r'(\d{2}):(\d{2}):(\d{2}\.\d{3})\s*→\s*(\d{2}):(\d{2}):(\d{2}\.\d{3})\s+(SPEAKER_\d+)'
    )
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                continue
            m = pattern.search(line)
            if m:
                start_s = int(m.group(1))*3600 + int(m.group(2))*60 + float(m.group(3))
                end_s   = int(m.group(4))*3600 + int(m.group(5))*60 + float(m.group(6))
                speaker = m.group(7)
                turns.append({'start': start_s, 'end': end_s, 'speaker': speaker})
    return turns


def speakers_in_window(turns, start, end):
    """Retourne les speakers qui parlent dans la fenêtre [start, end] et leur temps."""
    spk_time = defaultdict(float)
    for t in turns:
        overlap_start = max(t['start'], start)
        overlap_end = min(t['end'], end)
        if overlap_end > overlap_start:
            spk_time[t['speaker']] += overlap_end - overlap_start
    return dict(sorted(spk_time.items(), key=lambda x: -x[1]))


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <arbitrage.srt> <speaker-turns.txt>")
        sys.exit(1)

    srt_path = sys.argv[1]
    turns_path = sys.argv[2]

    srt = parse_srt(srt_path)
    turns = parse_speaker_turns(turns_path)

    print(f"SRT: {len(srt)} segments")
    print(f"Turns: {len(turns)} segments")
    print()

    issues = []
    for seg in srt:
        spk_in_window = speakers_in_window(turns, seg['start'], seg['end'])
        
        if not spk_in_window:
            issues.append((seg, "NO_MATCH", {}))
            continue
        
        dominant_spk = list(spk_in_window.keys())[0]
        dominant_pct = spk_in_window[dominant_spk] / (seg['end'] - seg['start'])
        
        if len(spk_in_window) >= 2:
            srt_spk_in_window = seg['speaker'] in spk_in_window
            if seg['speaker'] != dominant_spk and dominant_pct > 0.3:
                issues.append((seg, "WRONG_SPEAKER", spk_in_window))
            elif len(spk_in_window) >= 2:
                issues.append((seg, "MULTI_SPEAKER", spk_in_window))
    
    # ── Rapport ─────────────────────────────────────────────────────
    print(f"{'='*80}")
    print(f"ANALYSE : {len(issues)} segments problématiques sur {len(srt)} ({len(issues)/len(srt)*100:.0f}%)")
    print(f"{'='*80}")
    
    by_type = defaultdict(list)
    for seg, typ, spk in issues:
        by_type[typ].append((seg, spk))
    
    for typ, items in sorted(by_type.items()):
        print(f"\n--- {typ} ({len(items)} segments) ---")
        for seg, spk in items[:10]:
            start_str = f"{int(seg['start']//3600):02d}:{int((seg['start']%3600)//60):02d}:{seg['start']%60:06.3f}"
            spk_list = ", ".join(f"{s}({t:.1f}s)" for s, t in spk.items())
            print(f"  #{seg['index']:3d} [{start_str}] SRT={seg['speaker']}  "
                  f"pyannote→ {spk_list}")
        if len(items) > 10:
            print(f"  ... et {len(items)-10} autres")
    
    # ── Suggestions ──────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("SUGGESTIONS")
    print(f"{'='*80}")
    
    multispk = len(by_type.get('MULTI_SPEAKER', []))
    wrongspk = len(by_type.get('WRONG_SPEAKER', []))
    nomatch = len(by_type.get('NO_MATCH', []))
    
    if multispk > 0:
        print(f"\n1. {multispk} segments couvrent plusieurs speakers pyannote → à splitter aux points de changement")
        print(f"   Script de split : utiliser les speaker turns pour découper chaque segment SRT")
    if wrongspk > 0:
        print(f"\n2. {wrongspk} segments où le speaker SRT ne correspond pas au dominant pyannote → à corriger")
    if nomatch > 0:
        print(f"\n3. {nomatch} segments sans aucun match pyannote → vérifier la synchro temporelle")


if __name__ == "__main__":
    main()
