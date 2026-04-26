#!/usr/bin/env python3
"""
Nettoie un fichier SRT raffiné :
1. Fusionne les segments adjacents du MÊME speaker (même fenêtre)
2. Supprime les segments trop courts (< 0.5s) ou vides
3. Supprime les entrées "[interjection — pas de texte attribuable]"
4. Renumérote en 1, 2, 3...
5. Format SRT standard compatible VLC / lecteurs vidéo

Usage: python clean_srt.py <input.srt> [output.srt]
"""

import sys, os, re


def fmt_srt_time(seconds: float) -> str:
    """Convertit des secondes en HH:MM:SS,mmm."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


def parse_srt(path: str):
    """Parse un fichier SRT → liste de {start, end, speaker, text}."""
    entries = []
    with open(path, encoding="utf-8") as f:
        content = f.read()
    
    pattern = re.compile(
        r'(?:\d+)\n'
        r'(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*'
        r'(\d{2}):(\d{2}):(\d{2}),(\d{3})\n'
        r'(SPEAKER_\d+):?\s*(.*?)(?=\n\d+\n|\n#|\Z)',
        re.DOTALL
    )
    
    for m in pattern.finditer(content):
        start = int(m.group(1))*3600 + int(m.group(2))*60 + int(m.group(3)) + int(m.group(4))/1000
        end   = int(m.group(5))*3600 + int(m.group(6))*60 + int(m.group(7)) + int(m.group(8))/1000
        entries.append({
            'start': start, 'end': end, 'speaker': m.group(9), 'text': m.group(10).strip()
        })
    return entries


def clean_srt(entries, min_duration=0.5, merge_same_speaker=True):
    """
    Nettoie une liste d'entrées SRT.
    - min_duration : durée minimale en secondes (les plus courts sont supprimés)
    - merge_same_speaker : fusionner les segments adjacents du même speaker
    """
    # 1. Supprimer les entrées vides / interjections sans texte / trop courtes
    cleaned = []
    for e in entries:
        text = e['text'].replace('\n', ' ').strip()
        dur = e['end'] - e['start']
        
        # Skip empty/injection-only
        if not text or '[interjection' in text.lower() or '[pas de texte' in text.lower():
            continue
        # Skip trop court
        if dur < min_duration:
            continue
        cleaned.append({**e, 'text': text})
    
    if not cleaned:
        return cleaned
    
    # 2. Fusionner segments adjacents du même speaker (max 15s par segment)
    if merge_same_speaker:
        merged = [cleaned[0].copy()]
        for e in cleaned[1:]:
            last = merged[-1]
            gap = e['start'] - last['end']
            current_dur = last['end'] - last['start']
            
            if e['speaker'] == last['speaker'] and gap < 0.8 and current_dur < 15:
                # Fusionner
                last['end'] = max(last['end'], e['end'])
                last['text'] = last['text'] + ' ' + e['text']
            else:
                merged.append(e.copy())
        cleaned = merged
    
    return cleaned


def write_srt(entries, path: str, header_comment: str = ""):
    """Écrit un fichier SRT standard."""
    with open(path, "w", encoding="utf-8") as f:
        if header_comment:
            for line in header_comment.strip().split('\n'):
                f.write(f"# {line.strip()}\n")
            f.write("\n")
        for i, e in enumerate(entries, 1):
            dur = e['end'] - e['start']
            if dur < 0.1:
                continue
            f.write(f"{i}\n")
            f.write(f"{fmt_srt_time(e['start'])} --> {fmt_srt_time(e['end'])}\n")
            # Nettoyer les retours à la ligne dans le texte
            clean_text = e['text'].replace('\n', ' ').strip()
            # Supprimer les doubles espaces
            clean_text = re.sub(r'\s+', ' ', clean_text)
            f.write(f"{e['speaker']}: {clean_text}\n\n")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <input.srt> [output.srt]")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else input_path.replace('.srt', '-clean.srt')
    
    entries = parse_srt(input_path)
    print(f"Entrées brutes : {len(entries)}")
    
    cleaned = clean_srt(entries, min_duration=0.5, merge_same_speaker=True)
    print(f"Entrées après nettoyage : {len(cleaned)}")
    
    removed = len(entries) - len(cleaned)
    
    # Stats
    speakers = {}
    for e in cleaned:
        spk = e['speaker']
        speakers[spk] = speakers.get(spk, 0) + 1
    
    print(f"Supprimées/fusionnées : {removed}")
    print(f"Speakers : {speakers}")
    
    header = (
        f"SRT nettoyé — prêt pour lecteur vidéo\n"
        f"Source : {os.path.basename(input_path)}\n"
        f"Entrées : {len(cleaned)} (retirées : {removed})\n"
    )
    write_srt(cleaned, output_path, header)
    print(f"✅ Fichier écrit : {output_path}")


if __name__ == "__main__":
    main()
