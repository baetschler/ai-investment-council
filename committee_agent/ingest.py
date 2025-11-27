import os
import json
import re
import numpy as np
import sys
import youtube_transcript_api

# DIAGNOSE: Prüfen, ob wir die falsche Datei importieren
print(f"🔍 Youtube API geladen von: {youtube_transcript_api.__file__}")
if "site-packages" not in str(youtube_transcript_api.__file__) and "dist-packages" not in str(youtube_transcript_api.__file__):
    print("\n⚠️ ACHTUNG: Es sieht so aus, als würdest du eine LOKALE Datei statt der Bibliothek importieren!")
    print("   Bitte benenne deine lokale Datei 'youtube_transcript_api.py' um (z.B. in 'my_transcript_test.py')")
    print("   und lösche ggf. den Ordner '__pycache__'.\n")

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from dotenv import load_dotenv
import google.generativeai as genai

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

# YouTube Transcript API Client (v1.2.3+)
ytt_api = YouTubeTranscriptApi()


# Lade Umgebungsvariablen (.env muss GOOGLE_API_KEY enthalten)
load_dotenv()
api_key = os.environ.get("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY nicht gefunden!")

genai.configure(api_key=api_key)

# --- KONFIGURATION ---
# Speicherort für unser "Simple Brain"
VECTOR_FILE = "xo_vectors.npy"  # Speichert die mathematischen Vektoren
TEXT_FILE = "xo_text.json"      # Speichert den lesbaren Text

# --- DEINE QUELLEN ---
# Füge hier echte Video-URLs von TraderXO (oder ähnlichen Strategen) ein.
video_urls = [
    "https://www.youtube.com/watch?v=zmgtPS7R2cs&t=3603s",
    "https://www.youtube.com/watch?v=tKsHsU_xAgI&t=3337s",
]

def get_transcript_text(video_url, languages=('de', 'en')):
    """Holt Text aus einem Video mit der neuen youtube-transcript-api (v1.2.3).

    Nutzt YouTubeTranscriptApi().fetch(...) mit Sprach-Fallback.
    """

    # Video-ID aus der URL extrahieren
    video_id_match = re.search(r"(?:v=|/)([0-9A-Za-z_-]{11}).*", video_url)
    if not video_id_match:
        print(f"❌ Keine gültige Video ID in URL: {video_url}")
        return None, None

    video_id = video_id_match.group(1)
    print(f"Lade Transkript für ID: {video_id}...")

    try:
        # fetch liefert ein FetchedTranscript-Objekt (iterierbar über Snippets)
        fetched = ytt_api.fetch(
            video_id,
            languages=list(languages),  # z.B. ['de', 'en']
            # optional: preserve_formatting=True
        )

        # Du kannst direkt über die Snippets iterieren:
        full_text = " ".join(snippet.text for snippet in fetched)

        return full_text, video_id

    except TranscriptsDisabled:
        print(f"❌ Transkripte sind für dieses Video deaktiviert: {video_url}")
        return None, None

    except NoTranscriptFound:
        print(f"❌ Kein Transkript für dieses Video gefunden: {video_url}")
        return None, None

    except Exception as e:
        print(f"❌ Allgemeiner Fehler bei {video_url}: {e}")
        return None, None


def chunk_text(text, chunk_size=1000):
    """Teilt Text in 1000-Zeichen Häppchen, damit sie ins Context-Window passen"""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def get_embedding(text):
    """Holt das Embedding von Gemini"""
    # Wir nutzen das embedding-004 Modell, das sehr effizient ist
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=text,
        task_type="retrieval_document", # Wichtig: Signalisiert, dass dies Dokumente zum Wiederfinden sind
        title="Investment Strategy"
    )
    return result['embedding']

# --- HAUPTPROGRAMM ---
if __name__ == "__main__":
    print("🧠 Starte Ingestion (Simple Mode)...")

    all_chunks = []
    all_embeddings = []

    for url in video_urls:
        text, vid_id = get_transcript_text(url)
        if text:
            chunks = chunk_text(text)
            print(f"Video {vid_id}: {len(chunks)} Chunks erstellt. Generiere Embeddings...")
            
            for chunk in chunks:
                # 1. Text speichern
                all_chunks.append(chunk)
                # 2. Vektor generieren
                emb = get_embedding(chunk)
                all_embeddings.append(emb)

    if all_chunks:
        # Speichern als NumPy Array (sehr schnell und effizient)
        print("Speichere Vektoren...")
        np.save(VECTOR_FILE, np.array(all_embeddings))
        
        # Speichern als JSON (lesbar)
        print("Speichere Text...")
        with open(TEXT_FILE, 'w') as f:
            json.dump(all_chunks, f)
            
        print(f"✅ FERTIG! {len(all_chunks)} Wissens-Häppchen gespeichert.")
        print(f"Dateien erstellt: {VECTOR_FILE}, {TEXT_FILE}")
        print("Du kannst jetzt das Hauptskript starten!")
    else:
        print("❌ Keine Daten geladen. Bitte überprüfe die YouTube URLs.")