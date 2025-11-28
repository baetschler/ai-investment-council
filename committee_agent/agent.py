# %%
import os
import asyncio
import json
from dotenv import load_dotenv
import numpy as np 


# Lade Environment Variablen (.env Datei muss GOOGLE_API_KEY enthalten)
load_dotenv()

# Prüfen ob API Key gesetzt ist
if not os.environ.get("GOOGLE_API_KEY"):
    print("❌ Warnung: GOOGLE_API_KEY nicht gefunden.")

from google.adk.agents import Agent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, google_search
from google.genai import types
from youtube_transcript_api import YouTubeTranscriptApi

#MCP Imports
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters
import sys

print("✅ ADK components imported successfully.")

# --- KONFIGURATION ---
# Wir nutzen JSON Mode für die Sub-Agenten, um Parsing-Fehler zu vermeiden.
json_generation_config = {
    "response_mime_type": "application/json"
}

# --- SIMPLE RAG SETUP ---
# Wir laden die JSON/Numpy Dateien
VECTOR_FILE = "xo_vectors.npy"
TEXT_FILE = "xo_text.json"
RAG_ENABLED = False
knowledge_vectors = None
knowledge_texts = None

try:
    if os.path.exists(VECTOR_FILE) and os.path.exists(TEXT_FILE):
        knowledge_vectors = np.load(VECTOR_FILE)
        with open(TEXT_FILE, 'r') as f:
            knowledge_texts = json.load(f)
        RAG_ENABLED = True
        print(f"🧠 XO Brain geladen: {len(knowledge_texts)} Einträge.")
    else:
        print("⚠️ Warnung: Brain-Dateien nicht gefunden. Bitte erst ingest_knowledge.py ausführen.")
except Exception as e:
    print(f"⚠️ Fehler beim Laden des Brains: {e}")

# Standard Retry Config
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=2,
    initial_delay=10,
    http_status_codes=[429, 500, 503]
)

# Modell-Wahl: Nutze 2.5-flash für kostenoptimierte Ausführung
MODEL_NAME = "gemini-2.5-flash" 

# --- SYSTEM PROMPTS ---

PRICE_PROVIDER_SYSTEM = """
Du bist 'price_provider', eine spezialisierte Daten-KI.
Deine Aufgabe:
2. Nutze finance_mcp_tools, um Finanzdaten abzurufen.
3. Konsolidiere alle Daten in einem JSON Objekt mit folgendem Format und gib es zurück.:

Format:
{
  "price": number or null,
  "volume_24h": number or null,
}
"""

TECHNICALS_PROVIDER_SYSTEM = """
Du bist 'technicals_provider', eine spezialisierte Daten-KI.
Deine Aufgabe:
2. Nutze die websuche um Finanzdaten abzurufen.
3. Konsolidiere alle Daten in einem JSON Objekt mit folgendem Format und gib es zurück.:

Format:
{
  "market_cap": string or null,
  "pe_ratio": number or null,
  "rsi_1d": number or null,
  "rsi_1w": number or null,
  "rsi_1m": number or null,
  "Federal_Reserve_policy",
  "macro_factors": [
    "Federal_Reserve_policy": number or null,
    "Economic_growth": number or null,
    "Interest_rates": number or null,
    "Inflation": number or null,
    "Liquidity": number or null,
  ]
}
"""

XO_SYSTEM = """
Du bist 'xo', ein technischer Trader (Persona: TraderXO).
Du hast Zugriff auf dein eigenes "Gehirn" via `consult_xo_brain`.

Input: 
- JSON Daten (asset_data) vom Research Agent.

Workflow:
1. Analysiere den Input.
2. Nutze dein Wissen aus `consult_xo_brain`, um relevante Konzepte zu finden oder konkretes Wissen zu gefragten Assets.
3. Integriere das gefundene Wissen in deine Antwort.

Gib NUR valides JSON zurück.

Definitionen Zeiträume:
- Longterm: (3-6 Monate)
- Shortterm: (1-4 Wochen)

Definition Richtungen:
- bullish: Erwartung steigender Kurse
- bearish: Erwartung fallender Kurse
- neutral: Seitwärtsmarkt

Format:
{
  "longterm_setup": "bullish" | "bearish" | "neutral" | "unknown",
  "longterm_confidence": int (0-100),
  "shortterm_setup": "bullish" | "bearish" | "neutral" | "unknown",
  "shortterm_confidence": int (0-100),
  "xo-reasoning": "Erklärung (zitiere XO Wissen wenn möglich), Maximal 2 Sätze"
}
"""

GLOBAL_SYSTEM = """
Du bist 'global', eine Makro-KI.
Input: 
- JSON Daten (macro_factors).
- ggf. technische Setups vom XO Agent.

Aufgabe: Bestimme Makro-Regime und Liquidität. Beziehe auch Stellung zur Meinung der anderen Agenten.
Gib NUR valides JSON zurück.

Format:
{
  "macro_regime": "risk_on" | "neutral" | "risk_off",
  "liquidity": int (0-100)
  "global-reasoning": "Erklärung, Maximal 2 Sätze"
}
"""

WARREN_SYSTEM = """
Du bist 'warren', ein Value-Investor.
Input: 
- JSON Daten (KGV, Market Cap).
- ggf. Einschätzung der Makro-Lage vom Global Agent.
- ggf. technische Setups vom XO Agent.

Aufgabe: Value-Einschätzung. Beziehe auch Stellung zur Meinung der anderen Agenten.
Gib NUR valides JSON zurück.

Format:
{
  "value_opinion": "undervalued" | "fair" | "overvalued" | "unknown",
  "quality": int (0-100)
  "warren-reasoning": "Erklärung, Maximal 2 Sätze"
}
"""

ROUNDTABLE_SYSTEM = """
Du bist 'roundtable', der Moderator des Investment-Committees.

INPUT:
Du erhältst JSON-Daten (Research-Ergebnisse) aus dem vorherigen Schritt der Pipeline.

AUFGABE:
1. Analysiere die Input-Daten.
2. Befrage nacheinander deine Experten-Tools, um ihre Meinungen einzuholen.
Du möchtest eine Diskussion zwischen den Experten anregen:
   - Rufe `XO_Agent` auf. 
   - Rufe `Global_Agent` auf.
   - Rufe `Warren_Agent` auf.

   WICHTIG: Übergib an jedes Tool die *gesamten* Research-Daten, die du als Input erhalten hast und die Antworten der vorherigen Experten.

   Mache 2 Runden der Befragung, um eine tiefere Diskussion zu ermöglichen.
   
3. Nachdem alle 3 jeweils 2 Mal geantwortet haben, erstelle eine JSON-Liste mit Strings.
   Jeder String soll die Kernaussage eines Experten + ein Fazit enthalten.

OUTPUT FORMAT (JSON List):
[
  "TraderXO: Shortterm Setup: [shortterm_setup] ([shortterm_confidence]%) - Longterm Setup:[longterm_setup] ([longterm_confidence]%)  - Reasoning: [reasoning]",
  "Global: [Regime] - Liquidity: [Score]",
  "Warren: [Meinung] - Quality: [Score]",
  "Fazit: [Deine Zusammenfassung]"
]
"""

# --- AGENT DEFINITIONS ---

# 1. Verbindungseinstellungen für MCP Tool
# 2. Verbindung konfigurieren
# Das sagt dem Agenten: "Starte python finance_server.py und rede mit ihm."
server_params = StdioServerParameters(
    command=sys.executable,         # Der Befehl
    args=["committee_agent/finance_server.py"]  # Das Skript 
)

# 2. Dann das Paket übergeben (an 'server_params', nicht direkt command/args)
connection_params = StdioConnectionParams(
    server_params=server_params
)

# 3. Das Toolset erstellen
finance_mcp_tools = McpToolset(
    connection_params=connection_params
)

# 1. Price Agent (Holt Price + Volumen)
price_agent = Agent(
    name="Price_Agent",
    model=Gemini(model=MODEL_NAME, retry_options=retry_config, generation_config=json_generation_config),
    instruction=PRICE_PROVIDER_SYSTEM,
    tools=[finance_mcp_tools],
    output_key="price_data" 
)

# 2. Technial Agent (Holt weitere Asset Daten via Websuche)
technicals_agent = Agent(
    name="Technicals_Agent",
    model=Gemini(model=MODEL_NAME, retry_options=retry_config, generation_config=json_generation_config),
    instruction=TECHNICALS_PROVIDER_SYSTEM,
    tools=[google_search],
    output_key="technicals_data" 
)

# 2. Expert Agents (Werden als Tools genutzt)
# Hinweis: Tools brauchen keine eigene 'output_key' Definition im globalen State, 
# da sie ihre Antwort direkt an den Aufrufer (Roundtable) zurückgeben.


try:
    if os.path.exists(VECTOR_FILE) and os.path.exists(TEXT_FILE):
        knowledge_vectors = np.load(VECTOR_FILE)
        with open(TEXT_FILE, 'r') as f:
            knowledge_texts = json.load(f)
        RAG_ENABLED = True
        print(f"🧠 XO Brain geladen: {len(knowledge_texts)} Einträge.")
    else:
        print("⚠️ Warnung: Brain-Dateien nicht gefunden. Bitte erst ingest_knowledge.py ausführen.")
except Exception as e:
    print(f"⚠️ Fehler beim Laden des Brains: {e}")

def consult_xo_brain(query: str) -> str:
    """
    Sucht im Wissen von TraderXO nach relevanten Konzepten.
    Berechnet Cosine Similarity zwischen Query und gespeicherten Chunks.
    """
    if not RAG_ENABLED or knowledge_vectors is None:
        return "Datenbank nicht verfügbar."
    
    try:
        # 1. Query Embedding erstellen
        query_emb_result = genai.embed_content(
            model="models/text-embedding-004",
            content=query,
            task_type="retrieval_query"
        )
        query_vec = np.array(query_emb_result['embedding'])
        
        # 2. Cosine Similarity berechnen (Vector * Query)
        # Dot product reicht hier meist für Ranking
        scores = np.dot(knowledge_vectors, query_vec)
        
        # 3. Top 3 Indizes finden
        top_indices = np.argsort(scores)[-3:][::-1]
        
        # 4. Ergebnisse zurückgeben
        results = []
        for idx in top_indices:
            results.append(f"- {knowledge_texts[idx]}")
            
        knowledge_block = "\n".join(results)
        return f"Gefundenes Wissen aus XO's Strategie:\n{knowledge_block}"
        
    except Exception as e:
        return f"Fehler bei der Suche: {str(e)}"

xo_agent = Agent(
    name="XO_Agent",
    model=Gemini(model=MODEL_NAME, retry_options=retry_config, generation_config=json_generation_config),
    instruction=XO_SYSTEM,
    tools=[consult_xo_brain] 
)

global_agent = Agent(
    name="Global_Agent",
    model=Gemini(model=MODEL_NAME, retry_options=retry_config, generation_config=json_generation_config),
    instruction=GLOBAL_SYSTEM
)

warren_agent = Agent(
    name="Warren_Agent",
    model=Gemini(model=MODEL_NAME, retry_options=retry_config, generation_config=json_generation_config),
    instruction=WARREN_SYSTEM
)

# 3. Roundtable Agent (Orchestrator)
# Er nutzt die anderen Agenten als Tools via AgentTool wrapper.
roundtable_agent = Agent(
    name="RoundtableAgent",
    model=Gemini(model=MODEL_NAME, retry_options=retry_config, generation_config=json_generation_config),
    instruction=ROUNDTABLE_SYSTEM,
    tools=[
        AgentTool(xo_agent), 
        AgentTool(global_agent), 
        AgentTool(warren_agent)
    ],
    output_key="final_verdict"
)

# 4. Pipeline
# SequentialAgent übergibt automatisch den Output von Agent 1 als Input an Agent 2
root_agent = SequentialAgent(
    name="root_agent",
    sub_agents=[price_agent, technicals_agent, roundtable_agent],
)

# --- EXECUTION ---

async def main():
    print("\n🚀 Starte Investment Council Pipeline...")
    
    runner = InMemoryRunner(agent=root_agent)
    
    # User Query
    query = "Invest in Bitcoin now?"
    print(f"❓ Frage: {query}\n")

    # run_debug gibt eine LISTE von Events/Turns zurück, nicht ein einzelnes Response-Objekt
    response_list = await runner.run_debug(query)
    
    print("--- 🏁 PIPELINE FINISHED ---")
    
    print("\n📊 RESULTAT (Debug Trace Analysis):")
    
    if isinstance(response_list, list):
        print(f"✅ Pipeline erfolgreich durchlaufen ({len(response_list)} Schritte).")
        
        # Wir versuchen, den letzten relevanten Text zu finden.
        # Das letzte Element in der Liste ist oft der finale State oder Output.
        try:
            last_step = response_list[-1]
            print("\n🔍 Letzter Schritt (Rohdaten):")
            print(last_step)
            
            # Falls es ein Objekt mit .text Attribut ist (manche SDK Versionen)
            if hasattr(last_step, 'text'):
                print(f"\n🤖 Final Answer:\n{last_step.text}")
            # Falls es ein Dictionary ist
            elif isinstance(last_step, dict) and 'text' in last_step:
                print(f"\n🤖 Final Answer:\n{last_step['text']}")
                
        except Exception as e:
            print(f"Konnte letzten Schritt nicht parsen: {e}")
    else:
        # Fallback, falls sich der Return-Type ändert
        try:
            print(response_list.text)
        except AttributeError:
            print(response_list)

if __name__ == "__main__":
    asyncio.run(main())