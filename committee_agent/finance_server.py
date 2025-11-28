import sys
import os
from mcp.server.fastmcp import FastMCP
import yfinance as yf
import json

# --- FIX: STÖRGERÄUSCHE UNTERDRÜCKEN ---
# Wir leiten stderr (Fehlermeldungen/Logs) temporär um, damit sie die MCP-Leitung nicht verstopfen.
# FastMCP nutzt stderr für eigene Logs, aber yfinance darf dort nicht reinpfuschen.
class SuppressYfinanceLogs:
    def __enter__(self):
        self._original_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr.close()
        sys.stderr = self._original_stderr

# --- SERVER SETUP ---
mcp = FastMCP("FinanceServer")

@mcp.tool()
def get_financial_fundamentals(ticker: str) -> str:
    """
    Holt Finanzdaten.
    Args:
        ticker: Ticker Symbol (z.B. TSLA, BTC-USD).
    """
    # WICHTIG: Wir zwingen den Output zu schweigen, während yfinance arbeitet
    with SuppressYfinanceLogs():
        try:
            ticker = ticker.strip().upper()
            if "BITCOIN" in ticker: ticker = "BTC-USD"
            
            stock = yf.Ticker(ticker)
            # fast_info ist oft schneller und weniger fehleranfällig als info
            info = stock.info 
            
            if not info or ('regularMarketPrice' not in info and 'currentPrice' not in info):
                 return json.dumps({"error": f"Keine Daten für '{ticker}'."})

            data = {
                "name": info.get("shortName", ticker),
                "price": info.get("currentPrice") or info.get("regularMarketPrice"),
                "pe_ratio": info.get("trailingPE"),
                "volume": info.get("volume"),
                "high_52w": info.get("fiftyTwoWeekHigh")
            }
            return json.dumps(data)
        except Exception as e:
            return json.dumps({"error": str(e)})

if __name__ == "__main__":
    mcp.run()