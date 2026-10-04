"""Default screening universe: companies that sit at a physical choke point of
the AI / data-center / electrification build-out.

The "MU pattern" is a supplier of a scarce, hard-to-expand input (HBM / DRAM)
whose pricing power shows up first in gross margins and inventory days, then
in FCF, while the stock still trades on trough-cycle multiples. The groups
below are the places that pattern is most likely to repeat. Edit freely, or
type your own tickers into the dashboard.
"""

UNIVERSE = {
    "Memory & Storage": ["MU", "WDC", "STX", "SNDK", "SIMO"],
    "Logic / Accelerators": ["NVDA", "AMD", "AVGO", "MRVL", "ARM", "ALAB", "CRDO"],
    "Foundry & Packaging": ["TSM", "INTC", "AMKR", "GFS", "UMC"],
    "Semi Equipment": ["ASML", "AMAT", "LRCX", "KLAC", "TER", "ONTO", "CAMT", "ACMR", "AEHR", "FORM", "KLIC"],
    "Optical & Networking": ["COHR", "LITE", "AAOI", "FN", "CIEN", "ANET", "CSCO", "NOK"],
    "Substrates & Components": ["ENTG", "MKSI", "AEIS", "VSH", "APH", "TTMI", "SANM", "CLS"],
    "Power & Cooling": ["VRT", "ETN", "GEV", "PWR", "MOD", "NVT", "POWL", "HUBB", "BE"],
    "Power Generation & Fuel": ["CEG", "VST", "TLN", "NRG", "CCJ", "LEU", "OKLO", "SMR"],
    "Materials & Metals": ["FCX", "SCCO", "MP", "ALB", "TECK"],
}

BENCHMARK = "SPY"
SECTOR_BENCHMARK = "SOXX"


def all_tickers():
    seen, out = set(), []
    for tickers in UNIVERSE.values():
        for t in tickers:
            if t not in seen:
                seen.add(t)
                out.append(t)
    return out


def group_of(ticker):
    for group, tickers in UNIVERSE.items():
        if ticker in tickers:
            return group
    return "Custom"
