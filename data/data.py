import datetime as dt
import requests

MOIS = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet",
        "août", "septembre", "octobre", "novembre", "décembre"]

# Coordonnées codées en dur : évite les ambiguïtés du géocodage
COORDS = {
    "phnom penh": (11.5564, 104.9282),
    "siem reap": (13.3633, 103.8564),
    "angkor": (13.4125, 103.8670),
    "battambang": (13.0957, 103.2022),
    "kampot": (10.6104, 104.1810),
    "kep": (10.4829, 104.3167),
    "sihanoukville": (10.6253, 103.5234),
    "koh rong": (10.7214, 103.2325),
    "mondulkiri": (12.4551, 107.1881),
    "ratanakiri": (13.7340, 106.9870),
}

# Mois en numéros. Valeurs INDICATIVES : à recouper avec tes guides.
SEASON_DATA = {
    "angkor": {
        "aliases": ["siem reap", "angkor wat"],
        "best": [11, 12, 1, 2], "avoid": [4, 5],
        "note": "Saison fraîche et sèche idéale. Mars-mai très chaud ; en saison des pluies, "
                "les temples sont moins fréquentés et la végétation est plus verte.",
    },
    "phnom penh": {
        "aliases": [],
        "best": [11, 12, 1, 2], "avoid": [4, 5],
        "note": "Ville visitable toute l'année, mais chaleur étouffante en avril-mai.",
    },
    "battambang": {
        "aliases": [],
        "best": [11, 12, 1, 2], "avoid": [4, 5],
        "note": "Routes parfois difficiles en pleine saison des pluies.",
    },
    "kampot": {
        "aliases": ["kep"],
        "best": [11, 12, 1, 2, 3], "avoid": [9, 10],
        "note": "Côte plus pluvieuse en septembre-octobre ; saison sèche idéale pour la rivière et Bokor.",
    },
    "koh rong": {
        "aliases": ["sihanoukville", "koh rong samloem"],
        "best": [11, 12, 1, 2, 3], "avoid": [7, 8, 9, 10],
        "note": "Mer agitée et visibilité réduite en saison des pluies ; certaines activités nautiques s'arrêtent.",
    },
    "mondulkiri": {
        "aliases": ["ratanakiri"],
        "best": [11, 12, 1, 2], "avoid": [8, 9],
        "note": "Hauts plateaux plus frais. Cascades plus belles en saison des pluies, mais pistes difficiles.",
    },
}

SAISONS_GENERALES = {
    "saison fraîche et sèche": "novembre à février",
    "saison chaude": "mars à mai",
    "saison des pluies": "juin à octobre",
}


def _parse_month(value: str):
    """Accepte '11', 'novembre', 'nov'... et renvoie 1-12 ou None."""
    v = (value or "").strip().lower()
    if not v:
        return None
    if v.isdigit() and 1 <= int(v) <= 12:
        return int(v)
    for i, nom in enumerate(MOIS, start=1):
        if nom.startswith(v[:3]):
            return i
    return None


def _find_season(place: str):
    p = place.lower().strip()
    for key, data in SEASON_DATA.items():
        if p == key or p in data["aliases"]:
            return key, data
    return None, None


def _fmt_months(nums):
    return ", ".join(MOIS[n - 1] for n in sorted(nums, key=lambda n: (n - 11) % 12))


def _geocode(city: str):
    key = city.lower().strip()
    if key in COORDS:
        return COORDS[key]
    r = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1, "countryCode": "KH"}, timeout=10,
    )
    results = r.json().get("results")
    return (results[0]["latitude"], results[0]["longitude"]) if results else None