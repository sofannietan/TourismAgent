from langchain.tools import tool
from datetime import datetime as dt
import requests
from langchain_community.tools import DuckDuckGoSearchRun

from data.data import *
from data.data import _parse_month, _find_season, _fmt_months , _geocode 
def build_tools(retriever, vectorstore, memory=None):
    def make_search_tool(retriever):
        @tool
        def search_in_corpus(query:str)-> str:
            """Looks for information on Cambodia on Wikipedia docs, and pdf guides. Use for history, culture, food,facts on Cambodia, and traveling tips. """
            docs=retriever.invoke(query)
            return '\n\n'.join(f"[Source: {d.metadata['source']}]\n{d.page_content}" for d in docs)
        return search_in_corpus
    
    @tool
    def convert_currency(montant: float, devise_source: str, devise_cible: str) -> str:
        """Convertit un montant entre deux devises (ex: USD, KHR, EUR)."""
        print(f"💱 [TOOL CONVERTISSEUR] Conversion de {montant} {devise_source} vers {devise_cible}...")
        try:
            url = f"https://open.er-api.com/v6/latest/{devise_source.upper()}"
            response = requests.get(url, timeout=5)
            data = response.json()
            
            if data.get("result") != "success":
                return "Erreur de taux de change."
                
            rates = data.get("rates", {})
            taux = rates.get(devise_cible.upper())
            if not taux:
                return f"Devise cible '{devise_cible}' inconnue."
                
            resultat = montant * taux
            return f"{montant} {devise_source.upper()} = {resultat:,.2f} {devise_cible.upper()} (Taux: {taux})"
        except Exception as e:
            return f"Erreur de conversion : {e}"
 
    @tool
    def get_current_date() -> str:
        """Renvoie la date du jour (AAAA-MM-JJ). À appeler avant get_weather
        ou get_best_season quand l'utilisateur dit 'dans 2 semaines' ou 'le mois prochain'."""
        return dt.date.today().isoformat()

    @tool
    def get_best_season(place: str, month: str = "") -> str:
        """Indique les meilleurs mois et ceux à éviter pour visiter un lieu du Cambodge
        (angkor/siem reap, phnom penh, battambang, kampot/kep, koh rong/sihanoukville,
        mondulkiri/ratanakiri). Si 'month' est donné (nom ou numéro), dit si c'est
        une bonne période. Sans lieu connu, renvoie les saisons générales."""
        key, data = _find_season(place)
        if not data:
            general = " ; ".join(f"{k} : {v}" for k, v in SAISONS_GENERALES.items())
            return (f"Lieu '{place}' non référencé (connus : {', '.join(SEASON_DATA)}). "
                    f"Saisons générales : {general}.")

        lines = [f"{key.title()} : idéal {_fmt_months(data['best'])} ; "
                 f"à éviter {_fmt_months(data['avoid'])}."]
        if month:
            m = _parse_month(month)
            if m is None:
                lines.append(f"Mois '{month}' non reconnu.")
            elif m in data["best"]:
                lines.append(f"{MOIS[m-1].title()} : très bonne période.")
            elif m in data["avoid"]:
                lines.append(f"{MOIS[m-1].title()} : période déconseillée.")
            else:
                lines.append(f"{MOIS[m-1].title()} : période correcte, sans être idéale.")
        lines.append(data["note"])
        return " ".join(lines)

    @tool
    def get_weather(city: str, start_date: str = "", days: int = 3) -> str:
        """Météo d'une ville du Cambodge via Open-Meteo. start_date au format AAAA-MM-JJ
        (par défaut aujourd'hui), days entre 1 et 7. Prévisions réelles jusqu'à 16 jours ;
        au-delà, renvoie ce qu'il a fait à la même période l'an dernier (indicatif)."""
        days = max(1, min(days, 7))
        try:
            start = dt.date.fromisoformat(start_date) if start_date else dt.date.today()
        except ValueError:
            return "Date invalide : utilise le format AAAA-MM-JJ."
        end = start + dt.timedelta(days=days - 1)

        try:
            coords = _geocode(city)
            if not coords:
                return f"Ville '{city}' introuvable."
            lat, lon = coords

            if (start - dt.date.today()).days <= 16 and start >= dt.date.today():
                url, kind = "https://api.open-meteo.com/v1/forecast", "Prévision"
                daily = "temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max"
                s, e = start, end
            else:
                # Hors fenêtre de prévision : même période l'an dernier
                url, kind = "https://archive-api.open-meteo.com/v1/archive", "Historique (même période, an dernier)"
                daily = "temperature_2m_max,temperature_2m_min,precipitation_sum"
                s = start.replace(year=start.year - 1) if start >= dt.date.today() else start
                e = s + dt.timedelta(days=days - 1)

            r = requests.get(url, params={
                "latitude": lat, "longitude": lon, "daily": daily,
                "start_date": s.isoformat(), "end_date": e.isoformat(),
                "timezone": "Asia/Phnom_Penh",
            }, timeout=10)
            r.raise_for_status()
            d = r.json()["daily"]
        except Exception as ex:
            return f"Météo indisponible ({type(ex).__name__}). Réessaie ou utilise get_best_season."

        lines = [f"{kind} pour {city.title()} :"]
        for i, day in enumerate(d["time"]):
            pluie = f"{d['precipitation_sum'][i]} mm"
            if "precipitation_probability_max" in d:
                pluie += f" (risque {d['precipitation_probability_max'][i]} %)"
            lines.append(f"- {day} : {d['temperature_2m_min'][i]}-{d['temperature_2m_max'][i]} °C, pluie {pluie}")
        return "\n".join(lines)
    seach_tool= make_search_tool(retriever=retriever)
    return [seach_tool, convert_currency, get_current_date, get_best_season, get_weather]
