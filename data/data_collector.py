import wikipediaapi

# Initialisation avec un User-Agent personnalisé (OBLIGATOIRE pour l'API Wikipédia)
wiki = wikipediaapi.Wikipedia(
    user_agent="TourismAgentBot/1.0 (contact@example.com)", 
    language="en"
)

sujets = [
    "Cambodia",
    "Khmer Empire",
    "Angkor Wat", 
    "Siem Reap",
    "Sihanoukville",
    "Phnom Penh",
    "Kampot",
    "Preah Vihear Temple",
    "Tonle Sap",
    "Cambodian cuisine"
]

for sujet in sujets:
    page = wiki.page(sujet)
    
    if page.exists():
        nom_fichier = f"{sujet.lower().replace(' ', '_').replace('(', '').replace(')', '')}.txt"
        contenu = f"# {page.title}\n\n{page.text}"
        
        with open(nom_fichier, "w", encoding="utf-8") as f:
            f.write(contenu)
            
        print(f"✅ Page '{page.title}' sauvegardée avec succès ({nom_fichier}).")
    else:
        print(f"❌ La page '{sujet}' est introuvable sur Wikipédia.")

print("Téléchargement terminé !")
