"""FR / EN interface. French is the source text: ``t("...")`` returns it as is, or its English
version when the user picked EN in the sidebar. Display only: nothing sent to the API changes.
"""
from __future__ import annotations

import re

import streamlit as st

LANGUAGES = ("fr", "en")
# Widget values that must survive a language switch (their labels change, so Streamlit
# would otherwise treat them as new widgets and reset them).
_KEEP_ON_SWITCH = ("ingredients", "f_allergies", "f_diet", "f_disliked", "f_top_k", "fridge_sel",
                   "dark_mode", "rate_query")


def lang() -> str:
    return st.session_state.get("lang", "fr")


def is_en() -> bool:
    return lang() == "en"


def t(text: str, **values) -> str:
    """``text`` in the current language; ``{name}`` placeholders are filled from ``values``."""
    out = EN.get(text, text) if is_en() else text
    return out.format(**values) if values else out


def keep_widget_values() -> None:
    for key in _KEEP_ON_SWITCH:
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]


def language_selector() -> None:
    st.radio("Langue / Language", LANGUAGES, key="lang", format_func=str.upper, horizontal=True,
             label_visibility="collapsed", on_change=keep_widget_values)


# The app's event log is stored in French (the admin page counts some messages); these
# patterns translate it for display only.
_LOG = [
    (r"^Application démarrée$", "Application started"),
    (r"^Connexion$", "Login"),
    (r"^Connexion en invité$", "Guest login"),
    (r"^Déconnexion$", "Logout"),
    (r"^Mot de passe changé$", "Password changed"),
    (r"^Échec de connexion \((.*)\)$", lambda m: f"Login failed ({t(m[1])})"),
    (r"^Changement de mot de passe refusé \((.*)\)$",
     lambda m: f"Password change refused ({t(m[1])})"),
    (r"^Compte créé : (\S+) \((.*)\)$", lambda m: f"Account created: {m[1]} ({t(m[2])})"),
    (r"^Profil mis à jour \((.*)\)$", lambda m: f"Profile updated ({m[1]})"),
    (r"^Rôle de (\S+) → (.*)$", lambda m: f"Role of {m[1]} → {t(m[2])}"),
    (r"^Compte (\S+) réactivé$", lambda m: f"Account {m[1]} reactivated"),
    (r"^Compte (\S+) désactivé$", lambda m: f"Account {m[1]} deactivated"),
    (r"^Mot de passe de (\S+) réinitialisé$", lambda m: f"Password of {m[1]} reset"),
    (r"^Compte (\S+) supprimé \(avec ses avis\)$",
     lambda m: f"Account {m[1]} deleted (with its feedback)"),
    (r"^Adresse de l'API changée : (.*)$", lambda m: f"API address changed: {m[1]}"),
]


def t_log(message: str) -> str:
    if is_en():
        for pattern, english in _LOG:
            m = re.match(pattern, message)
            if m:
                return english(m) if callable(english) else english
    return message


def t_actor(actor: str) -> str:
    return t(actor) if actor in ("système", "invité") else actor


EN = {
    # ---------------------------------------------------------------- shared
    "système": "system",
    "invité": "guest",
    "Gluten": "Gluten",
    "Produits laitiers": "Dairy",
    "Fruits à coque": "Tree nuts",
    "Crustacés": "Shellfish",
    "Œufs": "Eggs",
    "Soja": "Soy",
    "Aucun": "None",
    "Végétarien": "Vegetarian",
    "Vegan": "Vegan",
    "Pescétarien": "Pescatarian",
    "Saumon + citron": "Salmon + lemon",
    "Tomate + basilic": "Tomato + basil",
    "Poulet": "Chicken",
    "Chocolat + fraise": "Chocolate + strawberry",
    "J'aime": "Like",
    "Je n'aime pas": "Dislike",
    "Pas avec ça": "Not with this",
    "Cuisiné": "Cooked",
    "Allergique": "Allergic",
    "Plus allergique": "No longer allergic",
    "Ajouté à la liste": "Added to the list",
    "Invité": "Guest",
    "Profil Food.com n°{n}": "Food.com profile #{n}",
    "Utilisateur {id}": "User {id}",
    "Profil exemple n°{n} (id {id})": "Sample profile #{n} (id {id})",
    "Photo de la recette (Food.com)": "Recipe photo (Food.com)",
    "Photo d'un plat similaire (TheMealDB)": "Photo of a similar dish (TheMealDB)",
    "Les ingrédients de la recette": "The recipe's ingredients",
    "Suggestion d'exemple : le vrai modèle n'est pas encore branché.":
        "Sample suggestion: the real model is not connected yet.",
    "🧪 Résultats d'exemple": "🧪 Sample results",
    "Le vrai modèle n'est pas encore branché sur l'API : les suggestions sont tirées au hasard "
    "(mais toujours les mêmes).":
        "The real model is not connected to the API yet: suggestions are random (but stable).",
    "✅ Modèle réel": "✅ Real model",
    "Le serveur de recommandations ne répond pas. Lance-le dans un terminal : "
    "`uvicorn api.main:app`":
        "The recommendation server is not responding. Start it in a terminal: "
        "`uvicorn api.main:app`",
    "Utilisateur": "User",
    "Administrateur": "Administrator",
    "Enregistrer": "Save",
    "Identifiant": "Username",
    "Mot de passe": "Password",
    "Adresse e-mail": "E-mail address",
    "Prénom et nom": "Full name",
    "Noter des recettes": "Rate recipes",
    "Trouver des accords": "Find pairings",
    # ------------------------------------------------------------ menu / sidebar
    "Connexion": "Log in",
    "Mon frigo": "My fridge",
    "Mon historique": "My history",
    "Mon profil": "My profile",
    "Administration": "Administration",
    "À propos": "About",
    "Bienvenue ! Note quelques recettes pour que les suggestions collent à tes goûts "
    "(tu peux passer).":
        "Welcome! Rate a few recipes so suggestions match your taste (you can skip).",
    "Les ingrédients qui vont bien ensemble": "Ingredients that go well together",
    "🌙 Mode sombre": "🌙 Dark mode",
    "Sans compte": "No account",
    "Se déconnecter": "Log out",
    "Se connecter / créer un compte": "Log in / sign up",
    "🟢 Serveur connecté · {ms} ms": "🟢 Server connected · {ms} ms",
    "🔴 Serveur de recommandations injoignable": "🔴 Recommendation server unreachable",
    # ---------------------------------------------------------------- login
    "Bonjour {name} ! 👋": "Hello {name}! 👋",
    "👋 Bienvenue !": "👋 Welcome!",
    "Connecte-toi pour retrouver **tes préférences** (régime, allergies…), **ton frigo** et "
    "**ton historique**. Tu peux aussi essayer l'application sans compte.":
        "Log in to get back **your preferences** (diet, allergies…), **your fridge** and "
        "**your history**. You can also try the app without an account.",
    "Se connecter": "Log in",
    "Créer un compte": "Sign up",
    "ex. marie.d": "e.g. marie.d",
    "3 à 30 caractères : minuscules, chiffres, . _ -":
        "3 to 30 characters: lowercase letters, digits, . _ -",
    "ex. Marie Dupont": "e.g. Marie Dupont",
    "marie@exemple.fr": "marie@example.com",
    "Au moins {n} caractères": "At least {n} characters",
    "Confirmer le mot de passe": "Confirm password",
    "Créer mon compte": "Create my account",
    "Compte créé, bienvenue {name} ! 🎉": "Account created, welcome {name}! 🎉",
    "Continuer sans compte": "Continue without an account",
    "Tu pourras chercher des accords, mais pas enregistrer tes préférences.":
        "You can search for pairings, but not save your preferences.",
    # ---------------------------------------------------------------- pairings
    "Le modèle": "The model",
    "Recettes Food.com": "Food.com recipes",
    "Chimie des arômes": "Aroma chemistry",
    "on te proposera plus souvent cet ingrédient et ceux qui lui ressemblent":
        "this ingredient and similar ones will be suggested more often",
    "il ne te sera plus proposé": "it will not be suggested again",
    "il ne te sera plus proposé avec cette recherche":
        "it will not be suggested again for this search",
    "compté comme un « J'aime »": "counted as a like",
    "ni lui ni ses variantes ne te seront plus jamais proposés":
        "neither it nor its variants will ever be suggested again",
    "merci !": "thanks!",
    "Impossible d'enregistrer ton avis : le serveur ne répond pas.":
        "Could not save your feedback: the server is not responding.",
    "« {label} » enregistré pour {name} : {effect}.": "“{label}” saved for {name}: {effect}.",
    "🍽️ Qu'est-ce qui va bien avec… ?": "🍽️ What goes well with…?",
    "Choisis un ou plusieurs ingrédients que tu as sous la main : on te propose ceux qui "
    "s'accordent le mieux avec, en respectant tes allergies et ton régime.":
        "Pick one or more ingredients you have at hand: we suggest the ones that pair best with "
        "them, respecting your allergies and diet.",
    "Comment ça marche ?": "How does it work?",
    "1. **Choisis tes ingrédients** dans la liste (tape en français ou en anglais).\n"
    "2. Vérifie **tes contraintes** juste en dessous : elles viennent de ton profil, et tu "
    "peux les changer pour une seule recherche.\n"
    "3. Clique sur **Trouver des accords**.\n"
    "4. Donne ton avis sur les suggestions : il est pris en compte **dès ta prochaine "
    "recherche**.\n"
    "   - 👍 **J'aime** : l'ingrédient et ceux qui lui ressemblent remontent ;\n"
    "   - 👎 **Je n'aime pas** : il ne t'est plus proposé ;\n"
    "   - 🚫 **Pas avec ça** : il ne t'est plus proposé avec cette recherche-là ;\n"
    "   - ⚠️ **Allergie** : ni lui ni ses variantes (ex. arachide → beurre de cacahuète) ne te "
    "seront plus jamais proposés ;\n"
    "   - 🍳 **Cuisiné** : compte comme un « J'aime », avec une note.":
        "1. **Pick your ingredients** from the list.\n"
        "2. Check **your constraints** just below: they come from your profile, and you can "
        "change them for a single search.\n"
        "3. Click **Find pairings**.\n"
        "4. Rate the suggestions: your feedback applies **from your next search**.\n"
        "   - 👍 **Like**: the ingredient and similar ones move up;\n"
        "   - 👎 **Dislike**: it is not suggested again;\n"
        "   - 🚫 **Not with this**: it is not suggested again for that search;\n"
        "   - ⚠️ **Allergy**: neither it nor its variants (e.g. peanut → peanut butter) will "
        "ever be suggested again;\n"
        "   - 🍳 **Cooked**: counts as a like, with a note.",
    "Tes ingrédients": "Your ingredients",
    "Ex. saumon, tomate, chocolat…": "e.g. salmon, tomato, chocolate…",
    "Idées :": "Ideas:",
    "⚠️ sans {name}": "⚠️ no {name}",
    "🚫 {n} ingrédient(s) évité(s)": "🚫 {n} avoided ingredient(s)",
    "🔢 {n} suggestions": "🔢 {n} suggestions",
    "**Tes contraintes :** ": "**Your constraints:** ",
    "Modifier pour cette recherche": "Change for this search",
    "Allergies": "Allergies",
    "Aucune": "None",
    "Régime alimentaire": "Diet",
    "Ingrédients à éviter": "Ingredients to avoid",
    "Nombre de suggestions": "Number of suggestions",
    "Changer mes préférences par défaut (profil)": "Change my default preferences (profile)",
    "💡 Crée un compte pour enregistrer ces préférences une fois pour toutes.":
        "💡 Sign up to save these preferences once and for all.",
    "🔍 Trouver des accords": "🔍 Find pairings",
    "On cherche les meilleurs accords…": "Looking for the best pairings…",
    "👆 Ajoute au moins un ingrédient, ou clique sur une idée.":
        "👆 Add at least one ingredient, or click an idea.",
    "💙 tes goûts : {value} %": "💙 your taste: {value}%",
    "Niveau d'accord : {value} %": "Match level: {value}%",
    "Niveau d'accord": "Match level",
    "{icon} Ton avis « {label} » est enregistré.": "{icon} Your “{label}” feedback is saved.",
    "👍 J'aime": "👍 Like",
    "Il remontera, ainsi que les ingrédients qui lui ressemblent":
        "It will move up, along with similar ingredients",
    "👎 Je n'aime pas": "👎 Dislike",
    "Il ne te sera plus proposé (quelle que soit la recherche)":
        "It will not be suggested again (whatever the search)",
    "🚫 Pas avec ça": "🚫 Not with this",
    "Bon ingrédient, mais pas avec cette recherche : il ne te sera plus proposé avec ces "
    "ingrédients-là":
        "Good ingredient, but not for this search: it will not be suggested again with these "
        "ingredients",
    "⚠️ Allergie": "⚠️ Allergy",
    "Je suis allergique à cet ingrédient": "I am allergic to this ingredient",
    "**{name}** et toutes ses variantes ne te seront **plus jamais** proposés, même si tu "
    "cliques ensuite sur 👍.":
        "**{name}** and all its variants will **never** be suggested again, even if you click "
        "👍 afterwards.",
    "Tu pourras annuler depuis ton profil en cas d'erreur.":
        "You can undo this from your profile if it was a mistake.",
    "Oui, j'y suis allergique": "Yes, I am allergic",
    "🍳 Cuisiné": "🍳 Cooked",
    "Qu'as-tu préparé ?": "What did you make?",
    "Ex. saumon au four, sauce aneth et citron": "e.g. baked salmon, dill and lemon sauce",
    "➕ Ajouter": "➕ Add",
    "Ajoute cet ingrédient à ta recherche pour aller plus loin":
        "Add this ingredient to your search to go further",
    "### {n} suggestions pour": "### {n} suggestions for",
    "✨ personnalisé avec ton profil Food.com": "✨ personalised with your Food.com profile",
    "🧠 tes avis sont pris en compte ({details})": "🧠 your feedback is applied ({details})",
    "aimé(s)": "liked",
    "écarté(s)": "ruled out",
    "allergie(s)": "allergy(ies)",
    "réponse en {ms} ms": "answered in {ms} ms",
    "Détail des scores": "Score details",
    "D'où vient chaque suggestion : habitudes culinaires, chimie des arômes, tes goûts.":
        "Where each suggestion comes from: cooking habits, aroma chemistry, your taste.",
    "Aucune suggestion ne respecte toutes tes contraintes. Essaie de retirer une allergie ou "
    "de changer de régime.":
        "No suggestion meets all your constraints. Try removing an allergy or changing diet.",
    # ---------------------------------------------------------------- fridge
    "Frigo non enregistré : {error}": "Fridge not saved: {error}",
    "Noté : {a} + {b} 👍": "Saved: {a} + {b} 👍",
    "🧊 Mon frigo": "🧊 My fridge",
    "Dis-nous ce que tu as chez toi : on te dit **quoi associer avec ce que tu as déjà**, sans "
    "rien avoir à acheter.":
        "Tell us what you have at home: we tell you **what to pair with what you already have**, "
        "with nothing to buy.",
    "Ce que j'ai dans mon frigo et mes placards": "What I have in my fridge and cupboards",
    "Ajoute tes ingrédients (saumon, œufs, citron…)":
        "Add your ingredients (salmon, eggs, lemon…)",
    "Ajout rapide :": "Quick add:",
    "sans {name}": "no {name}",
    "🧊 {n} ingrédient(s)": "🧊 {n} ingredient(s)",
    " · selon ton profil : {constraints}": " · from your profile: {constraints}",
    " · 💡 crée un compte pour garder ton frigo d'une visite à l'autre":
        " · 💡 sign up to keep your fridge between visits",
    "🗑️ Vider": "🗑️ Empty",
    "Retirer tous les ingrédients du frigo ?": "Remove every ingredient from the fridge?",
    "Oui, vider": "Yes, empty it",
    "Ajoute au moins {n} ingrédients pour voir ce que tu peux associer.":
        "Add at least {n} ingredients to see what you can pair.",
    "🍳 Qu'est-ce que je peux faire avec ?": "🍳 What can I make with it?",
    "Mode": "Mode",
    "Autour d'un ingrédient": "Around one ingredient",
    "Les meilleures associations de mon frigo": "The best pairs in my fridge",
    "Ex. « j'ai du saumon, je le mets avec quoi ? »":
        "e.g. “I have salmon, what do I put with it?”",
    "On cherche les duos qui marchent le mieux dans ton frigo":
        "We look for the pairs that work best in your fridge",
    "Ingrédient de base": "Base ingredient",
    "Un ingrédient de ton frigo (🧊) ou n'importe quel autre.":
        "An ingredient from your fridge (🧊) or any other.",
    "🔍 Trouver les associations": "🔍 Find pairings",
    "On fouille ton frigo…": "Searching your fridge…",
    "🔍 Trouver les meilleurs duos": "🔍 Find the best pairs",
    "On teste toutes les combinaisons de ton frigo…":
        "Trying every combination in your fridge…",
    "👍 noté": "👍 saved",
    "Bonne idée !": "Good idea!",
    "Avec ce que tu as, {name} va le mieux avec :": "With what you have, {name} goes best with:",
    "Rien dans ton frigo ne s'accorde avec cet ingrédient en respectant tes contraintes. "
    "Regarde les idées à acheter juste en dessous.":
        "Nothing in your fridge pairs with this ingredient within your constraints. See the "
        "shopping ideas just below.",
    "Non proposés à cause de ton régime, de tes allergies ou de tes avis : ":
        "Not suggested because of your diet, allergies or feedback: ",
    "#### 🛒 Pour aller plus loin, à acheter": "#### 🛒 To go further, worth buying",
    "Ces ingrédients iraient très bien aussi, mais tu ne les as pas.":
        "These would also go very well, but you do not have them.",
    "⭐ **L'ingrédient star de ton frigo : {name}**. C'est celui qui s'accorde avec le plus de "
    "choses que tu as.":
        "⭐ **Your fridge's star ingredient: {name}**. It pairs with the most things you have.",
    "Les meilleurs duos avec ce que tu as": "The best pairs with what you have",
    "Aucune association possible avec tes contraintes actuelles.":
        "No pairing is possible with your current constraints.",
    # ---------------------------------------------------------------- recipes
    "Impossible d'enregistrer ta note : le serveur ne répond pas.":
        "Could not save your rating: the server is not responding.",
    "Note retirée : {name}": "Rating removed: {name}",
    "{stars} pour {name}. Merci, c'est pris en compte !":
        "{stars} for {name}. Thanks, it is taken into account!",
    "Sur Food.com : {mean}{n} avis": "On Food.com: {mean}{n} reviews",
    "⭐ Noter des recettes": "⭐ Rate recipes",
    "Le modèle apprend **tes goûts à partir des recettes que tu notes**, comme pour les "
    "utilisateurs de Food.com : les ingrédients des recettes que tu adores remontent dans tes "
    "suggestions, ceux des recettes que tu n'aimes pas descendent. Plus tu notes, plus c'est "
    "précis.":
        "The model learns **your taste from the recipes you rate**, as it does for Food.com "
        "users: ingredients of recipes you love move up in your suggestions, those of recipes "
        "you dislike move down. The more you rate, the more accurate it gets.",
    "**{n} recette(s) notée(s)**": "**{n} recipe(s) rated**",
    " · encore {n} pour bien démarrer": " · {n} more to get started",
    " · ton profil est lancé ✨": " · your profile is up and running ✨",
    "💡 En invité, tes notes ne valent que pour cette visite. Crée un compte pour les garder.":
        "💡 As a guest, your ratings only last for this visit. Sign up to keep them.",
    "Les recettes viennent de Food.com (noms en anglais). Clique sur les étoiles : 1 = pas du "
    "tout, 5 = j'adore. Clique à nouveau sur la même étoile pour retirer la note.":
        "Recipes come from Food.com. Click the stars: 1 = not at all, 5 = love it. Click the "
        "same star again to remove the rating.",
    "🚀 Pour commencer : 10 recettes populaires et variées":
        "🚀 To start: 10 popular, varied recipes",
    "Note celles que tu connais (ou dont l'idée te plaît ou te déplaît) : elles ont été "
    "choisies pour couvrir un maximum de goûts différents":
        "Rate the ones you know (or whose idea you like or dislike): they were chosen to cover "
        "as many different tastes as possible",
    ", en respectant ton régime et tes allergies.": ", respecting your diet and allergies.",
    "Tu as déjà noté toutes les recettes de départ 🎉":
        "You have already rated all the starter recipes 🎉",
    "C'est bon pour l'instant": "Good for now",
    "Passer, je noterai plus tard": "Skip, I will rate later",
    "🎲 Une recette à noter": "🎲 A recipe to rate",
    "À chaque visite, une nouvelle recette choisie pour découvrir tes goûts sur des "
    "ingrédients qu'on ne connaît pas encore.":
        "At each visit, a new recipe chosen to learn your taste on ingredients we do not know "
        "about yet.",
    "Pour en savoir plus sur tes goûts en : ": "To learn more about your taste for: ",
    "🤷 Je ne connais pas, une autre": "🤷 I don't know it, another one",
    "Plus aucune recette à te proposer pour l'instant. Bravo !":
        "No more recipes to suggest for now. Well done!",
    "🔍 Chercher une recette": "🔍 Search for a recipe",
    "Nom de la recette (en anglais)": "Recipe name",
    "ex. lasagna, chocolate cake, guacamole…": "e.g. lasagna, chocolate cake, guacamole…",
    "Aucune recette trouvée (pense à chercher en anglais : « pancake », « soup »…).":
        "No recipe found (try words such as “pancake” or “soup”).",
    "📋 Mes notes ({n})": "📋 My ratings ({n})",
    "Tu n'as encore noté aucune recette.": "You have not rated any recipe yet.",
    "Retirer cette note": "Remove this rating",
    # ---------------------------------------------------------------- history
    "Connecte-toi pour voir ton historique.": "Log in to see your history.",
    "📜 Mon historique": "📜 My history",
    "⟳ Actualiser": "⟳ Refresh",
    "Tous les avis que tu as donnés sur les suggestions.":
        "All the feedback you gave on suggestions.",
    "Aucun avis pour l'instant. Va sur **Trouver des accords** et clique sur 👍 ou 👎.":
        "No feedback yet. Go to **Find pairings** and click 👍 or 👎.",
    "Date": "Date",
    "Recherche": "Search",
    "depuis mon profil": "from my profile",
    "Suggestion": "Suggestion",
    "Avis": "Feedback",
    "Note": "Note",
    # ---------------------------------------------------------------- profile
    "Connecte-toi pour accéder à ton profil.": "Log in to access your profile.",
    "%d/%m/%Y à %H:%M": "%d/%m/%Y at %H:%M",
    "Modifications enregistrées ✅": "Changes saved ✅",
    "@{user} · {role} · membre depuis le {since} · dernière connexion le {last}":
        "@{user} · {role} · member since {since} · last login {last}",
    "👋 **Première étape :** indique tes **allergies** et ton **régime** ci-dessous. Elles "
    "seront appliquées à toutes tes recherches : les ingrédients incompatibles ne te seront "
    "jamais proposés. Tu pourras les modifier à tout moment. Ensuite, passe par **⭐ Noter "
    "des recettes** pour que les suggestions collent à tes goûts.":
        "👋 **First step:** set your **allergies** and **diet** below. They apply to all your "
        "searches: incompatible ingredients will never be suggested. You can change them at any "
        "time. Then visit **⭐ Rate recipes** so suggestions match your taste.",
    "🥗 Préférences alimentaires": "🥗 Food preferences",
    "🧠 Ce que le modèle sait de toi": "🧠 What the model knows about you",
    "✨ Personnalisation": "✨ Personalisation",
    "📇 Informations": "📇 Information",
    "🔒 Sécurité": "🔒 Security",
    "Ces préférences sont appliquées **automatiquement** à chacune de tes recherches. Les "
    "ingrédients incompatibles ne te seront jamais proposés.":
        "These preferences apply **automatically** to every search. Incompatible ingredients "
        "will never be suggested.",
    "Nombre de suggestions par recherche": "Number of suggestions per search",
    "Allergies et intolérances": "Allergies and intolerances",
    "Ingrédients que je n'aime pas": "Ingredients I dislike",
    "Ils ne te seront jamais proposés. Le bouton « 👎 Je n'aime pas » sur une suggestion fait "
    "la même chose (voir l'onglet « Ce que le modèle sait de toi »).":
        "They will never be suggested. The “👎 Dislike” button on a suggestion does the same "
        "(see the “What the model knows about you” tab).",
    "Enregistrer mes préférences": "Save my preferences",
    "Impossible d'enregistrer : le serveur ne répond pas.":
        "Could not save: the server is not responding.",
    "Le modèle ajuste ses suggestions selon **les recettes que tu notes**, **tes avis** (👍 👎 "
    "⚠️ sur les suggestions)":
        "The model adjusts its suggestions based on **the recipes you rate**, **your feedback** "
        "(👍 👎 ⚠️ on suggestions)",
    " et **les recettes notées par ton profil Food.com**":
        " and **the recipes rated by your Food.com profile**",
    ". Voici ce qu'il retient.": ". Here is what it remembers.",
    "⭐ **{n} recette(s) notée(s)**": "⭐ **{n} recipe(s) rated**",
    " : note-en au moins 10 pour des suggestions vraiment personnalisées.":
        ": rate at least 10 for truly personalised suggestions.",
    "🧪 Le vrai modèle n'est pas chargé sur l'API : seuls tes avis sont listés.":
        "🧪 The real model is not loaded on the API: only your feedback is listed.",
    "##### 💚 Ce qu'il pense que tu aimes": "##### 💚 What it thinks you like",
    "Tes 👍 et les ingrédients qui leur ressemblent": "Your 👍 and similar ingredients",
    ", plus tes recettes Food.com": ", plus your Food.com recipes",
    "Rien pour l'instant : mets 👍 sur des suggestions.":
        "Nothing yet: put a 👍 on some suggestions.",
    "##### 💔 Ce qu'il pense que tu n'aimes pas": "##### 💔 What it thinks you dislike",
    "Rien pour l'instant.": "Nothing yet.",
    "Ton profil Food.com n'a pas de recettes notées connues du modèle : la personnalisation ne "
    "vient que de tes avis.":
        "Your Food.com profile has no rated recipe known to the model: personalisation only "
        "comes from your feedback.",
    "##### Tes avis, un par un": "##### Your feedback, one by one",
    "Tu peux changer d'avis à tout moment : c'est pris en compte dès ta prochaine recherche.":
        "You can change your mind at any time: it applies from your next search.",
    "**👍 Aimés ({n})**": "**👍 Liked ({n})**",
    "👎 Plus maintenant": "👎 Not any more",
    "{} ne te sera plus proposé.": "{} will not be suggested again.",
    "**👎 Écartés ({n})**": "**👎 Ruled out ({n})**",
    "👍 En fait j'aime": "👍 Actually I like it",
    "{} peut de nouveau t'être proposé.": "{} can be suggested to you again.",
    "**⚠️ Allergies ({n})**": "**⚠️ Allergies ({n})**",
    "✅ Pas allergique": "✅ Not allergic",
    "Allergie à {} retirée.": "Allergy to {} removed.",
    "Aucun avis pour l'instant. Va sur **Trouver des accords** et donne ton avis sur les "
    "suggestions.":
        "No feedback yet. Go to **Find pairings** and rate some suggestions.",
    "Les « 🚫 Pas avec ça » ne comptent que pour la recherche où tu les as donnés : ils sont "
    "dans ton historique.":
        "“🚫 Not with this” only counts for the search where you gave it: see your history.",
    "Le modèle apprend les goûts des utilisateurs à partir de **recettes notées**. Le plus "
    "simple : **note des recettes** directement ici. Tu peux aussi relier ton compte à un "
    "profil Food.com existant : ses notes s'ajoutent aux tiennes.":
        "The model learns users' taste from **rated recipes**. The simplest way: **rate "
        "recipes** right here. You can also link your account to an existing Food.com profile: "
        "its ratings are added to yours.",
    "Profil Food.com": "Food.com profile",
    "Aucun : suggestions générales": "None: general suggestions",
    "Autre identifiant Food.com…": "Another Food.com id…",
    "Identifiant Food.com": "Food.com id",
    "ex. 1533": "e.g. 1533",
    "Utilisé seulement si « Autre identifiant » est coché.":
        "Only used if “Another id” is selected.",
    "Indique un identifiant Food.com.": "Enter a Food.com id.",
    "✨ Relié à : {name}": "✨ Linked to: {name}",
    "Identifiant de connexion": "Login username",
    "L'identifiant ne peut pas être modifié.": "The username cannot be changed.",
    "**Changer mon mot de passe**": "**Change my password**",
    "Mot de passe actuel": "Current password",
    "Nouveau mot de passe": "New password",
    "Confirmer le nouveau mot de passe": "Confirm the new password",
    "Changer le mot de passe": "Change password",
    "Mot de passe changé ✅": "Password changed ✅",
    "🔒 Ton mot de passe n'est jamais stocké en clair : seule une empreinte salée est gardée.":
        "🔒 Your password is never stored in clear text: only a salted hash is kept.",
    # ---------------------------------------------------------------- admin
    "⛔ Cette page est réservée aux administrateurs.": "⛔ This page is for administrators only.",
    "ℹ️ info": "ℹ️ info",
    "⚠️ attention": "⚠️ warning",
    "❌ erreur": "❌ error",
    "🛠️ Administration": "🛠️ Administration",
    "Connecté en tant que {name} (@{user})": "Logged in as {name} (@{user})",
    "📊 Vue d'ensemble": "📊 Overview",
    "👥 Utilisateurs": "👥 Users",
    "📜 Journaux": "📜 Logs",
    "🔌 Services & configuration": "🔌 Services & configuration",
    "Serveur (API)": "Server (API)",
    "🟢 En ligne": "🟢 Online",
    "🔴 Hors ligne": "🔴 Offline",
    "Temps de réponse de /health : {ms} ms": "/health response time: {ms} ms",
    "Réponse en {ms} ms": "Answered in {ms} ms",
    "Lancer `uvicorn api.main:app`": "Run `uvicorn api.main:app`",
    "Modèle": "Model",
    "🧪 Factice": "🧪 Mock",
    "✅ Réel": "✅ Real",
    "Factice tant que le vrai modèle n'est pas branché sur /recommend":
        "Mock until the real model is connected to /recommend",
    "Comptes actifs": "Active accounts",
    "dont {n} administrateur(s)": "including {n} administrator(s)",
    "Avis enregistrés": "Saved feedback",
    "Connexions (24 h)": "Logins (24 h)",
    "Échecs de connexion (24 h)": "Failed logins (24 h)",
    "Appels API (session serveur)": "API calls (server session)",
    "Appels API en erreur": "Failed API calls",
    "Dernière activité": "Latest activity",
    "Niveau": "Level",
    "Qui": "Who",
    "Événement": "Event",
    "Nom": "Name",
    "E-mail": "E-mail",
    "Rôle": "Role",
    "Statut": "Status",
    "✅ actif": "✅ active",
    "⛔ désactivé": "⛔ disabled",
    "Régime": "Diet",
    "Créé le": "Created on",
    "Dernière connexion": "Last login",
    "Serveur injoignable : impossible de lire les comptes. Vérifie l'adresse de l'API dans "
    "« Services & configuration ».":
        "Server unreachable: accounts cannot be read. Check the API address in “Services & "
        "configuration”.",
    "Gérer un compte": "Manage an account",
    "Compte": "Account",
    "Rôle de @{user} : {role}": "Role of @{user}: {role}",
    "⛔ Désactiver le compte": "⛔ Disable account",
    "@{user} désactivé": "@{user} disabled",
    "✅ Réactiver le compte": "✅ Reactivate account",
    "@{user} réactivé": "@{user} reactivated",
    "🔑 Réinitialiser le mot de passe": "🔑 Reset password",
    "Nouveau mot de passe temporaire pour @{user} : `{temp}`  \nTransmets-le à l'utilisateur : "
    "il pourra le changer dans son profil.":
        "New temporary password for @{user}: `{temp}`  \nPass it on to the user: they can "
        "change it in their profile.",
    "🗑️ Supprimer le compte": "🗑️ Delete account",
    "Supprimer définitivement @{user} ? Ses avis (j'aime, allergies…) sont supprimés avec le "
    "compte.":
        "Permanently delete @{user}? Their feedback (likes, allergies…) is deleted with the "
        "account.",
    "Oui, supprimer": "Yes, delete",
    "@{user} supprimé": "@{user} deleted",
    "Tu ne peux pas désactiver ni supprimer ton propre compte.":
        "You cannot disable or delete your own account.",
    "Créer": "Create",
    "Compte @{user} créé": "Account @{user} created",
    "Journal de l'application": "Application log",
    "Avis des utilisateurs": "User feedback",
    "Appels à l'API": "API calls",
    "Rechercher": "Search",
    "ex. connexion, camille…": "e.g. login, camille…",
    "⬇️ Exporter (CSV)": "⬇️ Export (CSV)",
    "Serveur de recommandations injoignable : impossible de lire les avis.":
        "Recommendation server unreachable: feedback cannot be read.",
    "Aucun avis enregistré pour l'instant.": "No feedback saved yet.",
    "Tous": "All",
    "N°": "No.",
    "Masquer les vérifications de santé (/health)": "Hide health checks (/health)",
    "Aucun appel enregistré depuis le démarrage du serveur Streamlit.":
        "No call recorded since the Streamlit server started.",
    "**Par route**": "**By route**",
    "**Derniers appels**": "**Latest calls**",
    "méthode": "method",
    "appels": "calls",
    "erreurs": "errors",
    "moyenne_ms": "mean_ms",
    "Serveur de recommandations (API)": "Recommendation server (API)",
    "Adresse de l'API": "API address",
    "Appliquée à tous les utilisateurs de l'application.": "Applies to every user of the app.",
    "Enregistrer l'adresse": "Save address",
    "Adresse de l'API mise à jour": "API address updated",
    "🔌 Tester la connexion": "🔌 Test the connection",
    "injoignable ou statut ≠ ok": "unreachable or status ≠ ok",
    "Route": "Route",
    "Résultat": "Result",
    "Temps": "Time",
    "Objectif du projet : moins de 2 s par recommandation.":
        "Project target: under 2 s per recommendation.",
    "Modèle et données": "Model and data",
    "- Embeddings entraînés : ✅ `artifacts/{file}` ({date})":
        "- Trained embeddings: ✅ `artifacts/{file}` ({date})",
    "- Embeddings entraînés : ❌ absents (lancer `scripts/train_nlgcl.py`)":
        "- Trained embeddings: ❌ missing (run `scripts/train_nlgcl.py`)",
    "- Pondération du classement : habitudes culinaires **{emb}** · chimie des arômes "
    "**{chem}** · goûts **{pref}**":
        "- Ranking weights: cooking habits **{emb}** · aroma chemistry **{chem}** · taste "
        "**{pref}**",
    "- Version du code : `{branch}` @ `{commit}`": "- Code version: `{branch}` @ `{commit}`",
    "Comptes": "Accounts",
    "Les comptes et les avis sont stockés dans la base SQLite de l'API (`data/app.db`, ou "
    "`FOODPAIRING_DB`). Les mots de passe y sont salés et hachés. Le journal de "
    "l'application, lui, est gardé en mémoire et repart de zéro au redémarrage de Streamlit.":
        "Accounts and feedback are stored in the API's SQLite database (`data/app.db`, or "
        "`FOODPAIRING_DB`). Passwords are salted and hashed there. The application log is kept "
        "in memory and starts from scratch when Streamlit restarts.",
    # ---------------------------------------------------------------- about
    "Chef de projet": "Project Lead",
    "Data Engineer": "Data Engineer",
    "ML Engineer — Modèle": "ML Engineer — Model",
    "ML Engineer — Ops": "ML Engineer — Ops",
    "ℹ️ À propos": "ℹ️ About",
    "#### Un projet étudiant de l'**ECE**, ING5, majeure **Data & IA** (2026-2027)":
        "#### A student project at **ECE**, ING5, **Data & AI** major (2026-2027)",
    "**Food Pairing** répond à une question de tous les jours : *« J'ai tel ingrédient, "
    "qu'est-ce qui va bien avec ? »*. Tu donnes un ou plusieurs ingrédients, l'application te "
    "propose ceux qui s'accordent le mieux, en tenant compte de tes **allergies**, de ton "
    "**régime** et de tes **goûts**, avec une **explication** pour chaque suggestion.\n\n"
    "Ce projet est réalisé dans le cadre du cours de **systèmes de recommandation** par le "
    "**groupe 5, « Pull_request »**.":
        "**Food Pairing** answers an everyday question: *“I have this ingredient, what goes "
        "well with it?”*. You give one or more ingredients, and the app suggests the ones that "
        "pair best, taking your **allergies**, **diet** and **taste** into account, with an "
        "**explanation** for each suggestion.\n\n"
        "This project is part of the **recommender systems** course, by **group 5, "
        "“Pull_request”**.",
    "👩‍🍳 L'équipe": "👩‍🍳 The team",
    "🔬 Comment ça marche ?": "🔬 How does it work?",
    "**1. Les données**": "**1. The data**",
    "Des centaines de milliers de recettes et d'avis d'utilisateurs du site **Food.com**, "
    "complétées par les **molécules aromatiques** de chaque ingrédient (FlavorGraph).":
        "Hundreds of thousands of recipes and user reviews from **Food.com**, completed with "
        "the **aroma molecules** of each ingredient (FlavorGraph).",
    "**2. Le modèle**": "**2. The model**",
    "Un **réseau de neurones sur graphe** apprend quels ingrédients vont bien ensemble. Il "
    "s'inspire de **NLGCL** (Xu et al., RecSys 2025), une méthode d'apprentissage contrastif "
    "récente.":
        "A **graph neural network** learns which ingredients go well together. It is inspired "
        "by **NLGCL** (Xu et al., RecSys 2025), a recent contrastive learning method.",
    "**3. Le classement**": "**3. The ranking**",
    "Chaque suggestion combine trois signaux : les **habitudes culinaires**, la **chimie des "
    "arômes** et **tes goûts**. Les ingrédients incompatibles avec tes allergies ou ton régime "
    "sont retirés.":
        "Each suggestion combines three signals: **cooking habits**, **aroma chemistry** and "
        "**your taste**. Ingredients incompatible with your allergies or diet are removed.",
    "📊 En chiffres": "📊 In numbers",
    "Recettes analysées": "Recipes analysed",
    "Ingrédients connus": "Known ingredients",
    "Utilisateurs": "Users",
    "Avis utilisés": "Reviews used",
    "Projet pédagogique, sans but commercial. Les suggestions sont générées automatiquement : "
    "vérifie toujours la composition des produits si tu as une allergie.  \nPhotos des "
    "ingrédients : [TheMealDB](https://www.themealdb.com) (usage pédagogique).":
        "Educational, non-commercial project. Suggestions are generated automatically: always "
        "check product ingredients if you have an allergy.  \nIngredient photos: "
        "[TheMealDB](https://www.themealdb.com) (educational use).",
    # ------------------------------------------------ messages from the app and the API
    "Le serveur ne répond pas : réessaie dans un instant (ou lance `uvicorn api.main:app`).":
        "The server is not responding: try again in a moment (or run `uvicorn api.main:app`).",
    "Les deux mots de passe ne correspondent pas.": "The two passwords do not match.",
    "Action réservée aux administrateurs.": "Action reserved for administrators.",
    "Adresse invalide (ex. http://127.0.0.1:8000).":
        "Invalid address (e.g. http://127.0.0.1:8000).",
    "Identifiant : 3 à 30 caractères, lettres minuscules, chiffres, . _ -":
        "Username: 3 to 30 characters, lowercase letters, digits, . _ -",
    "Adresse e-mail invalide.": "Invalid e-mail address.",
    "Le mot de passe doit faire au moins 8 caractères.":
        "The password must be at least 8 characters long.",
    "Le nom ne peut pas être vide.": "The name cannot be empty.",
    "L'identifiant Food.com doit être un nombre.": "The Food.com id must be a number.",
    "Régime inconnu.": "Unknown diet.",
    "Nombre de suggestions : entre 1 et 50.": "Number of suggestions: between 1 and 50.",
    "Compte introuvable.": "Account not found.",
    "Action réservée aux administrateurs": "Action reserved for administrators",
    "Indique un nom.": "Enter a name.",
    "Rôle inconnu.": "Unknown role.",
    "Cet identifiant est déjà pris.": "This username is already taken.",
    "Cette adresse e-mail est déjà utilisée.": "This e-mail address is already in use.",
    "Identifiant ou mot de passe incorrect.": "Wrong username or password.",
    "Ce compte est désactivé. Contacte un administrateur.":
        "This account is disabled. Contact an administrator.",
    "Mot de passe actuel incorrect.": "Wrong current password.",
    "Impossible : il doit rester au moins un administrateur actif.":
        "Not possible: at least one active administrator must remain.",
    "Tu ne peux pas désactiver ton propre compte.": "You cannot disable your own account.",
    "Tu ne peux pas supprimer ton propre compte.": "You cannot delete your own account.",
    "Plus aucune recette à te proposer.": "No more recipes to suggest.",
    "Recette introuvable.": "Recipe not found.",
    "Le catalogue de recettes n'est pas disponible (data/processed manquant).":
        "The recipe catalogue is not available (data/processed is missing).",
    "Tu n'as pas noté cette recette.": "You have not rated this recipe.",
}
