"""
Dataset-specific content for the Linear Regression Guide.

This module contains all text snippets, LaTeX formulas, descriptions,
and context information that vary depending on the selected dataset.
"""

from typing import Dict, Any


# ============================================================================
# MULTIPLE REGRESSION CONTENT
# ============================================================================

def get_multiple_regression_formulas(dataset_choice_mult: str) -> Dict[str, str]:
    """
    Get LaTeX formulas for multiple regression based on dataset.

    Args:
        dataset_choice_mult: The selected dataset

    Returns:
        Dictionary with 'general' and 'specific' LaTeX formulas
    """
    formulas = {
        "general": r"y_i = \beta_0 + \beta_1 \cdot x_{1i} + \beta_2 \cdot x_{2i} + \cdots + \beta_K \cdot x_{Ki} + \varepsilon_i"
    }

    if dataset_choice == "🏙️ Städte-Umsatzstudie (75 Städte)":
        formulas["specific"] = r"\text{Umsatz}_i = \beta_0 + \beta_1 \cdot \text{Preis}_i + \beta_2 \cdot \text{Werbung}_i + \varepsilon_i"
        formulas["context"] = "Handelskette in 75 Städten"
    elif dataset_choice_mult == "🏠 Häuserpreise mit Pool (1000 Häuser)":
        formulas["specific"] = r"\text{Preis}_i = \beta_0 + \beta_1 \cdot \text{Wohnfläche}_i + \beta_2 \cdot \text{Pool}_i + \varepsilon_i"
        formulas["context"] = "Hausverkäufe in Universitätsstadt"
    elif dataset_choice_mult == "🇨🇭 Schweizer Kantone":
        formulas["specific"] = r"\text{GDP}_i = \beta_0 + \beta_1 \cdot \text{Population}_i + \beta_2 \cdot \text{Foreign \%}_i + \varepsilon_i"
        formulas["context"] = "Schweizer Kantone Sozioökonomie"
    elif dataset_choice_mult == "🌤️ Schweizer Wetter":
        formulas["specific"] = r"\text{Temperature}_i = \beta_0 + \beta_1 \cdot \text{Altitude}_i + \beta_2 \cdot \text{Sunshine}_i + \varepsilon_i"
        formulas["context"] = "Schweizer Klimastationen"
    elif dataset_choice_mult == "🏦 World Bank (Global)":
        formulas["specific"] = r"\text{LifeExp}_i = \beta_0 + \beta_1 \cdot \text{GDP}_i + \beta_2 \cdot \text{Education}_i + \varepsilon_i"
        formulas["context"] = "World Bank Development"
    elif dataset_choice_mult == "💰 FRED (US Economy)":
        formulas["specific"] = r"\text{GDP}_i = \beta_0 + \beta_1 \cdot \text{Unemployment}_i + \beta_2 \cdot \text{Interest}_i + \varepsilon_i"
        formulas["context"] = "US Economic Indicators"
    elif dataset_choice_mult == "🏥 WHO (Health)":
        formulas["specific"] = r"\text{LifeExp}_i = \beta_0 + \beta_1 \cdot \text{Spending}_i + \beta_2 \cdot \text{Sanitation}_i + \varepsilon_i"
        formulas["context"] = "Global Health"
    elif dataset_choice_mult == "🇪🇺 Eurostat (EU)":
        formulas["specific"] = r"\text{GDP}_i = \beta_0 + \beta_1 \cdot \text{Employment}_i + \beta_2 \cdot \text{Education}_i + \varepsilon_i"
        formulas["context"] = "EU Economic Data"
    elif dataset_choice_mult == "🛰️ NASA POWER":
        formulas["specific"] = r"\text{CropYield}_i = \beta_0 + \beta_1 \cdot \text{Temperature}_i + \beta_2 \cdot \text{Solar}_i + \varepsilon_i"
        formulas["context"] = "Agro-Climatology"
    else:  # Elektronikmarkt / Default
        formulas["specific"] = r"\text{Umsatz}_i = \beta_0 + \beta_1 \cdot \text{Fläche}_i + \beta_2 \cdot \text{Marketing}_i + \varepsilon_i"
        formulas["context"] = "Elektronikmarkt-Kette"

    return formulas


def get_multiple_regression_descriptions(dataset_choice_mult: str) -> Dict[str, str]:
    """
    Get descriptions and context for multiple regression based on dataset.
    """
    descriptions = {}

    if dataset_choice_mult == "🏙️ Städte-Umsatzstudie (75 Städte)":
        descriptions["main"] = "Eine Handelskette untersucht in **75 Städten** den Zusammenhang zwischen Produktpreis, Werbeausgaben und Umsatz."
        descriptions["variables"] = {
            "x1": "Produktpreis (in CHF)",
            "x2": "Werbeausgaben (in 1'000 CHF)",
            "y": "Umsatz (in 1'000 CHF)"
        }
    elif dataset_choice_mult == "🏠 Häuserpreise mit Pool (1000 Häuser)":
        descriptions["main"] = "Eine Studie von **1000 Hausverkäufen** in einer Universitätsstadt untersucht den Einfluss von Wohnfläche und Pool auf den Hauspreis."
        descriptions["variables"] = {
            "x1": "Wohnfläche (sqft/10)",
            "x2": "Pool vorhanden (0/1)",
            "y": "Hauspreis (USD)"
        }
    elif dataset_choice_mult == "🇨🇭 Schweizer Kantone":
        descriptions["main"] = "**26 Schweizer Kantone** - Analyse des Zusammenhangs zwischen Bevölkerungsdichte, Ausländeranteil und Wirtschaftskraft."
        descriptions["variables"] = {
            "x1": "Bevölkerungsdichte (pro km²)",
            "x2": "Ausländeranteil (%)",
            "y": "BIP pro Kopf (CHF)"
        }
    elif dataset_choice_mult == "🌤️ Schweizer Wetter":
        descriptions["main"] = "**7 Schweizer Wetterstationen** - Untersuchung der Zusammenhänge zwischen Höhe, Sonnenstunden und Temperatur."
        descriptions["variables"] = {
            "x1": "Höhe über Meer (m)",
            "x2": "Sonnenstunden pro Jahr",
            "y": "Durchschnittstemperatur (°C)"
        }
    elif dataset_choice_mult == "🏦 World Bank (Global)":
        descriptions["main"] = "Analyse von **Entwicklungsindikatoren** weltweit (World Bank Data)."
        descriptions["variables"] = {
            "x1": "GDP per Capita (USD)",
            "x2": "Education Years",
            "y": "Life Expectancy (years)"
        }
    elif dataset_choice_mult == "💰 FRED (US Economy)":
        descriptions["main"] = "Analyse der **US-Wirtschaft** (Federal Reserve Economic Data) über die Zeit."
        descriptions["variables"] = {
            "x1": "Unemployment Rate (%)",
            "x2": "Interest Rate (%)",
            "y": "GDP (Billions USD)"
        }
    elif dataset_choice_mult == "🏥 WHO (Health)":
        descriptions["main"] = "Analyse der **globalen Gesundheitssysteme** (WHO Data)."
        descriptions["variables"] = {
            "x1": "Health Expenditure (USD)",
            "x2": "Sanitation Access (%)",
            "y": "Life Expectancy (years)"
        }
    elif dataset_choice_mult == "🇪🇺 Eurostat (EU)":
        descriptions["main"] = "Vergleich der **EU-Länder** bezüglich Wirtschaft und Bildung."
        descriptions["variables"] = {
            "x1": "Employment Rate (%)",
            "x2": "Tertiary Education (%)",
            "y": "GDP per Capita (EUR)"
        }
    elif dataset_choice_mult == "🛰️ NASA POWER":
        descriptions["main"] = "Einfluss von **Klimadaten** auf landwirtschaftliche Erträge."
        descriptions["variables"] = {
            "x1": "Temperature (°C)",
            "x2": "Solar Radiation (W/m²)",
            "y": "Crop Yield (tons/ha)"
        }
    else:  # Elektronikmarkt / Others
        descriptions["main"] = "Eine Elektronikmarkt-Kette analysiert **50 Filialen** - Zusammenhang zwischen Verkaufsfläche, Marketingbudget und Umsatz."
        descriptions["variables"] = {
            "x1": "Verkaufsfläche (100 qm)",
            "x2": "Marketingbudget (1'000 €)",
            "y": "Umsatz (Mio. €)"
        }

    return descriptions


# ============================================================================
# SIMPLE REGRESSION CONTENT
# ============================================================================

def get_simple_regression_content(dataset_choice: str, x_variable: str) -> Dict[str, Any]:
    """
    Get all content for simple regression based on dataset and x_variable.

    Args:
        dataset_choice: The selected dataset
        x_variable: The selected x variable

    Returns:
        Dictionary with labels, descriptions, formulas, etc.

    Raises:
        ValueError: If dataset_choice or x_variable is invalid
    """
    # Normalize dataset choice mapping if needed (legacy compatibility)
    # This allows this file to work even if strict matching is used
    
    content = {
        "x_label": "X",
        "y_label": "Y",
        "x_unit": "",
        "y_unit": "",
        "context_title": "Regression Analysis",
        "context_description": "Statistical analysis of relationship between variables.",
        "formula_latex": r"y = \beta_0 + \beta_1 \cdot x + \varepsilon"
    }

    # Elektronikmarkt
    if "Elektronikmarkt" in dataset_choice or "electronics" in dataset_choice: # Rough matching
        content.update({
            "y_label": "Umsatz (Mio. €)",
            "y_unit": "Mio. €",
            "context_title": "Elektronikmarkt-Analyse",
            "context_description": """
            Eine Elektronikmarkt-Kette analysiert den Zusammenhang zwischen Verkaufsfläche und Umsatz.
            Die Daten zeigen, wie sich eine Vergrößerung der Verkaufsfläche auf den Umsatz auswirkt.
            """
        })
        if x_variable == "Verkaufsfläche (m²)":
            content["x_label"] = "Verkaufsfläche (m²)"
            content["x_unit"] = "m²"

    # Städte-Umsatzstudie
    elif "Städte" in dataset_choice:
        if x_variable == "Preis (CHF)":
            content.update({
                "x_label": "Preis (CHF)",
                "y_label": "Umsatz (1'000 CHF)",
                "x_unit": "CHF",
                "y_unit": "1'000 CHF",
                "context_title": "Preisstrategie-Analyse",
                "context_description": "Eine Handelskette untersucht den Einfluss des Produktpreises auf den Umsatz."
            })
        else:  # Werbung
            content.update({
                "x_label": "Werbeausgaben (CHF1000)",
                "y_label": "Umsatz (1'000 CHF)",
                "x_unit": "1'000 CHF",
                "y_unit": "1'000 CHF",
                "context_title": "Werbeeffektivität",
                "context_description": "Eine Handelskette untersucht den Einfluss der Werbeausgaben auf den Umsatz."
            })

    # Häuserpreise
    elif "Häuser" in dataset_choice or "Haus" in dataset_choice:
        if x_variable == "Wohnfläche (sqft/10)":
            content.update({
                "x_label": "Wohnfläche (sqft/10)",
                "y_label": "Preis (USD)",
                "x_unit": "sqft/10",
                "y_unit": "USD",
                "context_title": "Wohnflächen-Analyse",
                "context_description": "Untersuchung des Einflusses der Wohnfläche auf den Hauspreis."
            })
        else:  # Pool
            content.update({
                "x_label": "Pool (0/1)",
                "y_label": "Preis (USD)",
                "x_unit": "0/1",
                "y_unit": "USD",
                "context_title": "Pool-Effekt-Analyse",
                "context_description": "Untersuchung des Einflusses eines Pools auf den Hauspreis."
            })

    # Schweizer Kantone
    elif "Kantone" in dataset_choice:
        content["y_label"] = "BIP pro Kopf (CHF)"
        content["y_unit"] = "CHF"
        content["context_title"] = "Schweizer Kantone (Sozioökonomie)"
        
        if x_variable == "Bevölkerungsdichte (Einwohner/km²)" or "Density" in x_variable:
            content.update({
                "x_label": "Bevölkerungsdichte (Einwohner/km²)",
                "x_unit": "Einw./km²",
                "context_description": "Zusammenhang zwischen Bevölkerungsdichte und Wirtschaftskraft."
            })
        elif x_variable == "Ausländeranteil (%)" or "Foreign" in x_variable:
            content.update({
                "x_label": "Ausländeranteil (%)",
                "x_unit": "%",
                "context_description": "Zusammenhang zwischen Ausländeranteil und Wirtschaftskraft."
            })
        elif x_variable == "Unemployment" in x_variable:
             content.update({
                "x_label": "Arbeitslosenquote (%)",
                "x_unit": "%",
                "context_description": "Zusammenhang zwischen Arbeitslosigkeit und Wirtschaftskraft."
            })

    # Schweizer Wetter
    elif "Wetter" in dataset_choice:
        content["y_label"] = "Jahresmitteltemperatur (°C)"
        content["y_unit"] = "°C"
        content["context_title"] = "Schweizer Wetterstationen"
        
        if x_variable == "Höhe über Meer (m)" or "Altitude" in x_variable:
            content.update({
                "x_label": "Höhe über Meer (m)",
                "x_unit": "m",
                "context_description": "Zusammenhang zwischen Höhe und Temperatur."
            })
        elif x_variable == "Sonnenstunden (h/Jahr)" or "Sunshine" in x_variable:
            content.update({
                "x_label": "Sonnenstunden (h/Jahr)",
                "x_unit": "h",
                "context_description": "Zusammenhang zwischen Sonnenstunden und Temperatur."
            })

    # World Bank
    elif "World Bank" in dataset_choice:
        content.update({
            "y_label": "Life Expectancy (years)",
            "y_unit": "years",
            "context_title": "World Bank Development Indicators",
            "context_description": "Analysis of global development metrics."
        })
        if "GDP" in x_variable:
            content["x_label"] = "GDP per Capita (USD)"
            content["x_unit"] = "USD"
        elif "Education" in x_variable:
            content["x_label"] = "Education Years"
            content["x_unit"] = "years"
            
    # FRED
    elif "FRED" in dataset_choice:
        content.update({
            "y_label": "GDP (Billions USD)",
            "y_unit": "B USD",
            "context_title": "US Economic Indicators (FRED)",
            "context_description": "Analysis of US economic performance."
        })
        if "Unemployment" in x_variable:
            content["x_label"] = "Unemployment Rate (%)"
            content["x_unit"] = "%"
        elif "Interest" in x_variable:
            content["x_label"] = "Interest Rate (%)"
            content["x_unit"] = "%"
            
    # WHO
    elif "WHO" in dataset_choice:
        content.update({
            "y_label": "Life Expectancy (years)",
            "y_unit": "years",
            "context_title": "WHO Global Health",
            "context_description": "Analysis of health system performance."
        })
        if "Expenditure" in x_variable or "Spend" in x_variable:
            content["x_label"] = "Health Expenditure (USD)"
            content["x_unit"] = "USD"
        elif "Sanitation" in x_variable:
            content["x_label"] = "Sanitation Access (%)"
            content["x_unit"] = "%"
            
    # Eurostat
    elif "Eurostat" in dataset_choice:
        content.update({
            "y_label": "GDP per Capita (EUR)",
            "y_unit": "EUR",
            "context_title": "Eurostat Economic Data",
            "context_description": "Analysis of EU member states."
        })
        if "Employment" in x_variable:
            content["x_label"] = "Employment Rate (%)"
            content["x_unit"] = "%"
        elif "Education" in x_variable:
            content["x_label"] = "Tertiary Education (%)"
            content["x_unit"] = "%"
            
    # NASA
    elif "NASA" in dataset_choice:
        content.update({
            "y_label": "Crop Yield (tons/ha)",
            "y_unit": "t/ha",
            "context_title": "NASA POWER Agro-Climatology",
            "context_description": "Impact of climate variables on agriculture."
        })
        if "Temperature" in x_variable:
            content["x_label"] = "Temperature (°C)"
            content["x_unit"] = "°C"
        elif "Solar" in x_variable:
            content["x_label"] = "Solar Radiation (W/m²)"
            content["x_unit"] = "W/m²"

    return content


def get_dataset_info(dataset_choice: str) -> Dict[str, Any]:
    """
    Get general information about a dataset.

    Args:
        dataset_choice: The selected dataset

    Returns:
        Dictionary with dataset information
    """
    info = {
        "name": dataset_choice,
        "type": "simulated",
        "source": "Generated",
        "description": "Dataset for regression analysis"
    }

    if "Schweizer" in dataset_choice or "🇨🇭" in dataset_choice or "🌤️" in dataset_choice:
        info.update({
            "type": "real",
            "source": "Switzerland",
            "description": "Authentic Swiss data for educational purposes"
        })
    elif any(api in dataset_choice for api in ["🏦", "💰", "🏥", "🇪🇺", "🛰️", "World Bank", "FRED", "WHO", "Eurostat", "NASA"]):
        info.update({
            "type": "api",
            "source": dataset_choice,  # approx
            "description": "Real data from international organizations"
        })

    return info