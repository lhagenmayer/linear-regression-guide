"""
ML Fundamentals Educational Content - Framework Agnostic.

This module covers foundational ML concepts that bridge statistics to ML:
KNN, Data Scaling, Curse of Dimensionality, Feature Engineering.
"""

from typing import Dict, Any, List
from dataclasses import dataclass

from .structure import (
    ContentElement, Chapter, Section, EducationalContent,
    Markdown, Metric, MetricRow, Formula, Plot, Table,
    Columns, Expander, InfoBox, WarningBox, SuccessBox,
    CodeBlock, Divider
)
from .builder import ContentBuilder


class MLFundamentalsContent(ContentBuilder):
    """
    Builds educational content for ML fundamentals.
    
    Covers: KNN, Data Scaling, Curse of Dimensionality, Feature Engineering.
    """
    
    def build(self) -> EducationalContent:
        """Build all chapters of ML fundamentals content."""
        return EducationalContent(
            title="📊 Machine Learning Fundamentals",
            subtitle="KNN, Scaling, and Feature Engineering",
            chapters=[
                self._chapter_9_0_ml_motivation(),
                self._chapter_9_1_ml_types(),
                self._chapter_9_2_knn(),
                self._chapter_9_3_data_scaling(),
                self._chapter_9_4_curse_of_dimensionality(),
                self._chapter_9_5_feature_engineering(),
                self._chapter_9_6_multiclass(),
            ]
        )
    
    # =========================================================================
    # CHAPTER 9.0: ML MOTIVATION & HISTORY
    # =========================================================================
    def _chapter_9_0_ml_motivation(self) -> Chapter:
        """Chapter 9.0: Why ML and brief history."""
        
        return Chapter(
            number="9.0",
            title="Motivation: Warum Machine Learning?",
            icon="🎯",
            sections=[
                InfoBox("""
**Kernidee:** Machine Learning löst Probleme, die zu komplex sind, 
um sie explizit zu programmieren.

Statt Regeln manuell zu definieren, **lernt** das System aus Daten.
"""),
                
                Expander("💡 Praktische Beispiele", [
                    Table(
                        headers=["Problem", "Traditionell", "ML-Lösung"],
                        rows=[
                            ["Spam-Filter", "1000+ handgeschriebene Regeln", "Lernt Muster aus Beispielen"],
                            ["Bilderkennung", "Pixel-Regeln? Unmöglich!", "CNN lernt Features automatisch"],
                            ["Turbinenausfall", "Physik-Modelle aufwendig", "Vorhersage aus Sensordaten"],
                            ["Textübersetzung", "Wörterbuch + Grammatik", "Transformer lernt Kontext"],
                        ]
                    ),
                ], expanded=True),
                
                Expander("📜 Kurze Geschichte der KI", [
                    Markdown("""
| Jahr | Meilenstein |
|------|-------------|
| 1950 | Turing: "Can machines think?" |
| 1956 | Dartmouth Workshop - Geburt der KI |
| 1958 | Perceptron - erstes neuronales Netz |
| 1969-1980 | AI Winter (XOR-Problem) |
| 1986 | Backpropagation wiederentdeckt |
| 1997 | Deep Blue besiegt Kasparov |
| 2012 | AlexNet - Deep Learning Revolution |
| 2017 | Transformer ("Attention is all you need") |
| 2022+ | GPT, BERT, generative AI |
"""),
                ]),
                
                Expander("🔗 Verbindung zur Statistik", [
                    Markdown("""
**Machine Learning baut auf Statistik auf:**

| Statistik | Machine Learning |
|-----------|-----------------|
| Lineare Regression | Gradient Descent optimiert MSE |
| Likelihood-Schätzung | Maximum Likelihood für Log. Reg. |
| Bayes-Theorem | Naive Bayes Classifier |
| Varianzanalyse | Feature Importance |
"""),
                    SuccessBox("Ihre statistischen Grundlagen sind die perfekte Basis für ML!"),
                ]),
            ]
        )
    
    # =========================================================================
    # CHAPTER 9.1: ML TYPES
    # =========================================================================
    def _chapter_9_1_ml_types(self) -> Chapter:
        """Chapter 9.1: Types of machine learning."""
        
        return Chapter(
            number="9.1",
            title="ML-Lerntypen: Supervised, Unsupervised, Reinforcement",
            icon="📚",
            sections=[
                Markdown("""
Machine Learning wird nach **zwei Dimensionen** kategorisiert:
1. Art der Zielgröße (kontinuierlich vs. kategorisch)
2. Verfügbarkeit von Labels (überwacht vs. unüberwacht)
"""),
                
                Expander("🎓 Überwachtes Lernen (Supervised)", [
                    Markdown("""
**Definition:** Trainingsdaten haben Labels (Y-Werte).

Das Modell lernt die Abbildung: X → Y
"""),
                    Table(
                        headers=["Aufgabe", "Zielgröße", "Beispiel"],
                        rows=[
                            ["Regression", "Kontinuierlich", "Hauspreis vorhersagen"],
                            ["Klassifizierung", "Kategorisch", "Spam erkennen"],
                        ]
                    ),
                    InfoBox("Das ist unser Fokus: Regression und Klassifizierung sind supervised learning!"),
                ], expanded=True),
                
                Expander("🔍 Unüberwachtes Lernen (Unsupervised)", [
                    Markdown("""
**Definition:** Trainingsdaten haben **keine** Labels.

Das Modell findet **Struktur** in den Daten selbst.
"""),
                    Table(
                        headers=["Aufgabe", "Ziel", "Beispiel"],
                        rows=[
                            ["Clustering", "Ähnliche Gruppen finden", "Kundensegmentierung"],
                            ["Dimensionsreduktion", "Features komprimieren", "PCA"],
                            ["Anomaly Detection", "Ausreißer finden", "Betrugserkennung"],
                        ]
                    ),
                ]),
                
                Expander("🎮 Reinforcement Learning", [
                    Markdown("""
**Definition:** Agent lernt durch Trial-and-Error mit Belohnung/Bestrafung.

**Beispiele:**
- AlphaGo (Go spielen)
- Roboter-Steuerung
- Autonomes Fahren
"""),
                ]),
                
                Expander("📊 ML-Typen Matrix", [
                    Table(
                        headers=["", "Kontinuierliche Zielgröße", "Kategorische Zielgröße"],
                        rows=[
                            ["**Supervised**", "Regression", "Klassifizierung"],
                            ["**Unsupervised**", "Dimensionsreduktion", "Clustering"],
                        ]
                    ),
                ]),
            ]
        )
    
    # =========================================================================
    # CHAPTER 9.2: KNN
    # =========================================================================
    def _chapter_9_2_knn(self) -> Chapter:
        """Chapter 9.2: K-Nearest Neighbors algorithm."""
        
        return Chapter(
            number="9.2",
            title="K-Nearest Neighbors (KNN)",
            icon="👥",
            sections=[
                Markdown("""
**KNN** ist einer der einfachsten und intuitivsten ML-Algorithmen.

**Prinzip:** "Sage mir, wer deine Nachbarn sind, und ich sage dir, wer du bist."
"""),
                
                Expander("📐 Der Algorithmus", [
                    Markdown("""
**Für einen neuen Datenpunkt x:**
1. Berechne Distanz zu allen Trainingspunkten
2. Finde die k nächsten Nachbarn
3. **Klassifizierung:** Mehrheitsvoting der k Nachbar-Labels
4. **Regression:** Durchschnitt der k Nachbar-Werte
"""),
                    Plot("knn_visualization", "KNN: Klassifizierung mit k=5", height=400),
                ], expanded=True),
                
                Expander("📏 Distanzmetriken", [
                    Markdown("**Euklidische Distanz (Standard):**"),
                    Formula(r"d(x, y) = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}"),
                    Markdown("**Manhattan-Distanz:**"),
                    Formula(r"d(x, y) = \sum_{i=1}^{n}|x_i - y_i|"),
                    Table(
                        headers=["Metrik", "Wann verwenden"],
                        rows=[
                            ["Euklidisch", "Standard für metrische Features"],
                            ["Manhattan", "Robuster bei vielen Dimensionen"],
                            ["Cosine", "Für Textvektoren (TF-IDF)"],
                        ]
                    ),
                ]),
                
                Expander("⚙️ Der Hyperparameter k", [
                    Markdown("""
**k** ist der wichtigste Hyperparameter von KNN:
"""),
                    Table(
                        headers=["k-Wert", "Eigenschaft", "Risiko"],
                        rows=[
                            ["k=1", "Sehr flexibel, memoriert Daten", "Overfitting, Rauschen"],
                            ["k=5-10", "Guter Mittelweg", "Meist empfohlen"],
                            ["k=100+", "Sehr glatt, stabil", "Underfitting möglich"],
                        ]
                    ),
                    Plot("knn_decision_boundaries", "Entscheidungsgrenzen für verschiedene k", height=350),
                    InfoBox("**Faustregel:** k = √n (Wurzel der Stichprobengröße)"),
                ]),
                
                Expander("✅ Vor- und Nachteile", [
                    Columns([
                        [
                            Markdown("""
**Vorteile:**
- Extrem einfach zu verstehen
- Keine Trainingsphase nötig
- Funktioniert bei kleinen Datasets
- Ergebnisse interpretierbar
"""),
                        ],
                        [
                            Markdown("""
**Nachteile:**
- Langsam bei großen Datensätzen (Distanzberechnung)
- Sensitiv gegenüber Skalierung
- Leidet unter Curse of Dimensionality
- Alle Features müssen numerisch sein
"""),
                        ],
                    ]),
                ]),
                
                Expander("💻 Implementation in scikit-learn", [
                    CodeBlock("""
from sklearn.neighbors import KNeighborsClassifier

# Modell erstellen
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')

# Trainieren
knn.fit(X_train, y_train)

# Vorhersagen
y_pred = knn.predict(X_test)

# Accuracy
print(f"Accuracy: {knn.score(X_test, y_test):.2%}")
""", language="python"),
                ]),
            ]
        )
    
    # =========================================================================
    # CHAPTER 9.3: DATA SCALING
    # =========================================================================
    def _chapter_9_3_data_scaling(self) -> Chapter:
        """Chapter 9.3: Data scaling/normalization."""
        
        return Chapter(
            number="9.3",
            title="Daten-Skalierung: Warum und Wie",
            icon="📏",
            sections=[
                WarningBox("""
**Das Problem:** Features haben unterschiedliche Skalen!

- Height: 3-10 cm
- Weight: 40-500 g

Bei Distanzberechnung dominiert Weight → unrealistische Ergebnisse!
"""),
                
                Expander("📐 MinMaxScaler (Normalisierung)", [
                    Markdown("Transformiert jeden Feature auf [0, 1]:"),
                    Formula(r"x' = \frac{x - x_{min}}{x_{max} - x_{min}}"),
                    Markdown("""
**Beispiel:**
- Original: height = 7.5, min=3, max=10
- Skaliert: (7.5 - 3) / (10 - 3) = 0.64
"""),
                    Table(
                        headers=["Eigenschaft", "Wert"],
                        rows=[
                            ["Output-Range", "[0, 1]"],
                            ["Robust gegen Ausreißer", "❌ Nein"],
                            ["Interpretation", "Relativ zu Min/Max"],
                        ]
                    ),
                ], expanded=True),
                
                Expander("📊 StandardScaler (Standardisierung)", [
                    Markdown("Transformiert zu Mittelwert=0, Std=1:"),
                    Formula(r"x' = \frac{x - \mu}{\sigma}"),
                    Markdown("(Z-Score Transformation)"),
                    Table(
                        headers=["Eigenschaft", "Wert"],
                        rows=[
                            ["Output Mittelwert", "0"],
                            ["Output Std", "1"],
                            ["Robust gegen Ausreißer", "⚠️ Besser als MinMax"],
                            ["Interpretation", "Anzahl Standardabweichungen"],
                        ]
                    ),
                ]),
                
                Expander("⚠️ Kritische Regel: Fit nur auf Training!", [
                    WarningBox("""
**Data Leakage vermeiden:**

Der Scaler wird NUR auf Trainingsdaten gefittet!
Test-Daten werden mit den Train-Parametern transformiert.
"""),
                    CodeBlock("""
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# NUR auf Training fitten!
X_train_scaled = scaler.fit_transform(X_train)

# Mit gleichen Parametern auf Test anwenden
X_test_scaled = scaler.transform(X_test)  # NICHT fit_transform!
""", language="python"),
                ]),
                
                Expander("🔧 Wann welchen Scaler?", [
                    Table(
                        headers=["Algorithmus", "Empfohlener Scaler", "Grund"],
                        rows=[
                            ["KNN", "MinMax oder Standard", "Distanzbasiert"],
                            ["SVM", "StandardScaler", "Kernel-Berechnungen"],
                            ["Lineare Regression", "Optional", "Nicht distanzbasiert"],
                            ["Decision Trees", "Nicht nötig", "Splittet auf Thresholds"],
                            ["Neural Networks", "MinMax [0,1]", "Aktivierungsfunktionen"],
                        ]
                    ),
                ]),
            ]
        )
    
    # =========================================================================
    # CHAPTER 9.4: CURSE OF DIMENSIONALITY
    # =========================================================================
    def _chapter_9_4_curse_of_dimensionality(self) -> Chapter:
        """Chapter 9.4: Curse of dimensionality."""
        
        return Chapter(
            number="9.4",
            title="Der Fluch der Dimensionalität",
            icon="🌌",
            sections=[
                Markdown("""
Der **Fluch der Dimensionalität** beschreibt Probleme, die auftreten,
wenn Daten in hochdimensionalen Räumen analysiert werden.
"""),
                
                Expander("📐 Das Kernproblem", [
                    Markdown("""
**Mit steigender Dimensionalität:**
- Das Volumen des Raums wächst **exponentiell**
- Datenpunkte werden **spärlich** verteilt
- Alle Punkte werden **gleich weit** voneinander entfernt!
"""),
                    Formula(r"\text{Volumen}_{d\text{-dim}} \propto r^d"),
                    Plot("curse_of_dimensionality", "Datendichte in verschiedenen Dimensionen", height=350),
                ], expanded=True),
                
                Expander("📊 Konsequenzen für Algorithmen", [
                    Table(
                        headers=["Algorithmus", "Problem", "Ab welcher Dimension?"],
                        rows=[
                            ["KNN", "Nearest ≈ Farthest Neighbor", "d > 10-20"],
                            ["Clustering", "Clusters werden diffus", "d > 15"],
                            ["Regression", "Mehr Features als Beobachtungen", "p > n"],
                        ]
                    ),
                    WarningBox("""
**Beispiel:** Bei 100 Features und 1000 Datenpunkten:
- Jeder Datenpunkt hat durchschnittlich nur 10 "Nachbarn"
- Lokale Strukturen gehen verloren
- KNN gibt quasi zufällige Vorhersagen!
"""),
                ]),
                
                Expander("🔧 Lösungsansätze", [
                    Table(
                        headers=["Strategie", "Methode", "Beispiel"],
                        rows=[
                            ["Feature Selection", "Nur relevante Features behalten", "SelectKBest, RFE"],
                            ["Dimensionsreduktion", "Features komprimieren", "PCA, t-SNE"],
                            ["Regularisierung", "Komplexität bestrafen", "L1 (Lasso), L2 (Ridge)"],
                            ["Mehr Daten", "Exponentiell mehr Samples", "Data Augmentation"],
                        ]
                    ),
                    InfoBox("**Faustregel:** Mindestens 5-10 Samples pro Feature für robuste Schätzung."),
                ]),
            ]
        )
    
    # =========================================================================
    # CHAPTER 9.5: FEATURE ENGINEERING
    # =========================================================================
    def _chapter_9_5_feature_engineering(self) -> Chapter:
        """Chapter 9.5: Feature engineering and exploration."""
        
        return Chapter(
            number="9.5",
            title="Feature Engineering & Datenexploration",
            icon="🔧",
            sections=[
                Markdown("""
**Feature Engineering** ist oft der wichtigste Schritt im ML-Workflow.

"Applied machine learning is basically feature engineering." - Andrew Ng
"""),
                
                Expander("🔍 Datenexploration (EDA)", [
                    Markdown("""
**Warum Daten visualisieren?**
1. Fehler entdecken (Missing Values, Ausreißer)
2. Klassenverteilung verstehen
3. Feature-Korrelationen erkennen
4. Lösbarkeit ohne ML prüfen
"""),
                    Table(
                        headers=["Visualisierung", "Zweck", "Tool"],
                        rows=[
                            ["Histogramm", "Verteilung einzelner Features", "plt.hist()"],
                            ["Scatterplot", "2D Zusammenhänge", "plt.scatter()"],
                            ["Pairplot", "Alle Feature-Paare", "sns.pairplot()"],
                            ["Korrelationsmatrix", "Feature-Abhängigkeiten", "sns.heatmap()"],
                        ]
                    ),
                ], expanded=True),
                
                Expander("📊 Feature-Typen", [
                    Table(
                        headers=["Typ", "Beschreibung", "Behandlung"],
                        rows=[
                            ["Kontinuierlich", "Beliebige Zahlen", "Direkt verwendbar, evtl. skalieren"],
                            ["Kategorisch", "Feste Kategorien", "One-Hot-Encoding"],
                            ["Ordinal", "Kategorien mit Ordnung", "Label-Encoding (1,2,3...)"],
                            ["Binär", "Ja/Nein", "0/1"],
                        ]
                    ),
                ]),
                
                Expander("🛠️ Feature Engineering Techniken", [
                    Table(
                        headers=["Technik", "Beschreibung", "Beispiel"],
                        rows=[
                            ["Polynomial Features", "Interaktionen und Potenzen", "x₁², x₁·x₂"],
                            ["Binning", "Kontinuierlich → Kategorien", "Alter → Altersgruppe"],
                            ["Log-Transformation", "Schiefe reduzieren", "log(income)"],
                            ["One-Hot-Encoding", "Kategorisch → Binär", "Farbe → is_red, is_blue..."],
                            ["Feature Extraction", "Aus Rohdaten ableiten", "Datum → Wochentag"],
                        ]
                    ),
                ]),
                
                Expander("💻 One-Hot-Encoding Beispiel", [
                    CodeBlock("""
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Kategorische Spalte
df['color'] = ['red', 'blue', 'green', 'red']

# One-Hot mit pandas
df_encoded = pd.get_dummies(df, columns=['color'])
# Ergebnis: color_red, color_blue, color_green (0/1)

# ODER mit scikit-learn
encoder = OneHotEncoder(sparse=False)
encoded = encoder.fit_transform(df[['color']])
""", language="python"),
                ]),
            ]
        )
    
    # =========================================================================
    # CHAPTER 9.6: MULTI-CLASS CLASSIFICATION
    # =========================================================================
    def _chapter_9_6_multiclass(self) -> Chapter:
        """Chapter 9.6: Multi-class and multi-label classification."""
        
        return Chapter(
            number="9.6",
            title="Multi-Klassen Klassifizierung",
            icon="🏷️",
            sections=[
                Markdown("""
Bisher haben wir binäre Klassifizierung betrachtet (2 Klassen).
Was wenn wir **mehr als 2 Klassen** haben?
"""),
                
                Expander("📊 Klassifizierungstypen", [
                    Table(
                        headers=["Typ", "Anzahl Klassen", "Beispiel"],
                        rows=[
                            ["Binär", "2", "Spam / Kein Spam"],
                            ["Multi-Klassen", "3+, genau eine", "Apfel, Orange, Zitrone"],
                            ["Multi-Label", "Mehrere gleichzeitig", "Film: Action UND Komödie"],
                        ]
                    ),
                ], expanded=True),
                
                Expander("🔢 Multi-Klassen: Softmax", [
                    Markdown("""
Bei Multi-Klassen liefert das Modell **Wahrscheinlichkeiten für jede Klasse**:
"""),
                    Formula(r"P(y=k|x) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}"),
                    Markdown("""
**Softmax-Eigenschaften:**
- Alle Wahrscheinlichkeiten summieren zu 1
- Vorhersage = Klasse mit höchster Wahrscheinlichkeit
"""),
                    Table(
                        headers=["Klasse", "z (logit)", "Softmax P(k)"],
                        rows=[
                            ["Apfel", "2.0", "0.65"],
                            ["Orange", "1.0", "0.24"],
                            ["Zitrone", "0.5", "0.11"],
                        ]
                    ),
                    InfoBox("Vorhersage: **Apfel** (höchste Wahrscheinlichkeit)"),
                ]),
                
                Expander("🏷️ Multi-Label Klassifizierung", [
                    Markdown("""
Bei Multi-Label kann jedes Beispiel **mehrere Klassen gleichzeitig** haben.

**Beispiel: Film-Genres**
- Film A: Action ✓, Komödie ✗, Thriller ✓
- Film B: Action ✗, Komödie ✓, Thriller ✗
"""),
                    Markdown("**Ansatz:** Ein binärer Classifier pro Label (One-vs-Rest)"),
                    CodeBlock("""
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

# Multi-Label Setup
y = [[1, 0, 1], [0, 1, 0], [1, 1, 0]]  # 3 Labels pro Sample

clf = OneVsRestClassifier(LogisticRegression())
clf.fit(X_train, y_train)
""", language="python"),
                ]),
                
                Expander("📈 Metriken für Multi-Klassen", [
                    Markdown("""
Die Metriken (Precision, Recall, F1) müssen **aggregiert** werden:
"""),
                    Table(
                        headers=["Averaging", "Methode", "Wann verwenden"],
                        rows=[
                            ["Macro", "Mittelwert über Klassen (ungewichtet)", "Alle Klassen gleich wichtig"],
                            ["Weighted", "Gewichtet nach Klassenhäufigkeit", "Imbalance berücksichtigen"],
                            ["Micro", "Globale TP/FP/FN", "Gesamtperformance"],
                        ]
                    ),
                    CodeBlock("""
from sklearn.metrics import f1_score

# Verschiedene Averaging-Methoden
f1_macro = f1_score(y_test, y_pred, average='macro')
f1_weighted = f1_score(y_test, y_pred, average='weighted')
""", language="python"),
                ]),
                
                SuccessBox("""
**Zusammenfassung Kapitel 9:**

Sie haben die ML-Grundlagen gemeistert:
1. ✅ ML-Motivation und Geschichte
2. ✅ Supervised vs. Unsupervised Learning
3. ✅ K-Nearest Neighbors (KNN)
4. ✅ Daten-Skalierung (MinMax, Standard)
5. ✅ Fluch der Dimensionalität
6. ✅ Feature Engineering
7. ✅ Multi-Klassen Klassifizierung

**Sie haben jetzt alle Grundlagen für praktisches Machine Learning!** 🎉
"""),
            ]
        )
