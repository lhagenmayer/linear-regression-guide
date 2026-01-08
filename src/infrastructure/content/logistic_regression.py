"""
Logistic Regression Educational Content - Framework Agnostic.

This module covers binary classification using logistic regression.
Covers: Sigmoid, Log-odds, Confusion Matrix, Precision/Recall/F1, ROC/AUC.
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


class LogisticRegressionContent(ContentBuilder):
    """
    Builds educational content for logistic regression and classification.
    
    This content extends the regression chapters to handle binary outcomes.
    """
    
    def build(self) -> EducationalContent:
        """Build all chapters of logistic regression content."""
        return EducationalContent(
            title="📊 Binary Classification - Logistic Regression",
            subtitle="From continuous Y to binary outcomes",
            chapters=[
                self._chapter_8_0_the_problem(),
                self._chapter_8_1_sigmoid_function(),
                self._chapter_8_2_logistic_model(),
                self._chapter_8_3_confusion_matrix(),
                self._chapter_8_4_precision_recall(),
                self._chapter_8_5_roc_curve(),
                self._chapter_8_6_class_imbalance(),
            ]
        )
    
    # =========================================================================
    # CHAPTER 8.0: THE PROBLEM
    # =========================================================================
    def _chapter_8_0_the_problem(self) -> Chapter:
        """Chapter 8.0: Why linear regression fails for binary Y."""
        s = self.stats
        
        return Chapter(
            number="8.0",
            title="Das Problem: Binäre Zielgröße",
            icon="❓",
            sections=[
                InfoBox(f"""
**Bisher:** Unsere Zielgröße Y war **kontinuierlich** (z.B. {s.get('y_label', 'Umsatz')}).

**Jetzt:** Was wenn Y nur zwei Werte annehmen kann?
- Spam / Kein Spam
- Krankheit / Keine Krankheit  
- Kauf / Kein Kauf
"""),
                
                Markdown("### 🤔 Naive Idee: Lineare Regression verwenden?"),
                
                Expander("❌ Warum lineare Regression versagt", [
                    Markdown("""
**Versuch:** Codiere Y als 0/1 und wende lineare Regression an.

**Modell:** ŷ = β₀ + β₁x
"""),
                    Plot("linear_on_binary", "Lineare Regression auf binäre Daten", height=350),
                    Markdown("""
**Probleme:**
1. **Output unbeschränkt:** ŷ kann < 0 oder > 1 sein!
2. **Keine Wahrscheinlichkeit:** ŷ = 1.5 ergibt keinen Sinn
3. **Heteroskedastizität:** Varianz der Residuen ist nicht konstant
4. **Keine echte Verteilung:** Y folgt nicht der Normalverteilung
"""),
                    WarningBox("Lineare Regression verletzt alle statistischen Annahmen bei binären Daten!"),
                ], expanded=True),
                
                Expander("✅ Die Lösung: Logistische Regression", [
                    Markdown("""
**Idee:** Modelliere die **Wahrscheinlichkeit** der positiven Klasse.

Statt Y direkt vorherzusagen, berechnen wir:
"""),
                    Formula(r"P(Y=1|X) = ?"),
                    Markdown("""
Diese Wahrscheinlichkeit muss:
- Zwischen 0 und 1 liegen
- Von den Features X abhängen
- Glatt und differenzierbar sein (für Optimierung)

→ Die **Sigmoid-Funktion** erfüllt alle Anforderungen!
"""),
                ]),
            ]
        )
    
    # =========================================================================
    # CHAPTER 8.1: SIGMOID FUNCTION
    # =========================================================================
    def _chapter_8_1_sigmoid_function(self) -> Chapter:
        """Chapter 8.1: The sigmoid function."""
        
        return Chapter(
            number="8.1",
            title="Die Sigmoid-Funktion",
            icon="📈",
            sections=[
                Markdown("Die **Sigmoid-Funktion** (logistische Funktion) transformiert beliebige reelle Zahlen in das Intervall (0, 1):"),
                
                Formula(r"\sigma(z) = \frac{1}{1 + e^{-z}}"),
                
                Expander("📊 Eigenschaften der Sigmoid-Funktion", [
                    Plot("sigmoid_function", "Die Sigmoid-Funktion σ(z)", height=350),
                    Table(
                        headers=["Eigenschaft", "Wert"],
                        rows=[
                            ["Wertebereich", "(0, 1)"],
                            ["σ(-∞)", "0"],
                            ["σ(0)", "0.5"],
                            ["σ(+∞)", "1"],
                            ["Symmetrie", "σ(-z) = 1 - σ(z)"],
                            ["Ableitung", "σ'(z) = σ(z)(1 - σ(z))"],
                        ]
                    ),
                    SuccessBox("Die Sigmoid-Funktion ist überall differenzierbar - perfekt für Gradient Descent!"),
                ], expanded=True),
                
                Expander("🔢 Beispielrechnungen", [
                    Table(
                        headers=["z", "σ(z)", "Interpretation"],
                        rows=[
                            ["-5", "0.007", "Fast sicher Klasse 0"],
                            ["-2", "0.12", "Wahrscheinlich Klasse 0"],
                            ["0", "0.50", "Keine Tendenz (50/50)"],
                            ["+2", "0.88", "Wahrscheinlich Klasse 1"],
                            ["+5", "0.993", "Fast sicher Klasse 1"],
                        ]
                    ),
                ]),
                
                Expander("🎛️ Interaktive Sigmoid mit Parameter", [
                    Markdown("**Beobachte, wie die Steigung β₁ die Funktion beeinflusst:**"),
                    Plot("sigmoid_interactive", "Sigmoid mit variabler Steigung", height=400),
                ]),
            ]
        )
    
    # =========================================================================
    # CHAPTER 8.2: LOGISTIC MODEL
    # =========================================================================
    def _chapter_8_2_logistic_model(self) -> Chapter:
        """Chapter 8.2: The logistic regression model."""
        s = self.stats
        
        return Chapter(
            number="8.2",
            title="Das Logistische Regressionsmodell",
            icon="🔢",
            sections=[
                Markdown("""
**Logistische Regression** kombiniert lineare Regression mit der Sigmoid-Funktion:
"""),
                
                Expander("📐 Das Modell in zwei Schritten", [
                    Markdown("**Schritt 1:** Lineare Kombination (wie bei OLS)"),
                    Formula(r"z = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots = \mathbf{X}\boldsymbol{\beta}"),
                    Markdown("**Schritt 2:** Transformation durch Sigmoid"),
                    Formula(r"\hat{p} = P(Y=1|X) = \sigma(z) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x)}}"),
                    InfoBox("z heisst auch **Log-Odds** oder **Logit**."),
                ], expanded=True),
                
                Expander("📊 Log-Odds und Odds Ratio", [
                    Markdown("""
**Odds** (Chancen):**
"""),
                    Formula(r"\text{Odds} = \frac{P(Y=1)}{P(Y=0)} = \frac{p}{1-p}"),
                    Markdown("**Log-Odds (Logit):**"),
                    Formula(r"\log\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1 x"),
                    Markdown("""
**Interpretation von β₁:**

Eine Einheit Erhöhung in X verändert die **Log-Odds** um β₁.

Oder äquivalent: Die **Odds** werden mit $e^{\\beta_1}$ multipliziert.
"""),
                    Table(
                        headers=["β₁", "e^β₁", "Effekt auf Odds"],
                        rows=[
                            ["0", "1.0", "Kein Effekt"],
                            ["0.5", "1.65", "Odds steigen um 65%"],
                            ["1.0", "2.72", "Odds verdoppeln sich fast"],
                            ["-0.5", "0.61", "Odds sinken um 39%"],
                        ]
                    ),
                ]),
                
                Expander("🎯 Entscheidungsgrenze", [
                    Markdown("""
**Klassifikationsregel:**
- Wenn $\\hat{p} > 0.5$: Sage Klasse 1 vorher
- Wenn $\\hat{p} \\leq 0.5$: Sage Klasse 0 vorher

**Die Entscheidungsgrenze** ist dort, wo $\\hat{p} = 0.5$:
"""),
                    Formula(r"\sigma(z) = 0.5 \Rightarrow z = 0 \Rightarrow \beta_0 + \beta_1 x = 0"),
                    Markdown("Dies definiert eine **Linie** (in 2D) oder **Hyperebene** (in höheren Dimensionen)."),
                    Plot("decision_boundary", "Entscheidungsgrenze der logistischen Regression", height=400),
                ]),
                
                Expander("⚠️ Limitation: Nur lineare Grenzen", [
                    WarningBox("""
**Logistische Regression kann nur linear separierbare Klassen trennen!**

Für komplexe, nicht-lineare Grenzen benötigen wir:
- Decision Trees
- Support Vector Machines (mit Kernel)
- Neural Networks
"""),
                ]),
            ]
        )
    
    # =========================================================================
    # CHAPTER 8.3: CONFUSION MATRIX
    # =========================================================================
    def _chapter_8_3_confusion_matrix(self) -> Chapter:
        """Chapter 8.3: Confusion matrix fundamentals."""
        
        return Chapter(
            number="8.3",
            title="Die Konfusionsmatrix",
            icon="🔲",
            sections=[
                Markdown("""
Die **Konfusionsmatrix** (Confusion Matrix) ist das fundamentale Werkzeug
zur Bewertung von Klassifikationsmodellen.
"""),
                
                Expander("📊 Die 2x2 Matrix", [
                    Markdown("""
|  | Vorhersage: Positiv | Vorhersage: Negativ |
|--|---------------------|---------------------|
| **Wahr: Positiv** | TP (True Positive) | FN (False Negative) |
| **Wahr: Negativ** | FP (False Positive) | TN (True Negative) |
"""),
                    Table(
                        headers=["Abkürzung", "Name", "Bedeutung"],
                        rows=[
                            ["TP", "True Positive", "Korrekt als positiv erkannt"],
                            ["TN", "True Negative", "Korrekt als negativ erkannt"],
                            ["FP", "False Positive", "Fälschlich als positiv (Fehlalarm)"],
                            ["FN", "False Negative", "Fälschlich als negativ (verpasst)"],
                        ]
                    ),
                ], expanded=True),
                
                Expander("🎯 Beispiel: Krebsdiagnose", [
                    Markdown("""
**Szenario:** 1000 Patienten, 50 haben tatsächlich Krebs.

|  | Test Positiv | Test Negativ |
|--|-------------|--------------|
| **Krebs** | 45 (TP) | 5 (FN) |
| **Kein Krebs** | 100 (FP) | 850 (TN) |
"""),
                    Markdown("""
**Interpretation:**
- 45 Krebspatienten wurden korrekt erkannt (TP)
- 5 Krebspatienten wurden verpasst (FN) - **kritisch!**
- 100 Gesunde wurden fälschlich als krank diagnostiziert (FP)
- 850 Gesunde wurden korrekt als gesund erkannt (TN)
"""),
                ]),
                
                Expander("📈 Interaktive Konfusionsmatrix", [
                    Plot("confusion_matrix_interactive", "Interaktive Konfusionsmatrix mit Threshold-Slider", height=450),
                ]),
            ]
        )
    
    # =========================================================================
    # CHAPTER 8.4: PRECISION & RECALL
    # =========================================================================
    def _chapter_8_4_precision_recall(self) -> Chapter:
        """Chapter 8.4: Precision, Recall, and F1-Score."""
        
        return Chapter(
            number="8.4",
            title="Precision, Recall & F1-Score",
            icon="🎯",
            sections=[
                Expander("📊 Accuracy - Warum sie nicht reicht", [
                    Formula(r"\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}"),
                    Markdown("""
**Problem:** Bei unbalancierten Klassen ist Accuracy irreführend!

**Beispiel:** 99% der E-Mails sind kein Spam.
- Ein Modell, das IMMER "Kein Spam" sagt, hat 99% Accuracy!
- Aber es erkennt keinen einzigen Spam (nutzlos).
"""),
                    WarningBox("Accuracy allein ist bei unbalancierten Daten NICHT aussagekräftig!"),
                ], expanded=True),
                
                Expander("🎯 Precision - Wie präzise sind positive Vorhersagen?", [
                    Formula(r"\text{Precision} = \frac{TP}{TP + FP}"),
                    Markdown("""
**Frage:** Von allen als positiv klassifizierten, wie viele sind wirklich positiv?

**Wann wichtig:** Wenn False Positives teuer sind.
- Spam-Filter: FP = wichtige E-Mail gelöscht
- Produktempfehlung: FP = irrelevante Empfehlung (nervt Kunden)
"""),
                ]),
                
                Expander("📥 Recall - Wie vollständig werden Positive erkannt?", [
                    Formula(r"\text{Recall} = \frac{TP}{TP + FN}"),
                    Markdown("""
**Frage:** Von allen tatsächlich Positiven, wie viele wurden erkannt?

**Wann wichtig:** Wenn False Negatives teuer sind.
- Krebsdiagnose: FN = Krebs nicht erkannt (lebensbedrohlich!)
- Asteroid-Erkennung: FN = Asteroid verpasst (katastrophal)
"""),
                ]),
                
                Expander("⚖️ Der Trade-off: Precision vs. Recall", [
                    Markdown("""
**Das Dilemma:** Precision und Recall stehen oft im Konflikt!

- **Threshold hoch (konservativ):** Hohe Precision, niedriger Recall
  - "Nur sichere positive Vorhersagen"
- **Threshold niedrig (aggressiv):** Hohe Recall, niedrige Precision
  - "Lieber zu viel als zu wenig erkennen"
"""),
                    Plot("precision_recall_tradeoff", "Precision-Recall Trade-off", height=350),
                ]),
                
                Expander("🔢 F1-Score - Das harmonische Mittel", [
                    Formula(r"F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}"),
                    Markdown("""
**Warum harmonisch statt arithmetisch?**

Das harmonische Mittel bestraft extreme Unterschiede:
- Precision=0.9, Recall=0.9 → F1=0.9 ✓
- Precision=0.99, Recall=0.01 → F1=0.02 (nicht 0.5!)

**Wann verwenden:** Wenn man einen ausgewogenen Kompromiss will.
"""),
                    Table(
                        headers=["Precision", "Recall", "F1-Score"],
                        rows=[
                            ["0.90", "0.90", "0.90"],
                            ["0.95", "0.70", "0.81"],
                            ["0.99", "0.10", "0.18"],
                            ["0.60", "0.95", "0.73"],
                        ]
                    ),
                ]),
            ]
        )
    
    # =========================================================================
    # CHAPTER 8.5: ROC CURVE
    # =========================================================================
    def _chapter_8_5_roc_curve(self) -> Chapter:
        """Chapter 8.5: ROC curve and AUC."""
        
        return Chapter(
            number="8.5",
            title="ROC-Kurve & AUC",
            icon="📈",
            sections=[
                Markdown("""
Die **ROC-Kurve** (Receiver Operating Characteristic) visualisiert die
Performance eines Klassifikators über alle möglichen Thresholds.
"""),
                
                Expander("📊 Die Achsen der ROC-Kurve", [
                    Markdown("""
- **X-Achse: False Positive Rate (FPR)**
"""),
                    Formula(r"FPR = \frac{FP}{FP + TN} = \text{Anteil Negative, die fälschlich als positiv klassifiziert werden}"),
                    Markdown("- **Y-Achse: True Positive Rate (TPR) = Recall**"),
                    Formula(r"TPR = \frac{TP}{TP + FN} = \text{Anteil Positive, die korrekt erkannt werden}"),
                ], expanded=True),
                
                Expander("🎨 Interpretation der ROC-Kurve", [
                    Plot("roc_curve", "ROC-Kurve mit verschiedenen Modellen", height=400),
                    Markdown("""
**Punkte auf der Kurve:**
- **(0, 0):** Threshold = 1.0, sage niemals "positiv"
- **(1, 1):** Threshold = 0.0, sage immer "positiv"
- **Perfekter Punkt (0, 1):** Keine FP, alle TP erkannt

**Kurven:**
- **Diagonale:** Zufälliger Classifier (AUC = 0.5)
- **Näher an (0,1):** Besseres Modell
"""),
                ]),
                
                Expander("📐 AUC - Area Under the Curve", [
                    Markdown("""
**AUC** (Area Under the ROC Curve) ist ein einziger Wert, der die
Gesamtqualität des Klassifikators zusammenfasst.
"""),
                    Table(
                        headers=["AUC", "Interpretation"],
                        rows=[
                            ["1.0", "Perfekter Klassifikator"],
                            ["0.9 - 1.0", "Exzellent"],
                            ["0.8 - 0.9", "Gut"],
                            ["0.7 - 0.8", "Akzeptabel"],
                            ["0.5 - 0.7", "Schwach"],
                            ["0.5", "Zufällig (nutzlos)"],
                        ]
                    ),
                    InfoBox("""
**Intuition:** AUC = Wahrscheinlichkeit, dass ein zufällig gewähltes
positives Beispiel einen höheren Score bekommt als ein zufällig gewähltes negatives.
"""),
                ]),
                
                Expander("🎛️ Interaktive ROC mit Threshold-Slider", [
                    Plot("roc_interactive", "ROC-Kurve mit Threshold-Animation", height=450),
                ]),
            ]
        )
    
    # =========================================================================
    # CHAPTER 8.6: CLASS IMBALANCE
    # =========================================================================
    def _chapter_8_6_class_imbalance(self) -> Chapter:
        """Chapter 8.6: Dealing with class imbalance."""
        
        return Chapter(
            number="8.6",
            title="Klassenimbalance - Wenn Accuracy lügt",
            icon="⚠️",
            sections=[
                WarningBox("""
**Das versteckte Problem:** In vielen realen Szenarien sind die Klassen stark unbalanciert!

- Kreditbetrug: 0.1% betrügerische Transaktionen
- Krankheitsdiagnose: 1% haben die Krankheit
- Spam: 5-10% aller E-Mails
"""),
                
                Expander("📊 Beispiel: COVID-19 Test", [
                    Markdown("""
**Szenario:** 10'000 Menschen getestet, 10 tatsächlich infiziert.

**Dummy-Modell:** Sage immer "nicht infiziert"

|  | Positiv | Negativ |
|--|---------|---------|
| **Infiziert** | 0 (TP) | 10 (FN) |
| **Gesund** | 0 (FP) | 9'990 (TN) |

**Accuracy = (0 + 9'990) / 10'000 = 99.9%** 🎉

**Aber:** Recall = 0 / 10 = **0%** - Kein einziger Kranker erkannt!
"""),
                    WarningBox("99.9% Accuracy, aber das Modell ist komplett nutzlos!"),
                ], expanded=True),
                
                Expander("🔧 Strategien gegen Klassenimbalance", [
                    Table(
                        headers=["Strategie", "Idee", "Umsetzung"],
                        rows=[
                            ["Stratified Sampling", "Klassenverteilung in jedem Split erhalten", "stratify=y bei train_test_split"],
                            ["Class Weights", "Minoritätsklasse stärker gewichten", "class_weight='balanced'"],
                            ["Oversampling", "Minoritätsklasse künstlich vergrößern", "SMOTE, Random Oversampling"],
                            ["Undersampling", "Majoritätsklasse reduzieren", "Random Undersampling"],
                            ["Andere Metriken", "F1, AUC statt Accuracy", "Immer bei Imbalance!"],
                        ]
                    ),
                ]),
                
                Expander("📈 Precision-Recall Kurve bei Imbalance", [
                    Markdown("""
**Bei starker Imbalance:** Precision-Recall Kurve ist aussagekräftiger als ROC!

**Warum?** Die ROC-Kurve berücksichtigt TN, die bei starker Imbalance
sehr groß und irreführend sein können.
"""),
                    Plot("precision_recall_curve", "Precision-Recall Kurve", height=400),
                ]),
                
                SuccessBox("""
**Zusammenfassung Kapitel 8:**

Sie haben logistische Regression und Klassifikationsmetriken gemeistert:
1. ✅ Warum lineare Regression für binäre Y versagt
2. ✅ Die Sigmoid-Funktion und ihre Eigenschaften
3. ✅ Log-Odds und Odds Ratio Interpretation
4. ✅ Die Konfusionsmatrix (TP, TN, FP, FN)
5. ✅ Precision, Recall, F1-Score
6. ✅ ROC-Kurve und AUC
7. ✅ Umgang mit Klassenimbalance

**Sie sind jetzt bereit für maschinelles Lernen!** 🚀
"""),
            ]
        )
