"""
ML-Brücke - Edukativer Content (Framework-agnostisch).

Dieses Modul definiert den Inhalt für die Brücke zwischen Statistik und ML
als REINE DATENSTRUKTUR.
Es behandelt: Loss Functions, Gradient Descent, Train/Val/Test, Bias-Variance, Cross-Validation.
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


class MLBridgeContent(ContentBuilder):
    """
    Builds educational content bridging Statistics to Machine Learning.
    
    This content follows the Simple/Multiple Regression chapters and
    introduces ML concepts using familiar statistical foundations.
    """
    
    def build(self) -> EducationalContent:
        """Build all chapters of ML bridge content."""
        return EducationalContent(
            title="📊 From Statistics to Machine Learning",
            subtitle="Bridging OLS regression to predictive modeling",
            chapters=[
                self._chapter_7_0_ml_perspective(),
                self._chapter_7_1_loss_functions(),
                self._chapter_7_2_gradient_descent(),
                self._chapter_7_3_train_val_test(),
                self._chapter_7_4_bias_variance(),
                self._chapter_7_5_cross_validation(),
                self._chapter_7_6_overfitting_demo(),
            ]
        )
    
    # =========================================================================
    # CHAPTER 7.0: THE ML PERSPECTIVE
    # =========================================================================
    def _chapter_7_0_ml_perspective(self) -> Chapter:
        """Chapter 7.0: Introduction to ML perspective."""
        s = self.stats
        
        # Dynamic variable names
        dataset = s.get('dataset_title', s.get('dataset_name', 'Datensatz'))
        y_label = s.get('y_label', 'Zielvariable')
        
        return Chapter(
            number="7.0",
            title="The Machine Learning Perspective",
            icon="🤖",
            sections=[
                InfoBox(f"""
**Der Paradigmenwechsel**: Von Statistik zu Machine Learning

Bisher haben wir den **{dataset}** Datensatz als statistisches Inferenzproblem betrachtet:
- Welchen Einfluss hat X auf **{y_label}**? (Signifikanz)
- Wie viel Varianz von **{y_label}** erklären wir? (R²)

In Machine Learning verschieben wir den Fokus auf **Vorhersage**:
- Wie genau können wir **{y_label}** für neue Daten vorhersagen?
- Wie gut generalisiert unser Modell auf unbekannte Fälle?
"""),
                
                Markdown("### 🎯 Zwei Perspektiven, ein Modell"),
                
                Table(
                    headers=["Aspekt", "Statistik", "Machine Learning"],
                    rows=[
                        ["Ziel", "Verstehen & Erklären", "Vorhersagen & Generalisieren"],
                        ["Fokus", "Koeffizienten-Interpretation", "Prognose-Genauigkeit"],
                        ["Gütemaß", "R², p-Werte, Konfidenzintervalle", "MSE auf Testdaten, Cross-Validation"],
                        ["Datennutzung", "Alle Daten für Schätzung", "Train/Validation/Test Split"],
                        ["Modellwahl", "Theoriegeleitet", "Datengetrieben (Hyperparameter-Tuning)"],
                        ["Annahmen", "Gauss-Markov zentral", "Generalisierung zentral"],
                    ]
                ),
                
                Expander("💡 Die Verbindung: OLS ist ein ML-Algorithmus!", [
                    Markdown("""
**Überraschung**: OLS (Ordinary Least Squares) ist bereits ein Machine Learning Algorithmus!

**OLS minimiert:**
"""),
                    Formula(r"SSE = \sum_{i=1}^{n}(y_i - \hat{y}_i)^2 = \sum_{i=1}^{n}(y_i - b_0 - b_1 x_i)^2"),
                    Markdown("""
In ML-Sprache heisst das: **OLS minimiert die MSE Loss-Funktion**.

Der einzige Unterschied: Statistik hat eine **geschlossene Lösung** (Normalgleichungen),
ML verwendet oft **iterative Optimierung** (Gradient Descent).
"""),
                    SuccessBox("Die statistische Regression, die wir gelernt haben, ist die Grundlage für viele ML-Algorithmen!"),
                ], expanded=True),
                
                Markdown("### 📊 Unser bisheriges Modell in ML-Notation"),
                MetricRow([
                    Metric("R² (Train)", self.fmt(s.get('r_squared', 0)), "Bisher unser Gütemaß"),
                    Metric("MSE (Train)", self.fmt(s.get('mse', 0)), "Loss-Funktion"),
                    Metric("n", str(s.get('n', 0)), "Trainingsdaten"),
                ]),
            ]
        )
    
    # =========================================================================
    # CHAPTER 7.1: LOSS FUNCTIONS
    # =========================================================================
    def _chapter_7_1_loss_functions(self) -> Chapter:
        """Chapter 7.1: Loss functions - connecting SSE to ML."""
        s = self.stats
        
        return Chapter(
            number="7.1",
            title="Loss Functions - Das Optimierungsziel",
            icon="📉",
            sections=[
                Markdown("""
Eine **Loss-Funktion** (Verlustfunktion) quantifiziert, wie weit unsere
Vorhersagen vom wahren Wert entfernt sind. Das Ziel des Trainings:
**Loss minimieren.**
"""),
                
                Expander("📐 Mean Squared Error (MSE)", [
                    Formula(r"MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2"),
                    Markdown("""
**Eigenschaften:**
- Quadriert Fehler → große Fehler werden stark bestraft
- Überall differenzierbar → gut für Optimierung
- **Verbindung zu OLS:** MSE = SSE/n
"""),
                    MetricRow([
                        Metric("SSE", self.fmt(s.get('sse', 0))),
                        Metric("MSE", self.fmt(s.get('mse', 0))),
                        Metric("RMSE", self.fmt(s.get('rmse', 0) if s.get('rmse') else (s.get('mse', 0)**0.5 if s.get('mse') else 0))),
                    ]),
                    InfoBox(f"""
**Interpretation:** Im Durchschnitt weichen unsere Vorhersagen um 
**{self.fmt((s.get('mse', 0)**0.5) if s.get('mse') else 0, 2)} {s.get('y_unit', 'Einheiten')}** vom wahren Wert ab (RMSE).
"""),
                ], expanded=True),
                
                Expander("📐 Mean Absolute Error (MAE)", [
                    Formula(r"MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|"),
                    Markdown("""
**Eigenschaften:**
- Intuitive Interpretation (durchschnittlicher absoluter Fehler)
- Robust gegen Ausreißer (keine Quadrierung)
- Nicht differenzierbar bei 0 (kann Optimierung erschweren)
"""),
                ]),
                
                Expander("🔬 3D Loss-Landschaft", [
                    Markdown("""
Die **Loss-Landschaft** zeigt den MSE als Funktion der Parameter β₀ und β₁.

**Ziel der Optimierung:** Das Minimum finden (tiefster Punkt der Oberfläche).
"""),
                    Plot("loss_surface_3d", "3D MSE Loss-Landschaft", height=500),
                ]),
            ]
        )
    
    # =========================================================================
    # CHAPTER 7.2: GRADIENT DESCENT
    # =========================================================================
    def _chapter_7_2_gradient_descent(self) -> Chapter:
        """Chapter 7.2: Gradient descent optimization."""
        s = self.stats
        
        return Chapter(
            number="7.2",
            title="Gradient Descent - Iterative Optimierung",
            icon="⬇️",
            sections=[
                Markdown("""
**Gradient Descent** ist der grundlegende Optimierungsalgorithmus im Machine Learning.

**Idee:** Bewege dich iterativ in Richtung des steilsten Abstiegs der Loss-Funktion.
"""),
                
                Expander("📐 Der Algorithmus", [
                    Markdown("**Update-Regel:**"),
                    Formula(r"\theta \leftarrow \theta - \alpha \cdot \nabla L(\theta)"),
                    Markdown("""
Wobei:
- θ = Parameter (β₀, β₁)
- α = **Learning Rate** (Schrittgröße)
- ∇L(θ) = **Gradient** der Loss-Funktion
"""),
                    Table(
                        headers=["Schritt", "Aktion"],
                        rows=[
                            ["1. Initialisierung", "Starte mit zufälligen β₀, β₁"],
                            ["2. Forward Pass", "Berechne Vorhersagen: ŷ = β₀ + β₁x"],
                            ["3. Loss berechnen", "MSE = Σ(y - ŷ)² / n"],
                            ["4. Gradient berechnen", "∂MSE/∂β₀, ∂MSE/∂β₁"],
                            ["5. Update", "β ← β - α · Gradient"],
                            ["6. Wiederholen", "Bis Konvergenz (Loss ändert sich nicht mehr)"],
                        ]
                    ),
                ], expanded=True),
                
                Expander("⚙️ Die Learning Rate α", [
                    Markdown("""
Die **Learning Rate** kontrolliert die Schrittgröße:
"""),
                    Table(
                        headers=["Learning Rate", "Effekt"],
                        rows=[
                            ["Zu klein (α = 0.0001)", "Sehr langsame Konvergenz, viele Iterationen"],
                            ["Optimal (α ≈ 0.01)", "Schnelle und stabile Konvergenz"],
                            ["Zu groß (α = 1.0)", "Divergenz! Springt über Minimum hinaus"],
                        ]
                    ),
                    Plot("learning_rate_comparison", "Learning Rate Vergleich", height=350),
                ]),
                
                Expander("🎬 Gradient Descent Animation", [
                    Markdown("**Beobachte, wie Gradient Descent das Minimum findet:**"),
                    Plot("gradient_descent_animation", "Gradient Descent auf MSE-Oberfläche", height=450),
                ]),
                
                Expander("🔄 OLS vs. Gradient Descent", [
                    Markdown("""
**Warum Gradient Descent, wenn OLS eine geschlossene Lösung hat?**
"""),
                    Table(
                        headers=["Aspekt", "OLS (Normalgleichungen)", "Gradient Descent"],
                        rows=[
                            ["Berechnung", "Einmalig (Matrix-Inversion)", "Iterativ (viele Updates)"],
                            ["Komplexität", "O(n·p² + p³)", "O(k·n·p) für k Iterationen"],
                            ["Große Daten", "Speicherprobleme bei p > 10⁵", "Skaliert gut (Batches)"],
                            ["Nicht-lineare Modelle", "Nicht möglich", "Funktioniert für Neural Networks"],
                        ]
                    ),
                    InfoBox("Für lineare Regression ist OLS optimal. Aber Gradient Descent ist die Grundlage für **Deep Learning**!"),
                ]),
            ]
        )
    
    # =========================================================================
    # CHAPTER 7.3: TRAIN/VALIDATION/TEST
    # =========================================================================
    def _chapter_7_3_train_val_test(self) -> Chapter:
        """Chapter 7.3: Train/validation/test split."""
        s = self.stats
        n = s.get('n', 100)
        dataset = s.get('dataset_title', s.get('dataset_name', 'Datensatz'))
        
        return Chapter(
            number="7.3",
            title="Train/Validation/Test Split",
            icon="✂️",
            sections=[
                WarningBox("""
**Das zentrale Problem:** Bisher haben wir R² auf den **gleichen Daten** berechnet,
auf denen wir trainiert haben. Das ist **Cheating**!

Ein Modell, das die Trainingsdaten perfekt anpasst, kann auf neuen Daten völlig versagen.
"""),
                
                Expander("📊 Die Drei-Wege-Aufteilung", [
                    Markdown(f"""
**Unser Datensatz '{dataset}':** n = {n} Beobachtungen

| Split | Anteil | Zweck |
|-------|--------|-------|
| **Training Set** | 70% ({int(n*0.7)}) | Modellparameter lernen (β₀, β₁) |
| **Validation Set** | 15% ({int(n*0.15)}) | Hyperparameter tunen, Overfitting erkennen |
| **Test Set** | 15% ({int(n*0.15)}) | Finale, unabhängige Evaluation |
"""),
                    Plot("data_split_visualization", "Train/Validation/Test Aufteilung", height=300),
                ], expanded=True),
                
                Expander("🎯 Warum 3 Splits?", [
                    Markdown("""
| Phase | Was passiert | Gefahr |
|-------|-------------|--------|
| **Training** | Modell lernt Parameter | Overfitting auf Trainingsdaten |
| **Validation** | Hyperparameter-Tuning | Overfitting auf Validation Set |
| **Test** | Finale Evaluation | Muss unberührt bleiben! |
"""),
                    WarningBox("**Goldene Regel:** Das Test Set darf **nur einmal** am Ende verwendet werden!"),
                    Markdown("""
**Analogie:**
- Training = Schulstoff lernen
- Validation = Übungs-Altklausuren
- Test = Echte Prüfung (nur einmal!)
"""),
                ]),
                
                Expander("📈 Train Score vs. Test Score", [
                    Markdown("""
**Generalization Gap = Train Score - Test Score**

| Szenario | Train Score | Test Score | Interpretation |
|----------|-------------|------------|----------------|
| Ideal | 0.85 | 0.83 | Kleiner Gap → gut generalisiert |
| Overfitting | 0.99 | 0.60 | Großer Gap → zu komplex |
| Underfitting | 0.50 | 0.48 | Beide schlecht → zu einfach |
"""),
                    MetricRow([
                        Metric("R² (Train)", self.fmt(s.get('r_squared', 0)), "Bekannt"),
                        Metric("R² (Test)", "?", "Nach Split berechnen"),
                    ]),
                ]),
                
                Expander("⚠️ Stratifiziertes Splitting", [
                    Markdown("""
**Problem bei unbalancierten Daten:**

Bei zufälligem Split könnte das Test Set eine andere Verteilung haben als die Trainingsdaten.

**Lösung:** Stratifiziertes Splitting erhält die Verteilung in jedem Split.
"""),
                    CodeBlock("""
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    stratify=y,  # Erhält Klassenverteilung
    random_state=42
)
""", language="python"),
                ]),
            ]
        )
    
    # =========================================================================
    # CHAPTER 7.4: BIAS-VARIANCE TRADE-OFF
    # =========================================================================
    def _chapter_7_4_bias_variance(self) -> Chapter:
        """Chapter 7.4: Bias-variance trade-off."""
        
        return Chapter(
            number="7.4",
            title="Bias-Variance Trade-off",
            icon="⚖️",
            sections=[
                Markdown("""
Der **Bias-Variance Trade-off** ist eines der fundamentalsten Konzepte im Machine Learning.
Er erklärt, warum komplexere Modelle nicht immer besser sind.
"""),
                
                Expander("📐 Die Zerlegung des Vorhersagefehlers", [
                    Formula(r"E[(y - \hat{y})^2] = \underbrace{\text{Bias}^2}_{\text{Systematischer Fehler}} + \underbrace{\text{Variance}}_{\text{Instabilität}} + \underbrace{\sigma^2}_{\text{Irreduzibler Fehler}}"),
                    Table(
                        headers=["Komponente", "Bedeutung", "Ursache"],
                        rows=[
                            ["Bias²", "Systematische Abweichung", "Modell zu einfach (Underfitting)"],
                            ["Variance", "Schwankung bei verschiedenen Trainingsdaten", "Modell zu komplex (Overfitting)"],
                            ["σ²", "Rauschen in den Daten", "Nicht reduzierbar (inherent noise)"],
                        ]
                    ),
                ], expanded=True),
                
                Expander("📊 Visualisierung: Modellkomplexität", [
                    Plot("bias_variance_tradeoff", "Bias-Variance Trade-off", height=400),
                    Markdown("""
**Interpretation:**
- **Links:** Hohes Bias, niedrige Variance → Underfitting
- **Mitte:** Optimaler Balancepunkt
- **Rechts:** Niedriges Bias, hohe Variance → Overfitting
"""),
                ]),
                
                Expander("🔗 Verbindung zu Gauss-Markov", [
                    Markdown("""
**In der statistischen Regression haben wir gelernt:**

OLS ist **BLUE** (Best Linear Unbiased Estimator) unter den Gauss-Markov Annahmen.
- "Best" = minimale Variance unter allen linearen Schätzern
- "Unbiased" = Bias = 0

**ML-Perspektive:** Manchmal akzeptieren wir etwas Bias, um die Variance zu reduzieren!
Das nennt man **Regularisierung** (Ridge, Lasso).
"""),
                    InfoBox("Gauss-Markov sagt: OLS ist effizient. Bias-Variance sagt: Manchmal ist kontrollierter Bias besser."),
                ]),
            ]
        )
    
    # =========================================================================
    # CHAPTER 7.5: CROSS-VALIDATION
    # =========================================================================
    def _chapter_7_5_cross_validation(self) -> Chapter:
        """Chapter 7.5: K-Fold cross-validation."""
        s = self.stats
        
        return Chapter(
            number="7.5",
            title="K-Fold Cross-Validation",
            icon="🔄",
            sections=[
                Markdown("""
**Problem:** Bei kleinen Datensätzen haben wir zu wenig Daten für einen aussagekräftigen Test Split.

**Lösung:** K-Fold Cross-Validation recycelt die Daten intelligent.
"""),
                
                Expander("📊 Wie funktioniert K-Fold CV?", [
                    Markdown("""
**Prinzip:** Teile Daten in K gleiche Teile (Folds). Trainiere K Mal, wobei
jedes Mal ein anderer Fold als Validation Set dient.
"""),
                    Plot("kfold_visualization", "K-Fold Cross-Validation (K=5)", height=350),
                    Markdown("""
| Iteration | Training Folds | Validation Fold | Score |
|-----------|----------------|-----------------|-------|
| 1 | Fold 2,3,4,5 | Fold 1 | 0.82 |
| 2 | Fold 1,3,4,5 | Fold 2 | 0.85 |
| 3 | Fold 1,2,4,5 | Fold 3 | 0.79 |
| 4 | Fold 1,2,3,5 | Fold 4 | 0.84 |
| 5 | Fold 1,2,3,4 | Fold 5 | 0.81 |

**Final Score:** Mean ± Std = 0.82 ± 0.022
"""),
                ], expanded=True),
                
                Expander("📈 Interpretation der Ergebnisse", [
                    Markdown("""
**Mean:** Durchschnittliche Performance über alle K Folds.

**Standard Deviation:** Stabilität der Performance.
- Niedrige Std → Konsistentes Modell
- Hohe Std → Modell ist volatil (manchmal gut, manchmal schlecht)
"""),
                    WarningBox("**Wichtig:** Cross-Validation verbessert NICHT die Performance! Sie gibt nur eine bessere *Schätzung* der echten Performance."),
                ]),
                
                Expander("💻 Implementation in scikit-learn", [
                    CodeBlock("""
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

# 5-Fold Cross-Validation
scores = cross_val_score(
    lr, X, y,
    cv=5,
    scoring='r2'
)

print(f"R² Scores: {scores}")
print(f"Mean R²: {scores.mean():.3f} ± {scores.std():.3f}")
""", language="python"),
                ]),
                
                Expander("❓ Wann Cross-Validation verwenden?", [
                    Table(
                        headers=["Szenario", "Empfehlung"],
                        rows=[
                            ["Kleine Datensätze (< 1000)", "✅ K-Fold CV (K=5 oder 10)"],
                            ["Große Datensätze (> 100k)", "❌ Einfacher Train/Test Split reicht"],
                            ["Hyperparameter-Tuning", "✅ Immer mit CV"],
                            ["Modellauswahl", "✅ CV für fairen Vergleich"],
                            ["Final Evaluation", "❌ Unabhängiges Test Set"],
                        ]
                    ),
                ]),
            ]
        )
    
    # =========================================================================
    # CHAPTER 7.6: OVERFITTING DEMO
    # =========================================================================
    def _chapter_7_6_overfitting_demo(self) -> Chapter:
        """Chapter 7.6: Interactive overfitting demonstration."""
        s = self.stats
        
        return Chapter(
            number="7.6",
            title="Overfitting Demo - Polynomgrad",
            icon="🎛️",
            sections=[
                Markdown("""
**Overfitting** passiert, wenn das Modell die Trainingsdaten "auswendig lernt"
statt die zugrundeliegende Struktur zu erfassen.

**Experiment:** Wir fitten Polynome unterschiedlichen Grades an unsere Daten.
"""),
                
                Expander("📊 Interaktive Polynomgrad-Demo", [
                    Markdown("""
**Beobachte:**
- Grad 1 (linear): Möglicherweise Underfitting
- Grad 3-5: Oft gute Balance
- Grad 15+: Definitiv Overfitting!
"""),
                    Plot("polynomial_overfitting_demo", "Polynomgrad und Overfitting", height=450),
                ], expanded=True),
                
                Expander("📈 Train- vs. Validation-Loss", [
                    Plot("train_val_curves", "Learning Curves: Train vs Validation Loss", height=400),
                    Markdown("""
**Interpretation:**
- **Gutes Modell:** Train- und Val-Loss fallen zusammen und stabilisieren sich
- **Overfitting:** Train-Loss fällt weiter, Val-Loss steigt wieder an
- **Underfitting:** Beide Losses sind hoch und stagnieren früh
"""),
                ]),
                
                Expander("🔧 Strategien gegen Overfitting", [
                    Table(
                        headers=["Strategie", "Idee", "Beispiel"],
                        rows=[
                            ["Early Stopping", "Stoppe Training wenn Val-Loss steigt", "Epoch 50 statt 100"],
                            ["Regularisierung", "Bestrafe große Koeffizienten", "Ridge: +λΣβ²"],
                            ["Feature Selection", "Weniger Features = weniger Komplexität", "Nur signifikante Variablen"],
                            ["Mehr Daten", "Mehr Beispiele → bessere Generalisierung", "Data Augmentation"],
                            ["Einfacheres Modell", "Weniger Parameter", "Linear statt Grad-15"],
                        ]
                    ),
                ]),
                
                SuccessBox("""
**Zusammenfassung Kapitel 7:**

Sie haben die Brücke von Statistik zu Machine Learning geschlagen:
1. ✅ Loss-Funktionen verstehen (MSE = SSE/n)
2. ✅ Gradient Descent als Alternative zu OLS
3. ✅ Train/Validation/Test Split für ehrliche Evaluation
4. ✅ Bias-Variance Trade-off
5. ✅ K-Fold Cross-Validation für robuste Schätzungen
6. ✅ Overfitting erkennen und vermeiden

**Nächstes Kapitel:** Logistische Regression für binäre Klassifikation!
"""),
            ]
        )
