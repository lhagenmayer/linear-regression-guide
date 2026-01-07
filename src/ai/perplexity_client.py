"""
Perplexity AI Client - R Output Interpretation.

Hauptfeature: Gesamtheitliche Interpretation des R-Outputs auf User-Anfrage.
"""

import os
import json
import time
import hashlib
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Generator, List
from enum import Enum

import requests

from ..config import get_logger

logger = get_logger(__name__)


class PerplexityModel(Enum):
    """Available Perplexity models."""
    SONAR_SMALL = "llama-3.1-sonar-small-128k-online"
    SONAR_LARGE = "llama-3.1-sonar-large-128k-online"
    SONAR_HUGE = "llama-3.1-sonar-huge-128k-online"


@dataclass
class PerplexityConfig:
    """Configuration for Perplexity API."""
    api_key: Optional[str] = None
    model: PerplexityModel = PerplexityModel.SONAR_SMALL
    temperature: float = 0.3
    max_tokens: int = 2048
    timeout: int = 60
    
    def __post_init__(self):
        if self.api_key is None:
            self.api_key = os.environ.get("PERPLEXITY_API_KEY")
            
            # Try Streamlit secrets
            if self.api_key is None:
                try:
                    import streamlit as st
                    self.api_key = st.secrets.get("perplexity", {}).get("api_key")
                except:
                    pass


@dataclass
class PerplexityResponse:
    """Response from Perplexity API."""
    content: str
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    citations: List[str] = field(default_factory=list)
    cached: bool = False
    latency_ms: float = 0.0
    error: bool = False


class PerplexityClient:
    """
    Perplexity AI Client für R-Output Interpretation.
    
    Hauptfeature:
        interpret_r_output(stats) - Interpretiert den gesamten R-Output
    
    Usage:
        client = PerplexityClient()
        
        if client.is_configured:
            response = client.interpret_r_output(stats_dict)
            print(response.content)
    """
    
    BASE_URL = "https://api.perplexity.ai/chat/completions"
    
    # System prompt für statistische Interpretation
    SYSTEM_PROMPT = """Du bist ein erfahrener Statistik-Professor, der Studierenden Regressionsanalysen erklärt.

Deine Aufgabe: Interpretiere den R-Output einer Regressionsanalyse GESAMTHEITLICH und VERSTÄNDLICH.

Struktur deiner Antwort:
1. **Zusammenfassung** (2-3 Sätze): Was sagt das Modell aus?
2. **Koeffizienten-Interpretation**: Was bedeuten β₀ und β₁ praktisch?
3. **Modellgüte**: Wie gut erklärt das Modell die Daten? (R², F-Test)
4. **Signifikanz**: Sind die Ergebnisse statistisch bedeutsam? (p-Werte, t-Tests)
5. **Praktische Bedeutung**: Was bedeutet das für die Praxis?
6. **Einschränkungen**: Worauf sollte man achten?

Regeln:
- Antworte IMMER auf Deutsch
- Verwende die konkreten Variablennamen aus dem Output
- Interpretiere ALLE Werte, nicht nur einzelne
- Erkläre für Studierende verständlich, aber fachlich korrekt
- Verwende Emojis sparsam für Struktur (📊, ✅, ⚠️)"""

    def __init__(self, config: Optional[PerplexityConfig] = None):
        self.config = config or PerplexityConfig()
        self._cache: Dict[str, PerplexityResponse] = {}
    
    @property
    def is_configured(self) -> bool:
        """Check if API key is configured."""
        return bool(self.config.api_key)
    
    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
    
    def _cache_key(self, content: str) -> str:
        return hashlib.md5(content.encode()).hexdigest()
    
    def interpret_r_output(
        self, 
        stats: Dict[str, Any],
        use_cache: bool = True
    ) -> PerplexityResponse:
        """
        Interpretiert den gesamten R-Output gesamtheitlich.
        
        Args:
            stats: Dictionary mit allen Regressionsstatistiken
            use_cache: Cache verwenden für wiederholte Anfragen
            
        Returns:
            PerplexityResponse mit vollständiger Interpretation
        """
        if not self.is_configured:
            return PerplexityResponse(
                content=self._get_fallback_interpretation(stats),
                model="fallback",
                cached=False,
                error=True
            )
        
        # R-Output generieren
        r_output = self._generate_r_output(stats)
        
        # Cache prüfen
        cache_key = self._cache_key(r_output)
        if use_cache and cache_key in self._cache:
            cached = self._cache[cache_key]
            cached.cached = True
            return cached
        
        # Prompt erstellen
        user_prompt = f"""Hier ist der vollständige R-Output einer Regressionsanalyse.
Bitte interpretiere ALLE Werte gesamtheitlich und erkläre, was sie bedeuten.

**Kontext der Analyse:**
{stats.get('context_title', 'Regressionsanalyse')}
{stats.get('context_description', '')}

**R-Output:**
```
{r_output}
```

**Zusätzliche Informationen:**
- Unabhängige Variable (X): {stats.get('x_label', 'X')}
- Abhängige Variable (Y): {stats.get('y_label', 'Y')}
- Stichprobengrösse: n = {stats.get('n', 'N/A')}

Bitte gib eine vollständige, gesamtheitliche Interpretation."""

        # API Request
        try:
            start_time = time.time()
            
            response = requests.post(
                self.BASE_URL,
                headers=self._get_headers(),
                json={
                    "model": self.config.model.value,
                    "messages": [
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                },
                timeout=self.config.timeout
            )
            
            latency = (time.time() - start_time) * 1000
            
            if response.status_code == 429:
                return PerplexityResponse(
                    content="⚠️ Rate Limit erreicht. Bitte versuche es in einer Minute erneut.",
                    model="error",
                    error=True
                )
            
            response.raise_for_status()
            data = response.json()
            
            result = PerplexityResponse(
                content=data["choices"][0]["message"]["content"],
                model=data.get("model", self.config.model.value),
                usage=data.get("usage", {}),
                citations=data.get("citations", []),
                cached=False,
                latency_ms=latency
            )
            
            # Cache speichern
            if use_cache:
                self._cache[cache_key] = result
            
            logger.info(f"Perplexity interpretation in {latency:.0f}ms")
            return result
            
        except requests.exceptions.Timeout:
            logger.error("Perplexity API timeout")
            return PerplexityResponse(
                content="⚠️ Zeitüberschreitung bei der API-Anfrage. Bitte erneut versuchen.",
                model="error",
                error=True
            )
        except Exception as e:
            logger.error(f"Perplexity API error: {e}")
            return PerplexityResponse(
                content=f"❌ API-Fehler: {str(e)}\n\n" + self._get_fallback_interpretation(stats),
                model="error",
                error=True
            )
    
    def _generate_r_output(self, stats: Dict[str, Any]) -> str:
        """Generiert den R-Style Output aus den Statistiken."""
        import numpy as np
        
        # Residuen-Statistiken
        residuals = stats.get('residuals', [0, 0, 0, 0, 0])
        if isinstance(residuals, np.ndarray):
            residuals = residuals.tolist()
        if len(residuals) < 5:
            residuals = [0, 0, 0, 0, 0]
        
        res_min = np.min(residuals)
        res_q1 = np.percentile(residuals, 25)
        res_med = np.median(residuals)
        res_q3 = np.percentile(residuals, 75)
        res_max = np.max(residuals)
        
        # Signifikanz-Sterne
        def get_stars(p):
            if p < 0.001: return "***"
            if p < 0.01: return "**"
            if p < 0.05: return "*"
            if p < 0.1: return "."
            return ""
        
        x_label = stats.get('x_label', 'X')[:12]
        y_label = stats.get('y_label', 'Y')
        
        intercept = stats.get('intercept', 0)
        slope = stats.get('slope', 0)
        se_intercept = stats.get('se_intercept', 0)
        se_slope = stats.get('se_slope', 0)
        t_intercept = stats.get('t_intercept', 0)
        t_slope = stats.get('t_slope', 0)
        p_intercept = stats.get('p_intercept', 1)
        p_slope = stats.get('p_slope', 1)
        
        r_squared = stats.get('r_squared', 0)
        r_squared_adj = stats.get('r_squared_adj', 0)
        mse = stats.get('mse', 0)
        df = stats.get('df', 0)
        
        # F-Statistik berechnen
        ssr = stats.get('ssr', 0)
        sse = stats.get('sse', 1)
        f_stat = (ssr / 1) / (sse / df) if df > 0 and sse > 0 else 0
        
        return f"""Call:
lm(formula = {y_label} ~ {x_label})

Residuals:
     Min       1Q   Median       3Q      Max 
{res_min:8.4f} {res_q1:8.4f} {res_med:8.4f} {res_q3:8.4f} {res_max:8.4f}

Coefficients:
              Estimate Std. Error t value Pr(>|t|)    
(Intercept)  {intercept:9.4f}   {se_intercept:9.4f}  {t_intercept:7.3f}   {p_intercept:.2e} {get_stars(p_intercept)}
{x_label:12s} {slope:9.4f}   {se_slope:9.4f}  {t_slope:7.3f}   {p_slope:.2e} {get_stars(p_slope)}
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Residual standard error: {np.sqrt(mse):.4f} on {df} degrees of freedom
Multiple R-squared:  {r_squared:.4f},    Adjusted R-squared:  {r_squared_adj:.4f}
F-statistic: {f_stat:.2f} on 1 and {df} DF,  p-value: {p_slope:.2e}"""

    def _get_fallback_interpretation(self, stats: Dict[str, Any]) -> str:
        """Fallback-Interpretation ohne API."""
        r2 = stats.get('r_squared', 0)
        slope = stats.get('slope', 0)
        p_slope = stats.get('p_slope', 1)
        n = stats.get('n', 0)
        x_label = stats.get('x_label', 'X')
        y_label = stats.get('y_label', 'Y')
        
        # R² Interpretation
        if r2 >= 0.8:
            r2_text = "sehr gut"
        elif r2 >= 0.6:
            r2_text = "gut"
        elif r2 >= 0.4:
            r2_text = "moderat"
        else:
            r2_text = "schwach"
        
        # Signifikanz
        if p_slope < 0.001:
            sig_text = "höchst signifikant (p < 0.001)"
        elif p_slope < 0.01:
            sig_text = "sehr signifikant (p < 0.01)"
        elif p_slope < 0.05:
            sig_text = "signifikant (p < 0.05)"
        else:
            sig_text = "nicht signifikant (p ≥ 0.05)"
        
        direction = "positiven" if slope > 0 else "negativen"
        
        return f"""## 📊 Automatische Interpretation

**Hinweis:** Diese Interpretation wurde ohne AI generiert (API nicht konfiguriert).

### Zusammenfassung
Das Modell zeigt einen {direction} Zusammenhang zwischen {x_label} und {y_label}.

### Modellgüte
- **R² = {r2:.4f}** → Das Modell erklärt **{r2*100:.1f}%** der Varianz ({r2_text})
- Basierend auf **n = {n}** Beobachtungen

### Signifikanz
Der Zusammenhang ist **{sig_text}**.

### Interpretation der Steigung
Pro Einheit Zunahme in {x_label} verändert sich {y_label} um **{slope:.4f}** Einheiten.

---
*Für eine detailliertere AI-Interpretation bitte Perplexity API-Key konfigurieren.*"""

    def stream_interpretation(
        self, 
        stats: Dict[str, Any]
    ) -> Generator[str, None, None]:
        """
        Streamt die Interpretation für Echtzeit-Anzeige.
        
        Yields:
            Text-Chunks während der Generierung
        """
        if not self.is_configured:
            yield self._get_fallback_interpretation(stats)
            return
        
        r_output = self._generate_r_output(stats)
        
        user_prompt = f"""Interpretiere diesen R-Output gesamtheitlich:

**Kontext:** {stats.get('context_title', 'Regressionsanalyse')}

```
{r_output}
```

Variablen: X = {stats.get('x_label', 'X')}, Y = {stats.get('y_label', 'Y')}, n = {stats.get('n', 'N/A')}"""

        try:
            response = requests.post(
                self.BASE_URL,
                headers=self._get_headers(),
                json={
                    "model": self.config.model.value,
                    "messages": [
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                    "stream": True,
                },
                timeout=self.config.timeout,
                stream=True
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            delta = chunk["choices"][0].get("delta", {})
                            if "content" in delta:
                                yield delta["content"]
                        except json.JSONDecodeError:
                            pass
                            
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            yield f"\n\n❌ Streaming fehlgeschlagen: {e}"
    
    def clear_cache(self):
        """Cache leeren."""
        self._cache.clear()
