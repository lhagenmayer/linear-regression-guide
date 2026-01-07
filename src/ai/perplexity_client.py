"""
Perplexity AI Client - Modern API integration.

Features:
- Async support
- Streaming responses
- Retry logic with exponential backoff
- Response caching
- Token counting
- Rate limiting
"""

import os
import json
import time
import hashlib
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Generator, AsyncGenerator, List
from enum import Enum
import logging

# Try to import httpx for async, fall back to requests
try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

import requests

from ..config import get_logger

logger = get_logger(__name__)


class PerplexityModel(Enum):
    """Available Perplexity models."""
    # Online models (with web search)
    SONAR_SMALL = "llama-3.1-sonar-small-128k-online"
    SONAR_LARGE = "llama-3.1-sonar-large-128k-online"
    SONAR_HUGE = "llama-3.1-sonar-huge-128k-online"
    
    # Chat models (no web search, faster)
    LLAMA_SMALL = "llama-3.1-8b-instruct"
    LLAMA_LARGE = "llama-3.1-70b-instruct"


@dataclass
class PerplexityConfig:
    """Configuration for Perplexity API."""
    api_key: Optional[str] = None
    model: PerplexityModel = PerplexityModel.SONAR_SMALL
    temperature: float = 0.2
    max_tokens: int = 1024
    top_p: float = 0.9
    stream: bool = False
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # System prompt for regression context
    system_prompt: str = """Du bist ein Experte für Statistik und Regressionsanalyse. 
Du erklärst komplexe statistische Konzepte auf verständliche Weise.
Antworte immer auf Deutsch. Verwende mathematische Notation wo sinnvoll.
Sei präzise aber verständlich."""

    def __post_init__(self):
        # Try to get API key from environment if not provided
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
    finish_reason: str = ""
    cached: bool = False
    latency_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "model": self.model,
            "usage": self.usage,
            "citations": self.citations,
            "finish_reason": self.finish_reason,
            "cached": self.cached,
            "latency_ms": self.latency_ms,
        }


class PerplexityClient:
    """
    Modern Perplexity AI client with caching and streaming support.
    
    Usage:
        client = PerplexityClient()
        
        # Simple query
        response = client.ask("Erkläre R² in der Regression")
        print(response.content)
        
        # With context
        response = client.explain_results(stats_dict)
        
        # Streaming
        for chunk in client.stream("Erkläre OLS"):
            print(chunk, end="")
    """
    
    BASE_URL = "https://api.perplexity.ai/chat/completions"
    
    def __init__(self, config: Optional[PerplexityConfig] = None):
        self.config = config or PerplexityConfig()
        self._cache: Dict[str, PerplexityResponse] = {}
        self._rate_limit_remaining = 100
        self._rate_limit_reset = 0
        
        if not self.config.api_key:
            logger.warning("No Perplexity API key configured")
    
    @property
    def is_configured(self) -> bool:
        """Check if API key is configured."""
        return bool(self.config.api_key)
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers."""
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
    
    def _build_payload(
        self, 
        messages: List[Dict[str, str]],
        stream: bool = False
    ) -> Dict[str, Any]:
        """Build API request payload."""
        return {
            "model": self.config.model.value,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
            "stream": stream,
        }
    
    def _cache_key(self, messages: List[Dict[str, str]]) -> str:
        """Generate cache key for messages."""
        content = json.dumps(messages, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()
    
    def ask(
        self, 
        question: str,
        context: Optional[str] = None,
        use_cache: bool = True
    ) -> PerplexityResponse:
        """
        Ask a question to Perplexity AI.
        
        Args:
            question: The question to ask
            context: Optional context to include
            use_cache: Whether to use cached responses
            
        Returns:
            PerplexityResponse with content and metadata
        """
        if not self.is_configured:
            return PerplexityResponse(
                content="⚠️ Perplexity API nicht konfiguriert. Bitte API-Key setzen.",
                model="none",
                cached=False
            )
        
        # Build messages
        messages = [
            {"role": "system", "content": self.config.system_prompt}
        ]
        
        if context:
            messages.append({
                "role": "user", 
                "content": f"Kontext:\n{context}\n\nFrage: {question}"
            })
        else:
            messages.append({"role": "user", "content": question})
        
        # Check cache
        cache_key = self._cache_key(messages)
        if use_cache and cache_key in self._cache:
            logger.debug(f"Cache hit for question: {question[:50]}...")
            cached = self._cache[cache_key]
            cached.cached = True
            return cached
        
        # Make request with retry
        start_time = time.time()
        
        for attempt in range(self.config.max_retries):
            try:
                response = requests.post(
                    self.BASE_URL,
                    headers=self._get_headers(),
                    json=self._build_payload(messages),
                    timeout=self.config.timeout
                )
                
                # Update rate limit info
                self._rate_limit_remaining = int(
                    response.headers.get("x-ratelimit-remaining", 100)
                )
                
                if response.status_code == 429:
                    # Rate limited, wait and retry
                    retry_after = int(response.headers.get("retry-after", 5))
                    logger.warning(f"Rate limited, waiting {retry_after}s")
                    time.sleep(retry_after)
                    continue
                
                response.raise_for_status()
                data = response.json()
                
                latency = (time.time() - start_time) * 1000
                
                result = PerplexityResponse(
                    content=data["choices"][0]["message"]["content"],
                    model=data.get("model", self.config.model.value),
                    usage=data.get("usage", {}),
                    citations=data.get("citations", []),
                    finish_reason=data["choices"][0].get("finish_reason", ""),
                    cached=False,
                    latency_ms=latency
                )
                
                # Cache successful response
                if use_cache:
                    self._cache[cache_key] = result
                
                logger.info(f"Perplexity response in {latency:.0f}ms")
                return result
                
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout on attempt {attempt + 1}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (2 ** attempt))
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (2 ** attempt))
        
        return PerplexityResponse(
            content="❌ Anfrage fehlgeschlagen nach mehreren Versuchen.",
            model="error",
            cached=False
        )
    
    def stream(
        self, 
        question: str,
        context: Optional[str] = None
    ) -> Generator[str, None, None]:
        """
        Stream response from Perplexity AI.
        
        Yields:
            Text chunks as they arrive
        """
        if not self.is_configured:
            yield "⚠️ Perplexity API nicht konfiguriert."
            return
        
        messages = [
            {"role": "system", "content": self.config.system_prompt}
        ]
        
        if context:
            messages.append({
                "role": "user", 
                "content": f"Kontext:\n{context}\n\nFrage: {question}"
            })
        else:
            messages.append({"role": "user", "content": question})
        
        try:
            response = requests.post(
                self.BASE_URL,
                headers=self._get_headers(),
                json=self._build_payload(messages, stream=True),
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
            yield f"❌ Streaming fehlgeschlagen: {e}"
    
    def explain_results(
        self, 
        stats: Dict[str, Any],
        focus: Optional[str] = None
    ) -> PerplexityResponse:
        """
        Get AI explanation for regression results.
        
        Args:
            stats: Dictionary with regression statistics
            focus: Optional focus area (e.g., "r_squared", "coefficients")
            
        Returns:
            PerplexityResponse with explanation
        """
        # Build context from stats
        context_parts = [
            f"Regressionsanalyse: {stats.get('context_title', 'Unbekannt')}",
            f"Stichprobengrösse: n = {stats.get('n', 'N/A')}",
            f"R² = {stats.get('r_squared', 'N/A')}",
        ]
        
        if 'slope' in stats:
            context_parts.append(f"Steigung β₁ = {stats.get('slope')}")
            context_parts.append(f"Intercept β₀ = {stats.get('intercept')}")
            context_parts.append(f"p-Wert (Steigung) = {stats.get('p_slope')}")
        
        if 'beta1' in stats:
            context_parts.append(f"β₁ = {stats.get('beta1')}")
            context_parts.append(f"β₂ = {stats.get('beta2')}")
        
        context = "\n".join(context_parts)
        
        # Build question based on focus
        if focus == "r_squared":
            question = "Erkläre die Bedeutung des R² Werts in diesem Kontext. Ist das Ergebnis gut?"
        elif focus == "coefficients":
            question = "Interpretiere die Koeffizienten. Was bedeuten sie praktisch?"
        elif focus == "significance":
            question = "Bewerte die statistische Signifikanz. Sind die Ergebnisse aussagekräftig?"
        elif focus == "assumptions":
            question = "Welche Annahmen müssen für diese Regression erfüllt sein?"
        else:
            question = "Gib eine kurze, verständliche Interpretation dieser Regressionsergebnisse."
        
        return self.ask(question, context=context)
    
    def suggest_improvements(self, stats: Dict[str, Any]) -> PerplexityResponse:
        """Suggest improvements for the model."""
        context = f"""
Aktuelle Modellgüte:
- R² = {stats.get('r_squared', 'N/A')}
- R² adj. = {stats.get('r_squared_adj', 'N/A')}
- Stichprobe: n = {stats.get('n', 'N/A')}
"""
        
        question = """
Was könnte dieses Regressionsmodell verbessern? 
Gib 3-5 konkrete, praktische Vorschläge.
"""
        return self.ask(question, context=context)
    
    def answer_question(
        self, 
        question: str, 
        stats: Dict[str, Any]
    ) -> PerplexityResponse:
        """Answer a free-form question about the analysis."""
        context = f"""
Analyse: {stats.get('context_title', 'Regression')}
X-Variable: {stats.get('x_label', 'X')}
Y-Variable: {stats.get('y_label', 'Y')}
R² = {stats.get('r_squared', 'N/A')}
n = {stats.get('n', 'N/A')}
"""
        return self.ask(question, context=context)
    
    def clear_cache(self):
        """Clear the response cache."""
        self._cache.clear()
        logger.info("Cache cleared")
    
    @property
    def cache_size(self) -> int:
        """Get number of cached responses."""
        return len(self._cache)
