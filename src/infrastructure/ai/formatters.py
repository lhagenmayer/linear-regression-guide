"""
R-Output Formatter.

Generiert simulierte R-Style-Output-Strings aus statistischen Dictionaries.
Wird genutzt, um Kontext für die KI-Interpretationsfunktionen bereitzustellen.
"""

from typing import Dict, Any, List
import numpy as np

class ROutputFormatter:
    """Formatiert Statistiken als R-Style-Output."""

    @staticmethod
    def format(stats: Dict[str, Any]) -> str:
        """
        Dispatch formatting based on available keys.
        """
        method = stats.get("method")
        
        if method == "knn":
            return ROutputFormatter.format_knn(stats)
        elif method == "logistic":
            return ROutputFormatter.format_logistic(stats)
        else:
            # Default to linear regression (simple or multiple)
            return ROutputFormatter.format_linear(stats)

    @staticmethod
    def format_linear(stats: Dict[str, Any]) -> str:
        """Formatiert linearen Regressions-Output (lm)."""
        # Handle residuals
        residuals = stats.get('residuals', [0, 0, 0, 0, 0])
        if hasattr(residuals, 'tolist'): residuals = residuals.tolist()
        if not residuals or len(residuals) < 5: residuals = [0, 0, 0, 0, 0]
        
        try:
            res_min = float(np.min(residuals))
            res_q1 = float(np.percentile(residuals, 25))
            res_med = float(np.median(residuals))
            res_q3 = float(np.percentile(residuals, 75))
            res_max = float(np.max(residuals))
        except:
            res_min = res_q1 = res_med = res_q3 = res_max = 0.0
        
        def get_stars(p):
            if p < 0.001: return "***"
            if p < 0.01: return "**"
            if p < 0.05: return "*"
            if p < 0.1: return "."
            return ""
        
        x_label = str(stats.get('x_label', 'X'))[:12]
        y_label = str(stats.get('y_label', 'Y'))
        
        # Check if multiple regression
        # Multiple regression has either: coefficients dict with >1 entries, feature_names list with >1 entries, or b1/beta1
        coeffs = stats.get('coefficients', {})
        feature_names = stats.get('feature_names', [])
        is_multiple = (
            (isinstance(coeffs, dict) and len(coeffs) > 1) or
            (isinstance(feature_names, list) and len(feature_names) > 1) or
            "b1" in stats or "beta1" in stats
        )
        
        if is_multiple:
            return ROutputFormatter._format_multiple_linear(stats, y_label, residuals)

        # Simple Regression
        intercept = float(stats.get('intercept', 0))
        slope = float(stats.get('slope', 0))
        
        # Try to get SE/t/p (default to safe values if missing)
        se_int = float(stats.get('se_intercept', 0))
        se_slope = float(stats.get('se_slope', 0))
        t_int = float(stats.get('t_intercept', 0))
        t_slope = float(stats.get('t_slope', 0))
        p_int = float(stats.get('p_intercept', 1))
        p_slope = float(stats.get('p_slope', 1))
        
        r2 = float(stats.get('r_squared', 0))
        r2_adj = float(stats.get('r_squared_adj', 0))
        f_stat = float(stats.get('f_statistic', 0))
        df = int(stats.get('df', 0))
        
        import math
        mse = float(stats.get('mse', 0))
        rmse = math.sqrt(mse) if mse > 0 else 0

        return f"""Call:
lm(formula = {y_label} ~ {x_label})

Residuals:
     Min       1Q   Median       3Q      Max 
{res_min:8.4f} {res_q1:8.4f} {res_med:8.4f} {res_q3:8.4f} {res_max:8.4f}

Coefficients:
              Estimate Std. Error t value Pr(>|t|)    
(Intercept)  {intercept:9.4f}   {se_int:9.4f}  {t_int:7.3f}   {p_int:.2e} {get_stars(p_int)}
{x_label:12s} {slope:9.4f}   {se_slope:9.4f}  {t_slope:7.3f}   {p_slope:.2e} {get_stars(p_slope)}
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Residual standard error: {rmse:.4f} on {df} degrees of freedom
Multiple R-squared:  {r2:.4f},    Adjusted R-squared:  {r2_adj:.4f}
F-statistic: {f_stat:.2f} on 1 and {df} DF,  p-value: {p_slope:.2e}"""

    @staticmethod
    def _format_multiple_linear(stats: Dict[str, Any], y_label: str, residuals: List[float]) -> str:
        """Helper for Multiple Regression Output."""
        try:
            res_min = float(np.min(residuals))
            res_q1 = float(np.percentile(residuals, 25))
            res_med = float(np.median(residuals))
            res_q3 = float(np.percentile(residuals, 75))
            res_max = float(np.max(residuals))
        except:
            res_min = res_q1 = res_med = res_q3 = res_max = 0.0

        def get_stars(p):
            if p < 0.001: return "***"
            if p < 0.01: return "**"
            if p < 0.05: return "*"
            if p < 0.1: return "."
            return ""

        # Extract intercept and its statistics
        intercept = float(stats.get('intercept', 0))
        se_intercept = float(stats.get('se_intercept', 0))
        t_intercept = float(stats.get('t_intercept', 0))
        p_intercept = float(stats.get('p_intercept', 1))
        
        # Extract feature names for legacy fallback
        feature_names = stats.get('feature_names', [])
        x1_label = feature_names[0] if len(feature_names) > 0 else stats.get('x1_label', 'X1')
        x2_label = feature_names[1] if len(feature_names) > 1 else stats.get('x2_label', 'X2')

        # Coefficients section - start with intercept
        coef_lines = [f"(Intercept)  {intercept:9.4f}   {se_intercept:9.4f}  {t_intercept:7.3f}   {p_intercept:.2e} {get_stars(p_intercept)}"]
        
        # Determine coefficients (slopes) and their metadata
        # Expecting 'coefficients' as a dict of {name: value} and 'se_coefficients', 't_values', 'p_values' as lists
        coeffs_dict = stats.get('coefficients', {})
        se_list = stats.get('se_coefficients', [])
        t_list = stats.get('t_values', [])
        p_list = stats.get('p_values', [])
        
        # Variable names (for X1, X2...)
        var_names = feature_names if feature_names else [x1_label, x2_label]
        
        if isinstance(coeffs_dict, dict) and len(coeffs_dict) > 0:
            # New structured format
            for i, (name, val) in enumerate(coeffs_dict.items()):
                se = se_list[i] if i < len(se_list) else 0.0
                t = t_list[i] if i < len(t_list) else 0.0
                p = p_list[i] if i < len(p_list) else 1.0
                coef_lines.append(f"{name:12s} {val:9.4f}   {se:9.4f}  {t:7.3f}   {p:.2e} {get_stars(p)}")
        else:
            # Fallback to legacy b1/b2/beta1/beta2 if necessary
            b1 = float(stats.get('b1') or stats.get('beta1') or 0)
            b2 = float(stats.get('b2') or stats.get('beta2') or 0)
            se_b1 = float(stats.get('se_beta1', 0))
            se_b2 = float(stats.get('se_beta2', 0))
            t_b1 = float(stats.get('t_beta1', 0))
            t_b2 = float(stats.get('t_beta2', 0))
            p_b1 = float(stats.get('p_beta1', 1))
            p_b2 = float(stats.get('p_beta2', 1))
            
            coef_lines.append(f"{x1_label:12s} {b1:9.4f}   {se_b1:9.4f}  {t_b1:7.3f}   {p_b1:.2e} {get_stars(p_b1)}")
            coef_lines.append(f"{x2_label:12s} {b2:9.4f}   {se_b2:9.4f}  {t_b2:7.3f}   {p_b2:.2e} {get_stars(p_b2)}")

        coeffs_str = "\n".join(coef_lines)

        r2 = float(stats.get('r_squared', 0))
        r2_adj = float(stats.get('r_squared_adj', 0))
        f_stat = float(stats.get('f_statistic', 0))
        df = int(stats.get('df', 0))
        p_f = float(stats.get('f_p_value') or stats.get('p_f') or stats.get('f_pvalue') or 1.0)
        
        import math
        mse = float(stats.get('mse', 0))
        rmse = math.sqrt(mse) if mse > 0 else 0

        # Number of predictors (k)
        k = len(coeffs_dict) if isinstance(coeffs_dict, dict) and coeffs_dict else 2

        return f"""Call:
lm(formula = {y_label} ~ {' + '.join(var_names)})

Residuals:
     Min       1Q   Median       3Q      Max 
{res_min:8.4f} {res_q1:8.4f} {res_med:8.4f} {res_q3:8.4f} {res_max:8.4f}

Coefficients:
              Estimate Std. Error t value Pr(>|t|)    
{coeffs_str}
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Residual standard error: {rmse:.4f} on {df} degrees of freedom
Multiple R-squared:  {r2:.4f},    Adjusted R-squared:  {r2_adj:.4f}
F-statistic: {f_stat:.2f} on {k} and {df} DF,  p-value: {p_f:.2e}"""

    @staticmethod
    def format_logistic(stats: Dict[str, Any]) -> str:
        """Formatiert logistischen Regressions-Output (glm)."""
        y_label = stats.get('y_label', 'Class')
        feature_names = stats.get('feature_names', ['X'])
        
        intercept = float(stats.get('intercept', 0))
        coeffs = stats.get('coefficients', [])
        if not isinstance(coeffs, list): coeffs = [coeffs]
        
        accuracy = float(stats.get('accuracy', 0))
        
        # Calculate AIC if not provided
        if 'aic' in stats and stats['aic'] is not None:
            aic_val = f"{float(stats['aic']):.2f}"
        else:
            # Try to calculate AIC: AIC = 2k - 2*ln(L)
            # For logistic regression: k = number of parameters (intercept + coefficients)
            # L = likelihood (we approximate using log-likelihood from loss if available)
            try:
                n_params = 1 + len(coeffs)  # intercept + coefficients
                loss_history = stats.get('loss_history', [])
                if loss_history and len(loss_history) > 0:
                    # Approximate log-likelihood from final loss (BCE loss ≈ -log(L)/n)
                    final_loss = loss_history[-1] if isinstance(loss_history, list) else loss_history
                    n_samples = stats.get('n_samples', len(stats.get('predictions', [])))
                    if n_samples > 0:
                        log_likelihood = -final_loss * n_samples
                        aic = 2 * n_params - 2 * log_likelihood
                        aic_val = f"{aic:.2f}"
                    else:
                        aic_val = None
                else:
                    aic_val = None
            except (ValueError, TypeError, KeyError):
                aic_val = None
            
            # Format AIC value or use "N/A" only if calculation is truly impossible
            if aic_val is None:
                aic_val = "N/A"
            
        # Check if we have SE/z/p for logistic
        se_coeffs = stats.get('se_coefficients', [])
        z_values = stats.get('z_values', [])
        p_values = stats.get('p_values', [])
        
        coef_str = ""
        # Intercept
        se_int = f"{stats['se_intercept']:9.4f}" if 'se_intercept' in stats else "   N/A  "
        z_int = f"{stats['z_intercept']:7.3f}" if 'z_intercept' in stats else "   N/A "
        p_int = f"{stats['p_intercept']:.2e}" if 'p_intercept' in stats else "   N/A "
        coef_str += f"(Intercept)  {intercept:9.4f}   {se_int}  {z_int}   {p_int}\n"
        
        for i, name in enumerate(feature_names):
            val = coeffs[i] if i < len(coeffs) else 0.0
            se = f"{se_coeffs[i]:9.4f}" if i < len(se_coeffs) else "   N/A  "
            z = f"{z_values[i]:7.3f}" if i < len(z_values) else "   N/A "
            p = f"{p_values[i]:.2e}" if i < len(p_values) else "   N/A "
            coef_str += f"{name:12s} {float(val):9.4f}   {se}  {z}   {p}\n"

        cm = stats.get('confusion_matrix')
        cm_str = str(np.array(cm)) if cm is not None else "N/A"

        return f"""Call:
glm(formula = {y_label} ~ {' + '.join(feature_names)}, family = binomial)

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
     N/A      N/A      N/A      N/A      N/A  

Coefficients:
              Estimate Std. Error z value Pr(>|z|)
{coef_str}
(Dispersion parameter for binomial family taken to be 1)

    Null deviance: N/A on N/A degrees of freedom
Residual deviance: N/A on N/A degrees of freedom
AIC: {aic_val}

Number of Fisher Scoring iterations: N/A

Confusion Matrix:
{cm_str}

Accuracy: {accuracy:.4f}"""

    @staticmethod
    def format_knn(stats: Dict[str, Any]) -> str:
        """Formatiert KNN-Output (Ähnlich wie caret::confusionMatrix)."""
        k = stats.get('k', 3)
        accuracy = float(stats.get('accuracy', 0))
        cm = stats.get('confusion_matrix')
        
        # R's caret package confusionMatrix output structure
        
        return f"""Confusion Matrix and Statistics

          Reference
Prediction {np.array(cm) if cm is not None else 'NA'}

Overall Statistics
                                          
               Accuracy : {accuracy:.4f}          
                 95% CI : (NA, NA)
    No Information Rate : NA              
    P-Value [Acc > NIR] : NA              
                                          
                  Kappa : NA              
                                          
 Mcnemar's Test P-Value : NA              

Statistics by Class:

                     Class: 0  Class: 1
Sensitivity                NA        NA
Specificity                NA        NA
Pos Pred Value             NA        NA
Neg Pred Value             NA        NA
Prevalence                 NA        NA
Detection Rate             NA        NA
Detection Prevalence       NA        NA
Balanced Accuracy          NA        NA

Model Information:
Method: k-Nearest Neighbors (k={k})"""
