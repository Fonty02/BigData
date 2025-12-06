class EmissionsEarlyStoppingCallback():
    """
    Early stopping basato su Adaptive Accuracy-Emission Ratio (AN-GES dal report.pdf).
    Ferma il training se AER_current < beta * EMA_AER.
    
    Per classificazione usa ROC-AUC, per regressione usa RSE (come da slide 10).
    """

    def __init__(self, tracker, alpha=0.9, beta=0.2, warmup_epochs=3, is_regression=False):
        self.tracker = tracker
        self.alpha = alpha  # Smoothing per EMA
        self.beta = beta    # Soglia moltiplicativa
        self.warmup_epochs = warmup_epochs
        self.is_regression = is_regression  # Determina quale metrica usare
        self.prev_performance = None
        self.prev_emissions = None
        self.ema_aer = 1e-6  # Inizializzazione piccola
        self.epoch = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        self.epoch += 1
        if self.tracker is None or metrics is None:
            return

        # Ottieni emissioni cumulative attuali
        current_emissions = getattr(self.tracker, '_total_emissions', 
                                    self.tracker._total_energy.kWh * getattr(self.tracker, '_country_iso_code_intensity', 0.5))

        # Usa RSE per regressione, ROC-AUC per classificazione (come da slide 10)
        if self.is_regression:
            current_performance = metrics.get("eval_rse", metrics.get("rse", 0))
        else:
            current_performance = metrics.get("eval_roc_auc", 0)

        if self.prev_performance is not None and self.prev_emissions is not None:
            # Evita divisione per zero: se la performance precedente è (quasi) zero
            if abs(self.prev_performance) > 1e-8:
                delta_perf = (current_performance - self.prev_performance) / abs(self.prev_performance) * 100
            else:
                # Non ha senso una variazione percentuale rispetto a zero: usa la variazione assoluta scalata
                delta_perf = (current_performance - self.prev_performance) * 100

            delta_emiss = (current_emissions - self.prev_emissions) / abs(self.prev_emissions) * 100 if self.prev_emissions > 0 else 0

            if delta_emiss != 0:
                aer_current = delta_perf / delta_emiss
            else:
                aer_current = 0  # Evita divisione per zero

            if self.epoch >= self.warmup_epochs:
                self.ema_aer = self.alpha * aer_current + (1 - self.alpha) * self.ema_aer
                if aer_current < self.beta * self.ema_aer:
                    print(f"Early stopping at epoch {self.epoch}: AER {aer_current:.4f} < {self.beta} * EMA {self.ema_aer:.4f}")
                    control.should_training_stop = True
            else:
                # Accumula per inizializzare EMA
                self.ema_aer = (self.ema_aer + aer_current) / 2

        self.prev_performance = current_performance
        self.prev_emissions = current_emissions

    def check_early_stopping(self, metrics, emissions):
        self.epoch += 1
        if self.tracker is None or metrics is None:
            return False

        current_emissions = emissions
        
        # Usa RSE per regressione, ROC-AUC per classificazione (come da slide 10)
        if self.is_regression:
            current_performance = metrics.get("eval_rse", metrics.get("rse", 0))
        else:
            current_performance = metrics.get("eval_roc_auc", 0)

        if self.prev_performance is not None and self.prev_emissions is not None:
            # Evita divisione per zero: se la performance precedente è (quasi) zero
            if abs(self.prev_performance) > 1e-8:
                delta_perf = (current_performance - self.prev_performance) / abs(self.prev_performance) * 100
            else:
                delta_perf = (current_performance - self.prev_performance) * 100

            delta_emiss = (current_emissions - self.prev_emissions) / abs(self.prev_emissions) * 100 if self.prev_emissions > 0 else 0

            if delta_emiss != 0:
                aer_current = delta_perf / delta_emiss
            else:
                aer_current = 0

            if self.epoch >= self.warmup_epochs:
                self.ema_aer = self.alpha * aer_current + (1 - self.alpha) * self.ema_aer
                if aer_current < self.beta * self.ema_aer:
                    print(f"Early stopping at epoch {self.epoch}: AER {aer_current:.4f} < {self.beta} * EMA {self.ema_aer:.4f}")
                    return True
            else:
                self.ema_aer = (self.ema_aer + aer_current) / 2

        self.prev_performance = current_performance
        self.prev_emissions = current_emissions
        return False