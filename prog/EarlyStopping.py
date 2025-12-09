class EmissionsEarlyStoppingCallback():
    def __init__(self, tracker, alpha=0.9, beta=0.2, warmup_epochs=3, is_regression=False, classic=False, patience=5):
        self.tracker = tracker
        self.alpha = alpha
        self.beta = beta
        self.warmup_epochs = warmup_epochs
        self.is_regression = is_regression
        self.classic = classic
        self.patience = patience
        self.prev_performance = None
        self.prev_emissions = None
        self.ema_aer = 1e-6
        self.epoch = 0
        self.best_performance = float('inf') if is_regression else float('-inf')
        self.patience_counter = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        self.epoch += 1
        if self.tracker is None or metrics is None:
            return

        current_emissions = getattr(self.tracker, '_total_emissions', 
                                    self.tracker._total_energy.kWh * getattr(self.tracker, '_country_iso_code_intensity', 0.5))

        if self.is_regression:
            current_performance = metrics.get("eval_rse", metrics.get("rse", 0))
        else:
            current_performance = metrics.get("eval_roc_auc", 0)

        if self.prev_performance is not None and self.prev_emissions is not None:
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
                    control.should_training_stop = True
            else:
                self.ema_aer = (self.ema_aer + aer_current) / 2

        self.prev_performance = current_performance
        self.prev_emissions = current_emissions

    def check_early_stopping(self, metrics, emissions):
        self.epoch += 1
        if self.tracker is None or metrics is None:
            return False

        current_emissions = emissions
        
        if self.is_regression:
            current_performance = metrics.get("eval_rse", metrics.get("rse", 0))
        else:
            current_performance = metrics.get("eval_roc_auc", 0)

        if self.classic:
            # Classic early stopping based only on metrics
            improved = (current_performance < self.best_performance) if self.is_regression else (current_performance > self.best_performance)
            if improved:
                self.best_performance = current_performance
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            if self.patience_counter >= self.patience:
                return True
        else:
            # Original AER-based stopping
            if self.prev_performance is not None and self.prev_emissions is not None:
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
                        return True
                else:
                    self.ema_aer = (self.ema_aer + aer_current) / 2

        self.prev_performance = current_performance
        self.prev_emissions = current_emissions
        return False