"""Ultralytics trainer that saves checkpoints by completed epoch number."""

from __future__ import annotations

import shutil

from ultralytics.models.yolo.detect import DetectionTrainer


class CompletedEpochDetectionTrainer(DetectionTrainer):


    def save_model(self):
        requested_period = int(self.save_period)
        if requested_period >= 0:
            return super().save_model()

        self.save_period = 0
        try:
            saved = super().save_model()
        finally:
            self.save_period = requested_period

        completed_epoch = int(self.epoch) + 1
        period = abs(requested_period)
        if saved and completed_epoch % period == 0:
            target = self.wdir / f"epoch{completed_epoch:03d}.pt"
            shutil.copy2(self.last, target)
        return saved
