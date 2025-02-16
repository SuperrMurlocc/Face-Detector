import comet_ml
import pytorch_lightning as pl

from skibidi_face_detector.dataset.small_celebrities import train_loader, test_loader
from skibidi_face_detector.dataset.celeba import val_num_classes, val_train_loader, val_test_loader, transformer, augment
from skibidi_face_detector.face_embedder.Model import Model

comet_ml.login(project_name="dlf-superhiper", directory="./")

config = {
    "algorithm": "bayes",
    "name": "Optimize ",
    "spec": {"maxCombo": 40, "objective": "maximize", "metric": "val_accuracy"},
    "parameters": {
        "learning_rate": {"type": "float", "scaling_type": "log_uniform", "min": 0.00001, "max": 0.001},
        "embedding_dim": {"type": "discrete", "values": [128, 256, 512, 1024, 2048]},
        "arc_face_margin": {"type": "float", "scaling_type": "uniform", "min": 0.1, "max": 0.9},
        "triplet_margin": {"type": "float", "scaling_type": "uniform", "min": 0.2, "max": 2.0},
        "scale": {"type": "float", "scaling_type": "uniform", "min": 10.0, "max": 50.0},
        "hidden_layer_features": {"type": "discrete", "values": [1024, 2048, 4096, 8192, 16384, 32768]},
        "p_dropout_1": {"type": "float", "scaling_type": "uniform", "min": 0.0, "max": 0.5},
        "p_dropout_2": {"type": "float", "scaling_type": "uniform", "min": 0.0, "max": 0.5},
        "arc_face_loss_multiplier": {"type": "float", "scaling_type": "uniform", "min": 0.0, "max": 1.0},
        "triplet_loss_multiplier": {"type": "float", "scaling_type": "uniform", "min": 0.0, "max": 1.0},
        "freeze_feature_extractor": {"type": "discrete", "values": [True, False]},
    },
    "trials": 1,
}

opt = comet_ml.Optimizer(config)

if __name__ == "__main__":
    for experiment in opt.get_experiments():
        model = Model(val_num_classes,
                      arc_face_margin=experiment.get_parameter('arc_face_margin'),
                      triplet_margin=experiment.get_parameter('triplet_margin'),
                      scale=experiment.get_parameter('scale'),
                      embedding_dim=experiment.get_parameter('embedding_dim'),
                      learning_rate=experiment.get_parameter('learning_rate'),
                      hidden_layer_features=experiment.get_parameter('hidden_layer_features'),
                      p_dropout_1=experiment.get_parameter('p_dropout_1'),
                      p_dropout_2=experiment.get_parameter('p_dropout_2'),
                      freeze_feature_extractor=experiment.get_parameter('freeze_feature_extractor'),
                      augments=augment,
                      transformer=transformer,
                      arc_face_loss_multiplier=experiment.get_parameter('arc_face_loss_multiplier'),
                      triplet_loss_multiplier=experiment.get_parameter('triplet_loss_multiplier'),
                      accuracy_loaders=(train_loader, test_loader)
                      )

        trainer = pl.Trainer(
            max_epochs=10,
        )

        trainer.fit(model, val_train_loader, val_test_loader)

        experiment.end()
