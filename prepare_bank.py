from skibidi_face_detector.dataset.players import idx_to_class, loader
from skibidi_face_detector.app import App
from resources.model import MODEL_FILE, MODEL_PARAMS

app = App()
app.load_model(
    MODEL_FILE,
    **MODEL_PARAMS
)
app.load_dataset(loader, idx_to_class=idx_to_class)
app.bank.save('resources/banks/players.pckl')
app.bank.present()
