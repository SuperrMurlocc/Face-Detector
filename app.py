import cv2
import einops
from skibidi_face_detector.app import App
import torch
from resources.model import MODEL_FILE, MODEL_PARAMS

app = App()
app.load_model(
    MODEL_FILE,
    **MODEL_PARAMS
)
app.bank.load('resources/banks/players.pckl')
app.bank.threshold = 0.5

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open camera.")
    exit()

try:
    while True:
        ret, frame = camera.read()

        if not ret:
            print("Error: Failed to capture image.")
            break

        tensor = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(einops.rearrange(tensor, 'h w c -> c h w'))
        tensor = tensor.float() / 255.0
        processed_image = app.assign_image(tensor)

        if processed_image is not None:
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
            processed_image = cv2.normalize(processed_image, None, 0, 255, cv2.NORM_MINMAX)  # Normalize if needed
            processed_image = processed_image.astype('uint8')

        cv2.imshow("Camera Feed", processed_image if processed_image is not None else frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    camera.release()
    cv2.destroyAllWindows()
