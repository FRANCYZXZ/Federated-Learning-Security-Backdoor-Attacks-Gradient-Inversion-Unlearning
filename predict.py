import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F

# CONFIGURAZIONE
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Inserisci qui il percorso dell'immagine ricostruita che vuoi testare
RECON_IMAGE_PATH = "reconstructed_images/ORIGINAL_peer8.png" 
# Inserisci il percorso del checkpoint del modello addestrato
CHECKPOINT_PATH = './checkpoints/backdoor_CIFAR10_ResNet18_IID_fl_defender_epoch_20.t7'

TARGET_CLASS_IDX = 0 # Plane (La classe che l'attaccante voleva imporre)
SOURCE_CLASS_IDX = 1 # Car (La classe reale dell'immagine)
LABELS_MAP = {0: 'Plane', 1: 'Car', 2: 'Bird', 3: 'Cat', 4: 'Deer', 
              5: 'Dog', 6: 'Frog', 7: 'Horse', 8: 'Ship', 9: 'Truck'}

print(f"--> Analisi dell'immagine: {RECON_IMAGE_PATH}")

# 1. Carica il Modello dal Checkpoint
from models import setup_model
model = setup_model("ResNet18", num_classes=10, tokenizer=None, embedding_dim=100)
model.to(DEVICE)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
if 'state_dict' in checkpoint:
    model.load_state_dict(checkpoint['state_dict'])
else:
    model.load_state_dict(checkpoint)
model.eval()

# 2. Carica e Preprocessa l'Immagine
# Deve subire la STESSA normalizzazione usata durante il training
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

try:
    img_pil = Image.open(RECON_IMAGE_PATH).convert('RGB')
    img_tensor = transform(img_pil).unsqueeze(0).to(DEVICE) # Aggiunge batch dim [1, 3, 32, 32]
except FileNotFoundError:
    print("Errore: Immagine non trovata. Esegui prima la ricostruzione.")
    img_tensor = None

if img_tensor is not None:
    # 3. Predizione
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probs, 1)
        
    pred_idx = predicted_idx.item()
    pred_label = LABELS_MAP.get(pred_idx, str(pred_idx))
    conf_pct = confidence.item() * 100

    # 4. Visualizzazione Risultato
    print("\n" + "="*40)
    print(" RISULTATO INFERENZA")
    print("="*40)
    print(f"Immagine Reale (Source): {LABELS_MAP[SOURCE_CLASS_IDX]}")
    print(f"Obiettivo Attacco (Target): {LABELS_MAP[TARGET_CLASS_IDX]}")
    print("-" * 20)
    print(f"PREDIZIONE MODELLO: {pred_label.upper()}")
    print(f"Confidenza: {conf_pct:.2f}%")
    print("-" * 20)

    if pred_idx == TARGET_CLASS_IDX:
        print("SUCCESSO ATTACCO! Il modello riconosce la Backdoor nell'immagine ricostruita.")
    elif pred_idx == SOURCE_CLASS_IDX:
        print("ATTACCO FALLITO (o Difesa Riuscita). Il modello vede l'oggetto originale, non il trigger.")
    else:
        print("RISULTATO INCERTO. Il modello vede un'altra classe.")
