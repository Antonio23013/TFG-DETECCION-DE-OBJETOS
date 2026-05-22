import os
import time
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision.models.detection.ssdlite import SSDLite320_MobileNet_V3_Large_Weights
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssd import SSDHead

# ==============================================================================
# ⚙️ CONFIGURACIÓN DE RUTAS GENÉRICAS
# ==============================================================================
# Cambia estas rutas al directorio local donde tengas tus datos
DATASET_DIR = './data/YOLO'
OUTPUT_DIR = './runs/ssd_mobilenet'
NUM_CLASSES = 4  # 3 Clases (Civil, Militar, Helicóptero) + 1 Fondo
BATCH_SIZE = 32
NUM_EPOCHS = 15

os.makedirs(os.path.join(OUTPUT_DIR, 'checkpoints'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'metrics'), exist_ok=True)

# ==============================================================================
# 📊 CLASE DATASET
# ==============================================================================
class YoloToSSDDataset(Dataset):
    def __init__(self, images_dir, labels_dir, width=320, height=320, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.width = width
        self.height = height
        self.images = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        label_name = img_name.rsplit('.', 1)[0] + '.txt'
        label_path = os.path.join(self.labels_dir, label_name)

        image_orig = Image.open(img_path).convert("RGB")
        w_orig, h_orig = image_orig.size
        image = image_orig.resize((self.width, self.height))

        boxes, labels = [], []
        scale_x, scale_y = self.width / w_orig, self.height / h_orig

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    cls = int(float(parts[0])) + 1
                    x_c, y_c, w, h = map(float, parts[1:])

                    # Conversión YOLO a SSD (Pixels escalados)
                    x_c_pixel, y_c_pixel = x_c * w_orig, y_c * h_orig
                    w_pixel, h_pixel = w * w_orig, h * h_orig

                    x_min = max(0, (x_c_pixel - w_pixel/2) * scale_x)
                    y_min = max(0, (y_c_pixel - h_pixel/2) * scale_y)
                    x_max = min(self.width, (x_c_pixel + w_pixel/2) * scale_x)
                    y_max = min(self.height, (y_c_pixel + h_pixel/2) * scale_y)

                    if x_max > x_min and y_max > y_min:
                        boxes.append([x_min, y_min, x_max, y_max])
                        labels.append(cls)

        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0), dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}

        if self.transform:
            image = self.transform(image)
        return image, target

# ==============================================================================
# 🧠 UTILIDADES DEL MODELO
# ==============================================================================
def get_transform():
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def collate_fn(batch):
    return tuple(zip(*batch))

def build_model(num_classes):
    weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    model = ssdlite320_mobilenet_v3_large(weights=weights)
    in_channels = [672, 480, 512, 256, 256, 128]
    model.head = SSDHead(in_channels, model.anchor_generator.num_anchors_per_location(), num_classes)
    return model

# ==============================================================================
# 🚀 BUCLE PRINCIPAL DE ENTRENAMIENTO
# ==============================================================================
if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"🔥 Iniciando entrenamiento en: {device}")

    # 1. Cargar Datos
    train_dataset = YoloToSSDDataset(
        os.path.join(DATASET_DIR, 'train', 'images'),
        os.path.join(DATASET_DIR, 'train', 'labels'),
        transform=get_transform()
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=2)

    # 2. Inicializar Modelo
    model = build_model(NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.0005)
    
    loss_log_path = os.path.join(OUTPUT_DIR, 'metrics', 'loss_history.txt')
    with open(loss_log_path, "w") as f:
        f.write("Epoch,Avg_Loss,Time_Min\n")

    # 3. Entrenamiento
    for epoch in range(NUM_EPOCHS):
        start = time.time()
        model.train()
        epoch_loss = 0

        for i, (images, targets) in enumerate(train_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            epoch_loss += losses.item()

            if i % 50 == 0:
                print(f"   Epoch {epoch+1}/{NUM_EPOCHS} | Batch {i}/{len(train_loader)} | Loss: {losses.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        time_epoch = (time.time() - start) / 60

        print(f"🏁 FIN ÉPOCA {epoch+1} | Loss Medio: {avg_loss:.4f} | Tiempo: {time_epoch:.1f} min")

        # Guardar Checkpoint y Métricas
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'checkpoints', f'ssd_v3_epoch_{epoch+1}.pth'))
        with open(loss_log_path, "a") as f:
            f.write(f"{epoch+1},{avg_loss:.6f},{time_epoch:.2f}\n")

    print("\n🎉 Entrenamiento completado con éxito.")