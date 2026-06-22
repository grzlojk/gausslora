import torch
from torchvision import transforms
from transformers import ViTImageProcessor
from datasets import load_dataset

# --- 1. DATASET MANAGER ---
class DatasetManager:
    """
    Handles loading, splitting, and label mapping for different datasets.
    """
    @staticmethod
    def get_dataset(name, subset_size=None, model_id="google/vit-base-patch16-224"):
        print(f"   ⚙️ Loading Image Processor for: {model_id}")
        try:
            processor = ViTImageProcessor.from_pretrained(model_id)
        except Exception:
            # Fallback dla starszych modeli lub innej nazewnictwa
            processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        
        name = name.lower()
        
       # --- A. Load Raw Data ---
        if name == "cifar10":
            ds = load_dataset("cifar10")
            num_labels = 10
            label_key = 'label'
            image_key = 'img'

        elif name == "cifar100":
            ds = load_dataset("cifar100")
            num_labels = 100
            label_key = 'fine_label' # CIFAR100 ma coarse_label i fine_label
            image_key = 'img'
            
        elif name == "stanfordcars":
            # Używamy sprawdzonego mirrora, jeśli oficjalny nie działa
            try:
                ds = load_dataset("tanganke/stanford_cars")
            except:
                ds = load_dataset("alkzar90/stanford-cars-dataset")
            num_labels = 196
            label_key = 'label'
            image_key = 'image'

        elif name == "oxfordpets":
            # Oxford-IIIT Pet
            ds = load_dataset("timm/oxford-iiit-pet") 
            num_labels = 37
            label_key = 'label'
            image_key = 'image'

        elif name == "dtd":
            ds = load_dataset("tanganke/dtd")
            num_labels = 47
            label_key = 'label'
            image_key = 'image'

        elif name == "eurosat":
            # EuroSAT (RGB version)
            ds = load_dataset("timm/eurosat-rgb")
            num_labels = 10
            label_key = 'label'
            image_key = 'image'

        elif name == "fgvc":
            # ZMIANA: Używamy stabilnego mirrora Parquet zamiast uszkodzonego HuggingFaceM4
            # Repozytorium: https://huggingface.co/datasets/Donghyun99/FGVC-Aircraft
            try:
                ds = load_dataset("Donghyun99/FGVC-Aircraft")
            except Exception:
                # Fallback: Czasem dataset jest pod inną nazwą lub wymaga trust_remote_code
                ds = load_dataset("clane9/fgvc-aircraft")

            num_labels = 100
            label_key = 'label' 
            image_key = 'image'
            
            if 'val' in ds and 'validation' not in ds:
                ds['validation'] = ds['val']

        elif name == "resisc45":
            # NWPU-RESISC45
            ds = load_dataset("timm/resisc45")
            num_labels = 45
            label_key = 'label'
            image_key = 'image'

        else:
            raise ValueError(f"Dataset {name} not supported.")

        # --- B. Subset for Speed (Optional) ---
        if subset_size:
            print(f"   ✂️ Subsetting {name} to {subset_size} samples...")
            if 'train' in ds:
                ds['train'] = ds['train'].select(range(min(len(ds['train']), subset_size)))
            
            # Obsługa różnych nazw splitów walidacyjnych
            test_split_name = next((k for k in ['test', 'validation', 'val'] if k in ds), None)
            if test_split_name:
                ds[test_split_name] = ds[test_split_name].select(range(min(len(ds[test_split_name]), subset_size)))

        img_size = processor.size.get("height", 224)
        mean = processor.image_mean
        std = processor.image_std
        
        fast_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        def transform(batch):
            input_images = batch.get(image_key)
            # Szybka transformacja przy użyciu torchvision
            pixel_values = torch.stack([fast_transform(x.convert("RGB")) for x in input_images])
            
            return {
                'pixel_values': pixel_values,
                'labels': torch.tensor(batch[label_key], dtype=torch.long)
            }
        # ==========================================

        train_split = ds['train']
        
        # Szukanie splitu walidacyjnego
        if 'test' in ds:
            val_split = ds['test']
        elif 'validation' in ds:
            val_split = ds['validation']
        else:
            print("   ⚠️ No validation split found. Splitting train set 80/20.")
            splits = ds['train'].train_test_split(test_size=0.2)
            train_split = splits['train']
            val_split = splits['test']

        train_ds = train_split.with_transform(transform)
        val_ds = val_split.with_transform(transform)

        return train_ds, val_ds, num_labels