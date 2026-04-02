import os
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

def inicjalizuj_system_wizyjny():
    print(">>> Architekt Von Wiktor budzi instancje modelu BLIP...")
    model_id = "Salesforce/blip-image-captioning-base"
    
    # Bezpośrednie ładowanie procesora i modelu
    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_id)
    
    # Wykorzystaj potęgę hardware'u (GPU jeśli masz, inaczej CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    return processor, model, device

def analizuj_zdjecie(processor, model, device, sciezka):
    if not os.path.exists(sciezka):
        return f"BŁĄD: Plik '{sciezka}' nie istnieje w folderze!"
    
    print(f">>> Analiza obrazu: {sciezka}...")
    img = Image.open(sciezka).convert('RGB')
    
    # Przygotowanie pikseli dla AI
    inputs = processor(img, return_tensors="pt").to(device)
    
    # Generowanie opisu (sampling dla lepszej jakości)
    out = model.generate(**inputs, max_new_tokens=50)
    return processor.decode(out[0], skip_special_tokens=True)

if __name__ == "__main__":
    print("="*45)
    print("   VISION VON WIKTOR v1.2 - FINAL STAGE")
    print("="*45)

    try:
        # 1. Start silnika
        p, m, d = inicjalizuj_system_wizyjny()
        
        # 2. Plik testowy - upewnij się, że masz go w folderze!
        test_file = "test.jpg" 
        
        wynik = analizuj_zdjecie(p, m, d, test_file)
        
        print("\n" + "-"*45)
        print(f"RAPORT ARCHITEKTA: {wynik.upper()}")
        print("-" * 45)
        
    except Exception as e:
        print(f"\n[!] Krytyczny błąd systemu: {e}")
    
    print("\nStatus: Zadanie wykonane. Sighișoara coraz bliżej.")