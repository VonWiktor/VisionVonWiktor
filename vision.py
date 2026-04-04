import os
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from datetime import datetime

def inicjalizuj_system_wizyjny():
    print(">>> Architekt Von Wiktor budzi instancje modelu BLIP...")
    model_id = "Salesforce/blip-image-captioning-base"
    
    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_id)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print(f">>> System aktywny na: {device.type.upper()}")
    return processor, model, device

def analizuj_zdjecie(processor, model, device, sciezka):
    try:
        img = Image.open(sciezka).convert('RGB')
        inputs = processor(img, return_tensors="pt").to(device)
        
        # Generowanie opisu
        out = model.generate(**inputs, max_new_tokens=50)
        opis = processor.decode(out[0], skip_special_tokens=True)
        return opis
    except Exception as e:
        return f"BŁĄD ANALIZY: {e}"

if __name__ == "__main__":
    print("="*45)
    print("   VISION VON WIKTOR v2.0 - MASS ANALYSIS")
    print("="*45)

    try:
        p, m, d = inicjalizuj_system_wizyjny()
        
        # Znajdź wszystkie zdjęcia w folderze
        rozszerzenia = ('.jpg', '.jpeg', '.png', '.bmp')
        pliki = [f for f in os.listdir('.') if f.lower().endswith(rozszerzenia)]
        
        if not pliki:
            print("[!] Brak zdjęć do analizy w bieżącym folderze.")
        else:
            print(f">>> Znaleziono {len(pliki)} plików. Rozpoczynam raport...")
            
            # Tworzenie pliku raportu
            with open("raport_wizyjny.txt", "a", encoding="utf-8") as raport:
                raport.write(f"\n--- SESJA ANALIZY: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
                
                for plik in pliki:
                    wynik = analizuj_zdjecie(p, m, d, plik)
                    linia_raportu = f"PLIK: {plik} | OPIS: {wynik.upper()}"
                    
                    print(f"[+] {linia_raportu}")
                    raport.write(linia_raportu + "\n")
            
            print("-" * 45)
            print(">>> Raport został zapisany w 'raport_wizyjny.txt'")
        
    except Exception as e:
        print(f"\n[!] Krytyczny błąd systemu: {e}")
    
    print("\nStatus: Zadanie wykonane. Kwadracik zaraz będzie zielony.")