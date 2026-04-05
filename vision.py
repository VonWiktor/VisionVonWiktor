import os
import warnings
import time
import csv
from datetime import datetime

# =====================================================================
# 1. TOTALNE WYCISZENIE I TRYB OFFLINE
# =====================================================================
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_OFFLINE"] = "1" # Blokuje łączenie z netem (usuwa warning)

import transformers
from transformers.utils import logging as hf_logging
transformers.logging.set_verbosity_error()
hf_logging.disable_progress_bar()  # Całkowicie wyłącza pasek ładowania (Loading weights...)
# =====================================================================

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

def main():
    print("    VISION VON WIKTOR v3.2 - LITE & CHATTY")
    print("================================================================")
    print()
    
    image_files = ["test.jpg"] 
    
    print(f">>> Znaleziono {len(image_files)} obrazów. Generuję dłuższe opisy...")
    print(">>> [SYSTEM] Uruchamianie rdzenia AI...")
    
    start_load = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Ładowanie z lokalnego cache (dzięki HF_HUB_OFFLINE=1)
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    
    load_time = time.time() - start_load
    print(f">>> [SYSTEM] Model wczytany w {load_time:.2f}s. Akceleracja: {device.upper()}")
    
    results = []
    
    for img_path in image_files:
        try:
            start_infer = time.time()
            
            raw_image = Image.open(img_path).convert('RGB')
            inputs = processor(raw_image, return_tensors="pt").to(device)
            
            # 2. WYMUSZANIE DŁUŻSZYCH OPISÓW
            out = model.generate(
                **inputs, 
                max_new_tokens=80,      # Maksymalna długość
                min_new_tokens=25,      # Zmusza go do napisania minimum 25 tokenów
                num_beams=5,            # Przeszukuje więcej opcji dla lepszego sensu zdania
                repetition_penalty=1.2  # Lekko karze za powtarzanie tych samych słów
            )
            
            caption = processor.decode(out[0], skip_special_tokens=True)
            
            infer_time = time.time() - start_infer
            print(f"[+] {img_path} -> {caption.upper()} ({infer_time:.2f}s)")
            
            results.append([img_path, caption.upper()])
            
        except FileNotFoundError:
            print(f"[-] Nie znaleziono pliku: {img_path}")
        except Exception as e:
            print(f"[-] Błąd podczas analizy {img_path}: {e}")
            
    print("----------------------------------------------------------------")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"raport_wizyjny_{timestamp}.csv"
    
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Plik", "Opis_AI"])
        writer.writerows(results)
        
    print(f">>> Analiza gotowa! Plik zapisano jako: {csv_filename}")

if __name__ == "__main__":
    main()