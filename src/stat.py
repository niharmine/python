import requests
import time

URL = "https://api.frankfurter.app/latest?from=EUR&to=USD"

latences = []

for i in range(10):
    debut = time.time()
    response = requests.get(URL)
    fin = time.time()
    
    latence_ms = (fin - debut) * 1000
    latences.append(latence_ms)
    
    print(f"Appel {i+1:2} : {latence_ms:6.2f} ms ({response.status_code})")

# Statistiques
moyenne = sum(latences) / len(latences)

print(f"\n📊 Statistiques :")
print(f"   Moyenne : {moyenne:.2f} ms")
print(f"   Min     : {min(latences):.2f} ms")
print(f"   Max     : {max(latences):.2f} ms")