import gcsfs
import pandas as pd
import re

GCP_PROJECT = 'Speech-Emotion-1976'
BUCKET_PATH = 'speech-emotion-bucket/Raw/crema-d/AudioWAV'

# Mapping CREMA-D (modifie ici si tu veux ajouter/exclure une émotion)
EMO_MAP = {
    'NEU': 0, 'HAP': 1, 'SAD': 2, 'ANG': 3, 'FEA': 4, 'DIS': 5
}

# Regex pour choper le code émotion dans le nom du fichier
pattern = re.compile(r'_([A-Z]{3})_')

# Connexion GCP
fs = gcsfs.GCSFileSystem(project=GCP_PROJECT)
file_list = fs.ls(BUCKET_PATH)

rows = []
for path in file_list:
    filename = path.split('/')[-1]
    m = pattern.search(filename)
    if not m:
        print(f"Nom de fichier inattendu: {filename}")
        continue
    emo_code = m.group(1)
    if emo_code not in EMO_MAP:
        print(f"Code émotion inconnu: {emo_code} ({filename})")
        continue
    rows.append({'file': filename, 'label': EMO_MAP[emo_code]})

df = pd.DataFrame(rows)
df.to_csv('cremad_labels.csv', index=False)
print(f'CSV généré avec {len(df)} lignes: cremad_labels.csv')
