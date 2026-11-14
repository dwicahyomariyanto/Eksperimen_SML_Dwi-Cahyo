# Eksperimen_SML_Dwi-Cahyo

Eksperimen & otomatisasi preprocessing dataset **Credit Scoring (German Credit)**.

```
Eksperimen_SML_Dwi-Cahyo
├── .github/workflows/preprocessing.yml   # GitHub Actions (advance)
├── credit_raw/credit_raw.csv             # dataset mentah
└── preprocessing
    ├── Eksperimen_Dwi-Cahyo.ipynb       # eksperimen manual (template MSML)
    ├── automate_Dwi-Cahyo.py            # otomatisasi preprocessing (skilled)
    └── credit_preprocessing/             # dataset siap latih (train.csv, test.csv)
```

Menjalankan preprocessing secara lokal:

```bash
pip install -r requirements.txt
python preprocessing/automate_Dwi-Cahyo.py
```
