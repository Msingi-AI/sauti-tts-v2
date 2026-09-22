# Spelling eval: v2_step_0018000 (30 clips, re-scored 2026-09-22 after ITN fixes)

| category | n | WER | CER | F0-match | item acc | item CER |
|---|---|---|---|---|---|---|
| spelling | 30 | 0.158 | 0.071 | 100% | 78.0% | 0.152 |

Spelling item accuracy by language: sw=82.4%, en=58.3%, cs=91.7%

WER/CER are on the ITN'd hypothesis; item acc = recall of phone numbers, codes, acronyms and quantities after ITN.

## Misses

- `sp_sw_003` expected ['KQ4B7X2'] got ['KQ', 'B7X2'] — Whisper heard: "Nambari yako ya usajili ni KQ ne B7X mbili."
- `sp_sw_005` expected ['KDA', '123B'] got ['KDA', '12'] — Whisper heard: "Gari lenye nambari KDA, moja mbili tatubili lisimamishwa jana."
- `sp_sw_008` expected ['PIN', '2468'] got ['PIN', '8'] — Whisper heard: "Weka PIN ya kombili nsita nane. Kisha bonyeza sawa."
- `sp_en_003` expected ['KQ4B7X2'] got ['K24B7X2'] — Whisper heard: "Your booking reference is K2 4B7X2."
- `sp_en_004` expected ['KQ100', '8'] got ['100', '8'] — Whisper heard: "Flight Kaique 100 departs at 8 in the morning."
- `sp_en_006` expected ['KRA', 'NSSF'] got ['KII'] — Whisper heard: "The KII office is next to the Iniciusif building."
- `sp_en_007` expected ['AB123CD'] got ['IB123CD'] — Whisper heard: "Your order number is IB123CD."
- `sp_cs_003` expected ['KQ4B7X2'] got ['KQ47X2'] — Whisper heard: "Booking reference yako ni KQ 47X2 usipoteze."
