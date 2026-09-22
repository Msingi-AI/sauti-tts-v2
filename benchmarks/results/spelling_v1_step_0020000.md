# Spelling eval: v1_step_0020000 (30 clips, re-scored 2026-09-22 after ITN fixes)

| category | n | WER | CER | F0-match | item acc | item CER |
|---|---|---|---|---|---|---|
| spelling | 30 | 0.133 | 0.058 | 100% | 68.3% | 0.159 |

Spelling item accuracy by language: sw=64.7%, en=58.3%, cs=83.3%

WER/CER are on the ITN'd hypothesis; item acc = recall of phone numbers, codes, acronyms and quantities after ITN.

## Misses

- `sp_sw_004` expected ['KRA', '0800722222'] got ['KRA', '080072222'] — Whisper heard: "Wasiliana na KRA kupitia nambari 080072222."
- `sp_sw_005` expected ['KDA', '123B'] got ['KDA12'] — Whisper heard: "Gari lenye nambari KDA moja mbili tatubi lili simamishwa jana."
- `sp_sw_007` expected ['NHIF', 'NSSF'] got ['NAHIF', 'NSCACF'] — Whisper heard: "Kikao cha NAHIF na NSCACF kitaanza kesho asubuhi."
- `sp_sw_008` expected ['PIN', '2468'] got ['PIN', '46'] — Whisper heard: "Weka PIN ya kombili nne sita nani kisha bonyeza sawa."
- `sp_sw_009` expected ['KQ100', '4'] got ['KQ100', '9'] — Whisper heard: "Dege ya KQ moja sifuri, sifuri itaondoka saa 9 asubuhi."
- `sp_en_003` expected ['KQ4B7X2'] got ['K24B7X2'] — Whisper heard: "Your booking reference is K2 4B7X2."
- `sp_en_004` expected ['KQ100', '8'] got ['K2100', '8'] — Whisper heard: "Flight K2-100 departs at 8 in the morning."
- `sp_en_006` expected ['KRA', 'NSSF'] got ['KRI', 'NSF'] — Whisper heard: "The KRI office is next to the NSF building."
- `sp_en_007` expected ['AB123CD'] got ['IB123CD'] — Whisper heard: "Your order number is IB123CD."
- `sp_cs_003` expected ['KQ4B7X2'] got ['KQ', 'B7X2'] — Whisper heard: "Booking reference yako ni KQ ne B7X2 usiipoteze."
- `sp_cs_007` expected ['AB123CD'] got ['123'] — Whisper heard: "Oda yako A, B, moja mbili tatu. Sidi imefika ofisini."
