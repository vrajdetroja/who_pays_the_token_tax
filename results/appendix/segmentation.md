# Segmentation at each dose level

FLORES-200 devtest sentence 29 (FLORES-200 (NLLB Team, 2024), CC-BY-SA 4.0); in English: "The find also grants insight into the evolution of feathers in birds."

Tokens are separated by ·, a leading space is ▁, and a token that splits a character shows its bytes in hex inside ⟨⟩. Each level keeps the first k merges of the language's 16,384-entry tokenizer (k in the table below).

### Basque (eus_Latn; 96 bytes)

```text
r = 1     21 tokens  Aur·k·ik·untzak·,·▁gainera·,·▁l·um·ek·▁hegazti·etan·▁izan·▁duten·▁bilaka·erari·▁buruzko·▁xehetasunak·▁ematen·▁ditu·.
r = 1.5   31 tokens  A·ur·k·ik·un·tzak·,·▁gainera·,·▁l·um·ek·▁h·eg·az·ti·etan·▁izan·▁duten·▁bil·aka·er·ari·▁buruzko·▁x·eh·etasun·ak·▁ematen·▁ditu·.
r = 2.25  46 tokens  A·ur·k·ik·un·tz·ak·,·▁g·ain·era·,·▁l·u·m·ek·▁h·e·g·az·ti·etan·▁izan·▁du·ten·▁b·il·ak·a·er·ari·▁b·ur·uz·ko·▁·x·eh·eta·sun·ak·▁e·ma·ten·▁ditu·.
```

### Russian (rus_Cyrl; 117 bytes)

```text
r = 1     16 tokens  На·ход·ка·▁также·▁позволяет·▁ознакомиться·▁с·▁э·волю·цией·▁п·ерь·ев·▁у·▁птиц·.
r = 1.5   22 tokens  На·ход·ка·▁также·▁позволяет·▁о·зна·ком·иться·▁с·▁э·вол·ю·цией·▁п·ерь·ев·▁у·▁п·ти·ц·.
r = 2.25  35 tokens  Н·а·ход·ка·▁так·же·▁п·оз·в·ол·я·ет·▁о·з·на·к·ом·ить·ся·▁с·▁э·в·ол·ю·ци·ей·▁п·ер·ь·ев·▁у·▁п·ти·ц·.
```

### Arabic (arb_Arab; 91 bytes)

```text
r = 1     13 tokens  يمن·ح·▁الاك·تشاف·▁أيضاً·▁نظرة·▁على·▁تطور·▁الري·ش·▁في·▁الطيور·.
r = 1.5   22 tokens  يم·ن·ح·▁ال·اك·تش·اف·▁أيض·اً·▁ن·ظ·رة·▁على·▁تط·ور·▁الري·ش·▁في·▁الط·ي·ور·.
r = 2.25  28 tokens  يم·ن·ح·▁ال·ا·ك·ت·ش·اف·▁أ·ي·ض·اً·▁ن·ظ·رة·▁على·▁ت·ط·ور·▁ال·ري·ش·▁في·▁الط·ي·ور·.
```

### Chinese (cmn_Hani; 60 bytes)

```text
r = 1     12 tokens  这一·发现·也有·助于·深入·了解·鸟·类·羽毛·的·进化·。
r = 1.5   19 tokens  这一·发现·也·有·助·于·深·入·了解·⟨e9 b8⟩·⟨9f⟩·类·⟨e7 be⟩·⟨bd⟩·毛·的·进·化·。
r = 2.25  29 tokens  这·一·发·现·也·有·⟨e5 8a⟩·⟨a9⟩·于·⟨e6 b7⟩·⟨b1⟩·入·了·⟨e8 a7⟩·⟨a3⟩·⟨e9⟩·⟨b8⟩·⟨9f⟩·⟨e7⟩·⟨b1⟩·⟨bb⟩·⟨e7 be⟩·⟨bd⟩·⟨e6 af⟩·⟨9b⟩·的·进·化·。
```

### Tamil (tam_Taml; 242 bytes)

```text
r = 1     19 tokens  கண்ட·ுபிட·ிப்ப·ானது·▁பறவ·ைகளில்·▁இற·க·ுகளின்·▁பர·ிணாம·▁வளர்ச்சிய·ைப்·▁பற்றிய·▁நுண்ண·றிவ·ை·▁வழங்குகிறது·.
r = 1.5   28 tokens  க·ண்ட·ுபிட·ிப்ப·ானது·▁பற·வ·ைகளில்·▁இற·க·ுகளின்·▁பர·ிண·ாம·▁வளர·்ச்ச·ிய·ைப்·▁பற்றிய·▁ந·ுண·்ண·ற·ிவ·ை·▁வழங்க·ுகிறது·.
r = 2.25  41 tokens  க·ண்ட·ுப·ிட·ிப்ப·ான·து·▁ப·ற·வ·ைகள·ில்·▁இ·ற·க·ுகள·ின்·▁பர·⟨e0 ae bf e0 ae⟩·⟨a3⟩·ாம·▁வ·ள·ர·்ச·்ச·ிய·ைப்·▁ப·ற்ற·ிய·▁ந·ுண·்ண·ற·ிவ·ை·▁வழ·ங்க·ுகிறது·.
```

## The dose on FLORES-200 devtest

Tokens per byte on all 1,012 devtest sentences, relative to the language's full tokenizer, beside the held-out FineWeb-2 r each level was built to (±3%). FLORES is out of domain for every tokenizer (news, travel and children's-encyclopedia sentences), so it shows how far the dose carries beyond the text it was built on.

| language | level | merges kept | r, held-out | r, FLORES devtest |
|---|---|---|---|---|
| Basque | 1 | 16,127 | 1.000 | 1.000 |
| Basque | 1.5 | 1,665 | 1.514 | 1.494 |
| Basque | 2.25 | 225 | 2.280 | 2.237 |
| Russian | 1 | 16,127 | 1.000 | 1.000 |
| Russian | 1.5 | 1,956 | 1.500 | 1.487 |
| Russian | 2.25 | 285 | 2.267 | 2.239 |
| Arabic | 1 | 16,127 | 1.000 | 1.000 |
| Arabic | 1.5 | 1,652 | 1.514 | 1.493 |
| Arabic | 2.25 | 191 | 2.271 | 2.204 |
| Chinese | 1 | 16,127 | 1.000 | 1.000 |
| Chinese | 1.5 | 1,842 | 1.503 | 1.494 |
| Chinese | 2.25 | 362 | 2.263 | 2.239 |
| Tamil | 1 | 16,127 | 1.000 | 1.000 |
| Tamil | 1.5 | 2,071 | 1.499 | 1.470 |
| Tamil | 2.25 | 380 | 2.243 | 2.162 |
