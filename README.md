# מדד דיג לז'רז'ור מהחוף – ניצנים, אשקלון, אשדוד, זיקים

## התקנה
```
pip install requests pandas numpy statsmodels
```

## הרצה
```
python fishing_index.py                         # כל החופים וכל הדגים, 5 ימים קדימה
python fishing_index.py --beach nitzanim --species gumbar levrak --days 7
python fishing_index.py --demo                  # בדיקה בלי אינטרנט
```
מה יוצא מההרצה:
- רשימת חלונות הזמן הטובים בכל חוף ולכל דג, עם הגורם שהכי מגביל בכל חלון.
- הקובץ `fishing_index.csv`, עם כל השעות וכל תתי-הציונים.
- הקובץ `fishing_index.html`, מפת חום לפי שעה שנוח לפתוח בטלפון.

## כיול מהיומן שלך
1. רושמים כל יציאה בקובץ `logbook_template.csv`, גם יציאות שלא תפסת בהן כלום.
2. אחרי 30–40 יציאות מריצים: `python calibrate.py logbook.csv`
3. מעתיקים את העמודה `fitted_weight` אל `WEIGHTS` שבקובץ `fishing_index.py`.

## מה אפשר לשנות
- `BEACHES`: הקואורדינטות של החופים (הן משוערות) וכיוון החוף.
- `SPECIES`: פרופיל לכל דג: עונה, טמפרטורת מים, אור, גלים ועכירות. אלה הנחות פתיחה שצריך לכייל.
- `WEIGHTS`: המשקל של כל גורם.
- `SAFETY`: מתי שעה מסומנת כלא בטוחה.

## מקור הנתונים
הנתונים מגיעים מ-Open-Meteo. השימוש חינמי לשימוש לא מסחרי, בתנאי שמציינים את המקור.
