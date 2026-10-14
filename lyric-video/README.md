# קליפ מילים – "מכות עם עצמי"

קליפ מילים בעברית ב-1080p: כתב יד גדול מימין ומשמאל, כל מילה נכתבת בדיוק ברגע שהיא מושרת,
ואנימציות כניסה ויציאה רכות למילים נבחרות. הקליפ נפתח בתמונת הקאבר עם שם השיר, שם השיר מתמוסס
מהתמונה לפני שהשירה מתחילה, והקליפ נשאר על אותה תמונה עד הסוף. התמונה רגועה: בלי רעידות, בלי פולסים
ובלי הבהובים.

**הקליפ המוכן:** [`video/lyric_video_1080p.mp4`](video/lyric_video_1080p.mp4) (1080p, 30fps, 3:32).
מוכן להעלאה ליוטיוב. אפשר להוריד אותו מכפתור ה-Download בעמוד הקובץ בגיטהאב.

## מה יש כאן

| קובץ | תפקיד |
|---|---|
| `lyrics.txt` | המילים כפי שהן מושרות בפועל, עם תגיות עיצוב (מיקום, גודל, אפקט) |
| `timings.json` | זמן התחלה/סיום לכל מילה (נוצר אוטומטית, אפשר לתקן ידנית) |
| `render.py` | מנוע הרינדור: רקע רגוע, מעברי קאבר, טקסט ואפקטים |
| `align.py` | סנכרון מילה-מילה (forced alignment) עם Whisper עברי של ivrit.ai |
| `separate.py` | הפרדת השירה מהמוזיקה עם Demucs (לסנכרון מדויק יותר) |
| `transcribe.py` | תמלול ראשוני לזיהוי מבנה השיר |
| `plot_timings.py` | גרף בדיקה: עוצמת השירה מול זמני המילים |
| `make_clean_background.py` | מוחק את שם השיר מהקאבר ויוצר את תמונת הרקע לקליפ (`background.webp`) |
| `assets/` | השיר, הקאבר עם שם השיר (`cover_title.webp`), הרקע בלי הכיתוב (`background.webp`), התמונה המקורית (`cover.webp`) והפונטים |

## עיצוב

- **כתב יד** (כתב יד CLM) לשורות השיר, בצבע שמנת עם צל רך.
- **Frank Ruhl Libre בזהב** ל"צמאה נפשי לאל חי" בסוף, מוגדל להדגשה, על אותו רקע כמו שאר הקליפ.
- **Rubik Dirt** (פונט מחוספס) למילים חזקות: "אין מצב", "ומתרסקים", "בלאגן", "רב מכות".
- הרקע כמעט סטטי: זום איטי מאוד (3% לאורך כל השיר), וינייט עדין וגריין דק.
- אפקטים רכים (בלי קפיצות, בלי רעידת מסך ובלי הבהוב):
  `stamp`/`crash` (נכנס מעט מוגדל ומתייצב), `slam`/`fall` (גולש למקום מלמעלה),
  `crash` ביציאה (מתפרק לרסיסים שנסחפים לאט), `sink` (שוקע ומטשטש), `wind` (נמחק ברוח),
  `glitch` (פיצול צבעים שמתפוגג בהדרגה), `angel`/`devil` (זוהר לבן / טקסט אדום), `prayer` (התגלות רכה).

## רינדור מחדש

```bash
pip install pillow numpy
python render.py --audio assets/song.mp3 --out output/lyric_video.mp4
python render.py --audio assets/song.mp3 --artist "שם האמן"          # שם אמן בפינה הימנית העליונה
python render.py --audio assets/song.mp3 --title "שם אחר"             # שם השיר בפינה השמאלית העליונה
python render.py --audio assets/song.mp3 --stills 20,66,104 --out output/p.mp4   # תמונות בדיקה
```

`--offset 0.1` מקדים את כל המילים ב-0.1 שניות (ערך שלילי מאחר).

## שינויים נפוצים

- **להזיז מילה בזמן**: לערוך את `start` שלה ב-`timings.json`.
- **לשנות צד, גודל או אפקט**: לערוך את התגיות בסוף השורה ב-`lyrics.txt`.
  שינוי תגיות לא דורש סנכרון מחדש. הוספה או הסרה של מילים כן דורשת.

## סנכרון מחדש (רק אם משנים מילים)

```bash
python separate.py song.wav <demucs_models_dir> stems/            # mdx_extra_q
python transcribe.py <ivrit-ai/whisper-large-v3-turbo-ct2> stems/vocals.wav first_pass.json
python align.py <model_dir> stems/vocals.wav first_pass.json lyrics.txt timings.json --vocals stems/vocals.wav
```

## רישיונות פונטים

Ktav Yad CLM: GPL עם חריגת פונטים (Culmus). Rubik Dirt ו-Frank Ruhl Libre: SIL Open Font License.
