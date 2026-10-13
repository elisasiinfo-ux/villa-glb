# קליפ מילים – "משהו חייב להשתנות"

קליפ מילים בעברית ב-1080p: כתב יד גדול מימין ומשמאל, כל מילה נכתבת בדיוק ברגע שהיא מושרת,
ואנימציות כניסה ויציאה מיוחדות למילים נבחרות.

**הקליפ המוכן:** [`video/lyric_video_1080p.mp4`](video/lyric_video_1080p.mp4) (1080p, 30fps, 3:32).
מוכן להעלאה ליוטיוב. אפשר להוריד אותו מכפתור ה-Download בעמוד הקובץ בגיטהאב.

## מה יש כאן

| קובץ | תפקיד |
|---|---|
| `lyrics.txt` | המילים כפי שהן מושרות בפועל, עם תגיות עיצוב (מיקום, גודל, אפקט) |
| `timings.json` | זמן התחלה/סיום לכל מילה (נוצר אוטומטית, אפשר לתקן ידנית) |
| `render.py` | מנוע הרינדור: רקע, תנועת מצלמה, גריין, פולסים לפי הביט, טקסט ואפקטים |
| `align.py` | סנכרון מילה-מילה (forced alignment) עם Whisper עברי של ivrit.ai |
| `separate.py` | הפרדת השירה מהמוזיקה עם Demucs (לסנכרון מדויק יותר) |
| `transcribe.py` | תמלול ראשוני לזיהוי מבנה השיר |
| `plot_timings.py` | גרף בדיקה: עוצמת השירה מול זמני המילים |
| `assets/` | השיר, תמונת הקאבר והפונטים |

## עיצוב

- **כתב יד** (כתב יד CLM) לשורות הרגילות, בצבע שמנת עם צל רך, כמו בקליפים של חנן בן ארי.
- **Rubik Dirt** (פונט מחוספס) למילים חזקות: "אין מצב", "ומתרסקים", "בלאגן", "רב מכות".
- **Frank Ruhl Libre** בזהב לקטע התפילה "צמאה נפשי". הרקע עובר לספיה חמה עם חלקיקי אור.
- אפקטים: `stamp` (חותמת + רעידה), `slam` (נוחת מלמעלה), `crash` (מתנפץ לרסיסים),
  `fall` (נופל וקופץ), `sink` (שוקע ומטשטש), `wind` (נמחק ברוח), `glitch` (בלאגן דיגיטלי),
  `angel`/`devil` (זוהר לבן / אדום מהבהב), `pulse` (פועם לפי הביט), `prayer` (התגלות רכה).

## רינדור מחדש

```bash
pip install pillow numpy librosa
python render.py --audio assets/song.mp3 --out output/lyric_video.mp4
python render.py --audio assets/song.mp3 --artist "שם האמן"          # שם אמן בפינה הימנית העליונה
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
