# شرح الكود البسيط

المشروع مكتوب بدوال عادية وبيانات من نوع `dict` و`list`. لا توجد تعريفات `class` في كود المشروع أو الاختبارات. المكتبات مثل Gemini وChroma وLangChain وFlask لها كائنات جاهزة نستعملها مباشرة.

## كل ملف بيعمل إيه؟

| الملف | وظيفته |
|---|---|
| `pharma/documents.py` | قراءة PDF وDOCX وTXT، تقسيم النص وحفظ الملف المرفوع |
| `pharma/provider.py` | الاتصال بـGemini، عمل embeddings وتوليد الرد |
| `pharma/retrieval.py` | فتح ChromaDB وحفظ النصوص والبحث فيها |
| `pharma/agent.py` | تنفيذ خطوات السؤال والتحقق من مصادر الإجابة |
| `pharma/schema.py` | تحديد حقول JSON المطلوبة والتحقق منها، باستخدام dictionaries |
| `pharma/domain.py` | تعليمات Gemini والأمثلة وقواعد الأسئلة الطبية |
| `pharma/web.py` | دوال Flask التي تستقبل طلبات الفرونت |
| `pharma/__main__.py` | أوامر التشغيل والفهرسة من التيرمنال |
| `pharma/evaluation.py` | تجربة الأسئلة وكتابة تقرير التقييم |
| `frontend/` | HTML وCSS وJavaScript |
| `knowledge/` | بيانات الأدوية ومصادرها |
| `tests/test_functions.py` | اختبارات بدوال عادية؛ لا تحتاجها لتشغيل الموقع |

## عند رفع ملف

1. الفرونت يرسل الملف إلى `/api/upload`.
2. `load_document()` يختار `PyPDFLoader` أو `Docx2txtLoader` أو `TextLoader`.
3. `split_documents()` يقسم النص: 700 حرف، وتداخل 150 حرف.
4. `embed_text()` يطلب embeddings من Gemini.
5. `add_documents()` يخزن النصوص وembeddings في ChromaDB على الجهاز.

لا يوجد OCR؛ ملف PDF المصوّر فقط لن يمكن استخراج نصه.

## عند كتابة سؤال

`ask_question()` يتحقق من السؤال، ثم يبحث عن النصوص المناسبة باستخدام `search()`. يرسل هذه النصوص إلى Gemini، ويتحقق من أن مصادر الرد واقتباساته موجودة بالفعل. النتيجة dictionary تُرسل إلى الفرونت كـJSON.

## التشغيل

من فولدر المشروع، بعد ضبط `GEMINI_API_KEY` في ملف `.env` أو نفس التيرمنال:

```powershell
.\.venv\Scripts\python.exe -m pharma.web
```

افتح `http://127.0.0.1:8000` وارفع الملف. ولتجهيز بيانات الأدوية الموجودة في `knowledge/`:

```powershell
.\.venv\Scripts\python.exe -m pharma index
```

ملفاتك المرفوعة محفوظة في `uploads/`، وقاعدة البيانات في `chroma_db/`. لا يحتاج تشغيل الموقع إلى ملفات الاختبارات أو `requirements-dev.txt`.

## حذف مستند

اختار ملف مرفوع من Search in، واضغط Delete selected document، ثم وافق على رسالة التأكيد. الدالة delete_document() تحذف الملف وبياناته والنصوص التابعة له من Chroma في كل موديلات embeddings، بدون حذف بيانات الأدوية الأساسية. الحذف لا يحتاج اتصال Gemini.
