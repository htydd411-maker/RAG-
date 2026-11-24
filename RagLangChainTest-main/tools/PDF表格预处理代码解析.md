# PDF 表格预处理两套方案代码解析

对应脚本：

- `tools/pdfTableSplitTest01.py` —— **视觉路线**：PDF → 整页图 → 目标检测框出表格 → 裁图 → 视觉大模型读图 → 图片描述进向量库
- `tools/pdfTableSplitTest02.py` —— **文本路线**：PDF → 文本层抽取 + camelot 结构化抽表 → 表格转 Markdown → 大模型转自然语言 → 合成预处理语料存 txt

两段的共同目标是一样的：**把 PDF 里"表格"这种结构破坏型内容，先变成自然语言，再交给 RAG 灌库**。区别只在于"读表格"这件事是让模型看图，还是让程序解析文本层。

---

## 一、pdfTableSplitTest01.py：视觉路线

### 1.1 依赖与运行前提

| 位置 | 依赖 | 作用 |
|---|---|---|
| [:9](file:///D:/ai%20lianxi/RAG/project/RagLangChainTest-main/tools/pdfTableSplitTest01.py#L9) `fitz` | PyMuPDF | PDF 整页渲染成位图 |
| [:10](file:///D:/ai%20lianxi/RAG/project/RagLangChainTest-main/tools/pdfTableSplitTest01.py#L10) `PIL` | Pillow | 图像读写、按框裁剪 |
| [:11](file:///D:/ai%20lianxi/RAG/project/RagLangChainTest-main/tools/pdfTableSplitTest01.py#L11) `torchvision` | 变换管线 | `Compose/ToTensor/Normalize` |
| [:12](file:///D:/ai%20lianxi/RAG/project/RagLangChainTest-main/tools/pdfTableSplitTest01.py#L12) `transformers` | TableTransformer | `AutoModelForObjectDetection` |
| [:15](file:///D:/ai%20lianxi/RAG/project/RagLangChainTest-main/tools/pdfTableSplitTest01.py#L15) `openai` | OpenAI 协议客户端 | 指向本机 One API |
| [:16-17](file:///D:/ai%20lianxi/RAG/project/RagLangChainTest-main/tools/pdfTableSplitTest01.py#L16) | chromadb | 内存向量库 |

注意脚本头部注释（第 3-5 行）写的 `pip install timm` 是**过时信息**，本机实际由 `torch + transformers` 撑起模型，`timm` 不是必需。

**必须以 `tools/` 为工作目录运行**：[第245行](file:///D:/ai%20lianxi/RAG/project/RagLangChainTest-main/tools/pdfTableSplitTest01.py#L245) 的 `"../input/健康档案(含表格01).pdf"` 和 [第75行](file:///D:/ai%20lianxi/RAG/project/RagLangChainTest-main/tools/pdfTableSplitTest01.py#L75) 的 `"../other/models/table-transformer-detection"` 都是相对路径。

```
cd D:\ai lianxi\RAG\project\RagLangChainTest-main\tools
D:\ai lianxi\RAG\project\.venv\Scripts\python.exe pdfTableSplitTest01.py
```

### 1.2 四阶段流水线

```
PDF ──(1) pdf2images──> 整页 PNG ──(2) detect_and_crop_save_table──> 表格小图 PNG
                                                                      │
             查询问题 <──(4) search top1 <── 内存 Chroma <──(3) 图片描述+向量
                                  │
                                  └── 命中的原图再喂一次 image_qa ──> 最终答案
```

### 1.3 逐段解析

**(1) PDF 转整页图 —— [`pdf2images`](file:///D:/ai%20lianxi/RAG/project/RagLangChainTest-main/tools/pdfTableSplitTest01.py#L26)**

输出目录直接用 PDF 主文件名（去掉扩展名）当文件夹，所以 `input/健康档案(含表格01).pdf` 生成 `input/健康档案(含表格01)/`。每页 `page.get_pixmap()` 取默认 72dpi 位图，再 `Image.frombytes("RGB", ...)` 转 PIL 对象存 PNG（[第38-42行](file:///D:/ai%20lianxi/RAG/project/RagLangChainTest-main/tools/pdfTableSplitTest01.py#L38)）。

> 这里**没有传 `matrix` 放大倍率**，72dpi 对 A4 大约 595×842 像素，表格里的细线和小字号容易糊。这是整条链路精度的第一道天花板。

**(2) 表格定位 —— TableTransformer**

- [`MaxResize`](file:///D:/ai%20lianxi/RAG/project/RagLangChainTest-main/tools/pdfTableSplitTest01.py#L49)：等比缩到长边 800，这是 DETR 系模型的输入约定。
- [`detection_transform`](file:///D:/ai%20lianxi/RAG/project/RagLangChainTest-main/tools/pdfTableSplitTest01.py#L64)：缩放 → 转张量 → 用 ImageNet 均值方差归一化。
- [第74-76行](file:///D:/ai%20lianxi/RAG/project/RagLangChainTest-main/tools/pdfTableSplitTest01.py#L74)：从本地目录加载 `microsoft/table-transformer-detection`，`local_files_only=True` 保证不联网。模型在**模块导入时**就加载，不是用到才加载。
- [`box_cxcywh_to_xyxy`](file:///D:/ai%20lianxi/RAG/project/RagLangChainTest-main/tools/pdfTableSplitTest01.py#L84) / [`rescale_bboxes`](file:///D:/ai%20lianxi/RAG/project/RagLangChainTest-main/tools/pdfTableSplitTest01.py#L89)：模型输出的是 DETR 风格的**归一化中心点+宽高** `(cx, cy, w, h)`，必须换算成 Pillow `crop()` 要的**左上右下** `(x1, y1, x2, y2)`，并乘以原图宽高还原到真实像素。这是本脚本最容易看错的一段。
- [`outputs_to_objects`](file:///D:/ai%20lianxi/RAG/project/RagLangChainTest-main/tools/pdfTableSplitTest01.py#L97)：`logits.softmax(-1).max(-1)` 取每个 query 的最大概率类别；[第136行](file:///D:/ai%20lianxi/RAG/project/RagLangChainTest-main/tools/pdfTableSplitTest01.py#L136) 手动追加一个 `"no object"` 标签兜住被判空的 query。
- [`detect_and_crop_save_table`](file:///D:/ai%20lianxi/RAG/project/RagLangChainTest-main/tools/pdfTableSplitTest01.py#L121)：把每个框裁下来存成 `table_images/page_N_i.png`，命名里带页号，这是后面元数据能追溯图片来源的基础。

> **缺置信度阈值**：`outputs_to_objects` 只过滤 `"no object"`，不过滤低分框（官方示例一般会卡 `score > 0.5`）。所以表格线被误识别时会多裁出碎片小图，这些碎片照样进库。

**(3) 视觉模型读图 + 灌库**

- [`encode_image`](file:///D:/ai%20lianxi/RAG/project/RagLangChainTest-main/tools/pdfTableSplitTest01.py#L153)：本地 PNG → base64。
- [`image_qa`](file:///D:/ai%20lianxi/RAG/project/RagLangChainTest-main/tools/pdfTableSplitTest01.py#L157)：标准 OpenAI 多模态消息（`text` + `image_url`，URL 用 `data:image/...;base64,` 内联），[第160行](file:///D:/ai%20lianxi/RAG/project/RagLangChainTest-main/tools/pdfTableSplitTest01.py#L160) 当前是 `qwen-vl-ocr`，`temperature=0, seed=42` 求稳定复现。返回值特意拼成 `"{图片路径} 中包含的内容是{模型输出}"`。
  > 这个拼接是一把双刃剑：好处是**图片路径本身进了文本和向量**，按路径检索也能命中；坏处是路径里的 `page_1_0` 这种无意义 token 会稀释语义向量。
- [`get_embeddings`](file:///D:/ai%20lianxi/RAG/project/RagLangChainTest-main/tools/pdfTableSplitTest01.py#L181) / [`generate_vectors`](file:///D:/ai%20lianxi/RAG/project/RagLangChainTest-main/tools/pdfTableSplitTest01.py#L190)：`text-embedding-v1`，批量 25 条（v1/v2 的单请求上限就是 25，若换 v3/v4 必须降到 10）。
- [`NewVectorDBConnector`](file:///D:/ai%20lianxi/RAG/project/RagLangChainTest-main/tools/pdfTableSplitTest01.py#L200)：`chromadb.Client(Settings(allow_reset=True))` 是**内存版**客户端，[第204行](file:///D:/ai%20lianxi/RAG/project/RagLangChainTest-main/tools/pdfTableSplitTest01.py#L204) 每次 `reset()`，所以这个库**不落地、进程结束即消失**，灌库和查询必须在同一次运行里完成。`add_images` 相比 `add_documents` 多传一个 `metadatas=[{"image": 路径}]`，这是整个设计的命门：**元数据留原图路径，检索命中后再回原图重问**。

**(4) 两跳查询 —— [第264-271行](file:///D:/ai%20lianxi/RAG/project/RagLangChainTest-main/tools/pdfTableSplitTest01.py#L264)**

```python
results = new_db_connector.search(query, 1)      # 第一跳：语义检索，定位是哪张图
metadata = results["metadatas"][0]
response = image_qa(query, metadata[0]["image"]) # 第二跳：拿原图重新精读，回答具体问题
```

这套"**先检索定位、再对原图精读**"的两跳结构，比直接把第一次 OCR 文本当答案准确得多——第一跳的描述只是索引，第二跳才是答案来源。

### 1.4 本机现状与已知问题

| 问题 | 说明 |
|---|---|
| 模型名 | [第160行](file:///D:/ai%20lianxi/RAG/project/RagLangChainTest-main/tools/pdfTableSplitTest01.py#L160) 已改成 `qwen-vl-ocr`。此前用 `qwen-vl-max` 报 404，因为 DashScope compatible-mode 不认这个名字 |
| 走 One API 的前提 | 必须挂**通用 OpenAI 类型**渠道指向 `https://dashscope.aliyuncs.com/compatible-mode`。用「阿里通义千问」类型渠道会被 native 转换器**丢掉 `image_url` 字段**，报 `400 url error` |
| ID 冲突 | [第215行](file:///D:/ai%20lianxi/RAG/project/RagLangChainTest-main/tools/pdfTableSplitTest01.py#L215) 和 [第228行](file:///D:/ai%20lianxi/RAG/project/RagLangChainTest-main/tools/pdfTableSplitTest01.py#L228) 都写死 `f"id{i}"`；同一 collection 先后 add 两批必然 DuplicateID。当前只调一次 `add_images`，暂时没暴雷 |
| 静默空向量 | `get_embeddings` 出错时 `return []`，`generate_vectors` 会把空列表 `extend` 进去，最终 `embeddings` 长度和 `documents` 不一致，报错信息却不指向真正原因 |
| 旧图残留 | `table_images/` 从不清空。上一轮裁出 3 张、这一轮只裁出 2 张时，第 3 张旧图仍会被灌库 |

---

## 二、pdfTableSplitTest02.py：文本路线

### 2.1 依赖与运行前提

| 位置 | 依赖 | 作用 |
|---|---|---|
| [:12](file:///D:/ai%20lianxi/RAG/project/RagLangChainTest-main/tools/pdfTableSplitTest02.py#L12) `pdfplumber` | 文本层抽取 | 保留阅读顺序的纯文本 |
| [:13](file:///D:/ai%20lianxi/RAG/project/RagLangChainTest-main/tools/pdfTableSplitTest02.py#L13) `camelot` | 表格结构抽取 | 输出 `pandas.DataFrame` |
| [:14](file:///D:/ai%20lianxi/RAG/project/RagLangChainTest-main/tools/pdfTableSplitTest02.py#L14) `openai` | 客户端 | 指向本机 One API |
| [:15](file:///D:/ai%20lianxi/RAG/project/RagLangChainTest-main/tools/pdfTableSplitTest02.py#L15) `pandas` | 表格载体 | `to_markdown` |
| 隐性 | `tabulate` | **代码里没 import**，但 [第91行](file:///D:/ai%20lianxi/RAG/project/RagLangChainTest-main/tools/pdfTableSplitTest02.py#L91) 的 `df.to_markdown()` 依赖它，缺则 `ImportError` |

同样**必须以 `tools/` 为工作目录**：[第171行](file:///D:/ai%20lianxi/RAG/project/RagLangChainTest-main/tools/pdfTableSplitTest02.py#L171) `"../input/健康档案(含表格02).pdf"`、[第172行](file:///D:/ai%20lianxi/RAG/project/RagLangChainTest-main/tools/pdfTableSplitTest02.py#L172) `"./output.txt"`。

头部注释第 9-10 行的 `brew install ghostscript` 是 macOS 旧教程遗留：**camelot 2.0 已改用 pdfminer.six + OpenCV，Windows 上无需 Ghostscript**，本机实测未装 gs 也能正常出表。

### 2.2 五步流水线

```
(1) extract_text_from_pdf   每页文本  ─┐
(2) extract_tables_from_pdf 每页表格  ─┤
(3) generate_table_description        ─┼─> (4) process_pdf 按页交织 ─> (5) save_to_text_file
    DataFrame→Markdown→LLM 中文叙述  ─┘
```

### 2.3 逐段解析

**Step 1 [`extract_text_from_pdf`](file:///D:/ai%20lianxi/RAG/project/RagLangChainTest-main/tools/pdfTableSplitTest02.py#L39)**
逐页 `page.extract_text()`，有文本才 append，无文本仅打印警告。**只取文本层**，扫描件对它完全无效。

**Step 2 [`extract_tables_from_pdf`](file:///D:/ai%20lianxi/RAG/project/RagLangChainTest-main/tools/pdfTableSplitTest02.py#L55)**
`camelot.read_pdf(pdf_path, pages='all', flavor='lattice', line_scale=40)`：

- `lattice` 靠**表格线**定结构，适合有线表格（健康档案正是这种）。
- `line_scale=40` 是线条检测灵敏度：默认 15，调大更"宽松"，细线/淡线也能被认成表格线；调小适合粗线，能避免把文字误判成线。
- 一个 `if tables.n == 0` 的**回退逻辑**：lattice 抽不到时换 `flavor='stream'`（靠文字间距推断，适合无边框表）再试一次。
- 过滤条件 `len(table.df) > 0 and len(table.df.columns) > 0`，丢弃空表。
- 本机实测这份 PDF 出 **9 个表格**。

**Step 3 [`generate_table_description`](file:///D:/ai%20lianxi/RAG/project/RagLangChainTest-main/tools/pdfTableSplitTest02.py#L85)**
`to_markdown(index=False)` 把 DataFrame 转 Markdown 文本，塞进 prompt，明确要求"**千万不要使用表格方式进行描述**"——这句话是整段的关键：让模型输出连续散文，后续按句子切分灌库才不会破坏语义。模型用 [第100行](file:///D:/ai%20lianxi/RAG/project/RagLangChainTest-main/tools/pdfTableSplitTest02.py#L100) 的 `qwen-plus`，`max_tokens=1200, temperature=0`。

> [第103-107行](file:///D:/ai%20lianxi/RAG/project/RagLangChainTest-main/tools/pdfTableSplitTest02.py#L103) 的 `content` 写成 list 形式但里面只有 `text` 项，属于多余的写法（OpenAI 协议允许，纯文本用字符串即可），不影响运行。

**Step 4 [`process_pdf`](file:///D:/ai%20lianxi/RAG/project/RagLangChainTest-main/tools/pdfTableSplitTest02.py#L118)**
`tables_per_page` 以 `table.page`（camelot 页码 **1-based**）为 key 归组；然后遍历文本，每页开头插一行 `--- 这是第N页的内容如下： ---`，该页若有表格则逐条插 `表格i的详细内容是:...`。

> 页分隔标记是给下游 RAG 用的**上下文锚点**：即使 chunk 切在某页中间，也能从文本里看出这段属于第几页。

**Step 5 [`save_to_text_file`](file:///D:/ai%20lianxi/RAG/project/RagLangChainTest-main/tools/pdfTableSplitTest02.py#L160)**
`encoding='utf-8'` 落盘。**02 不建向量库**，产物 `output.txt` 是喂给灌库脚本的上游语料，也就是说它天然分两步：先预处理，再入库检索。

### 2.4 已知问题

| 问题 | 说明 |
|---|---|
| **页码错位风险** | [第142行](file:///D:/ai%20lianxi/RAG/project/RagLangChainTest-main/tools/pdfTableSplitTest02.py#L142) 的 `enumerate(pages_text, start=1)` 遍历的是**过滤后**的文本列表。若第 2 页无文本被跳过，原第 3 页就会被当成"第 2 页"，其表格挂错页。当前 PDF 每页都有文本所以没暴露 |
| 异常被吞 | 三个 Step 全用 `try/except` 兜住并只打日志（[第78行](file:///D:/ai%20lianxi/RAG/project/RagLangChainTest-main/tools/pdfTableSplitTest02.py#L78)、[第112行](file:///D:/ai%20lianxi/RAG/project/RagLangChainTest-main/tools/pdfTableSplitTest02.py#L112)、[第154行](file:///D:/ai%20lianxi/RAG/project/RagLangChainTest-main/tools/pdfTableSplitTest02.py#L154)）。LLM 失败会返回固定串 `"An error occurred while generating the table description."` 并**照样写进 output.txt**，形成脏语料 |
| 无去重 | 同一个表格若被 lattice 和 stream 各识别一次，会产生重复描述（当前逻辑只二选一，不会同时跑） |
| 耗时 | 本机 9 个表格 = 9 次文本模型调用，全程约 1-2 分钟 |

---

## 三、两套方案对比

| 维度 | 01 视觉路线 | 02 文本路线 |
|---|---|---|
| 读表方式 | 整页渲染成图，TableTransformer 框出表格，视觉模型看图 | pdfplumber 取文本 + camelot 解析表格线 |
| 是否要求文本层 | **不要求**，扫描件/图片型 PDF 也能处理 | **强要求**，扫描件直接抽不到东西 |
| 结构还原精度 | 依赖 OCR 模型，合并单元格、跨列表头易错 | lattice 模式对有线表结构还原最准，行列干净 |
| 额外重依赖 | torch / transformers / 221MB 模型权重 | camelot / opencv / pandas / tabulate |
| 是否需要大模型 | 需要**多模态**模型（`qwen-vl-ocr`） | 只需要普通文本模型（`qwen-plus`），更好配 |
| 产物 | 内存版 Chroma，**当次运行内**可直接查询 | `output.txt`，需再走灌库脚本 |
| 查询设计 | 两跳：检索定位图片 → 原图重新精读 | 无查询逻辑，纯预处理 |
| 计算成本 | 本地跑检测模型 + 每张图一次 VQA | 无本地推理，每个表一次文本 LLM |
| 适用场景 | 扫描档案、拍照件、排版复杂的表 | 电子版 PDF，表格规整，要批量低成本预处理 |

**结论**：电子版规整 PDF 优先用 02（结构准、便宜、模型好配）；扫描件或 02 抽不到表时才上 01。两者也可以串起来——02 出主语料，01 只补 02 漏掉的图片型页面。

---

## 四、公共前置：One API 网关与环境约束

**网关**（两个脚本共用同一个 token）

```
base_url = http://localhost:3000/v1
模型：qwen-plus（文本）/ text-embedding-v1（向量，1536 维）/ qwen-vl-ocr（视觉）
```

视觉模型必须经**通用 OpenAI 类型**渠道转发到 `https://dashscope.aliyuncs.com/compatible-mode`；渠道「代理」字段**不要写 `/v1`**，One API 会自己拼 `/v1/chat/completions`。

**本机 `.venv` 的三方约束**（这是最近踩的坑，值得记下来）

```
chromadb 0.5.5   → 用了 np.float_，必须 numpy < 2
pandas  3.1.1    → 要求 numpy >= 2.0.2      ← 与上一行硬冲突
camelot 2.0.0    → 要求 pandas >= 2.2.2
```

已采用的解法：**保 numpy 1.26.4 + 把 pandas 降到 2.2.3**，三条约束同时满足，且 `chromaDB/demo001` 持久库无需重灌。表现为 `import camelot` 报 `C extension: None not built` 时，真正原因通常在上一层的 numpy 版本检查，不要被这句文案误导。

---

## 五、运行与验证清单

```
# 02（文本路线）
cd D:\ai lianxi\RAG\project\RagLangChainTest-main\tools
D:\ai lianxi\RAG\project\.venv\Scripts\python.exe pdfTableSplitTest02.py
#   预期：日志出现 9 次「LLM给出的表格的描述是」+「成功: 文本已经成功保存到./output.txt中」

# 01（视觉路线）
D:\ai lianxi\RAG\project\.venv\Scripts\python.exe pdfTableSplitTest01.py
#   预期：每页「识别的表格数量: N」→ 生成 input/健康档案(含表格01)/table_images/*.png
#        → 「最后得到的检索结果是」带 image 路径 → 末行给出针对原图的回答
```

两个脚本都**不校验模型是否返回了合理内容**，日志里那行描述就是全部证据，所以抽查 `output.txt` 和最后一行回答的语义正确性是必要动作，不能只看没报错。

> 补充：01 的示例问题 `"王五的空腹血糖是多少"` 与该 PDF 内容未必匹配（档案主体是张三九），检索命中与否取决于当次裁图结果，**不代表代码链路有问题**——换问题前最好先确认对应字段确实被裁进了某张表格小图里。
