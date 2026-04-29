# AGENTS

> [!URGENT]
> **研究性项目 (Research Project)**
> 1. 本项目为 MVP（最小可行性产品），严禁过度工程化。
> 2. 你的所有思考过程和回复必须使用 **简体中文**。

## 1. 项目元数据 (Metadata)
- **核心目标**: 在本项目中，在不重训、不大改通用视觉大模型的前提下，用极轻量的跨光谱模态适配器（CSMA）把红外任务做到接近或超过重预训练的红外专用大模型（如 M-SpecGene），并保住大模型原有的开放词汇 / 零样本能力。。
- **项目类型**: MVP / 研究性项目
- **后端架构**: Python 3.11
- **版本管理**: Git
- **Conda 环境**: ExplicitLLM (Python 3.11)

## 2. 常用命令 (Commands)

### 2.1 Conda 环境管理

> [!CRITICAL]
> **所有 Python 相关命令必须在 ExplicitLLM 环境中执行**
> - 使用 `conda run -n RGBtest <command>` 确保命令在正确环境中运行
> - 或在命令前显式添加 `source activate RGBtest &&`
> - 如果需要使用llm可以依据 '.env'环境变量文件使用

> [!CRITICAL]
> **GPU 使用规范：必须且只能使用 GPU 0**
> - 所有涉及 GPU 的命令必须通过 `CUDA_VISIBLE_DEVICES=0` 指定
> - 示例：`CUDA_VISIBLE_DEVICES=0 conda run -n RGBtest python xxx`
> - **严禁**硬编码其他 GPU 索引，**严禁**省略 `CUDA_VISIBLE_DEVICES` 导致占用其他 GPU

```bash
# 激活项目环境（交互式 shell）
conda activate RGBtest

# 推荐：使用 conda run 执行命令（自动使用正确环境）
conda run -n RGBtest pip install xxx
conda run -n RGBtest python -m pytest xxx #注意不能conda run -n RGBtest pytest xxx，因为这样子pytest不会调用ExplicitLLM
conda run -n RGBtest python xxx

# 或者：在命令前激活环境
source activate RGBtest && xxx
```

### 2.2 代码质量检查
```bash
# 代码格式化
conda run -n RGBtest ruff format xxx

# 代码检查并自动修复
conda run -n RGBtest ruff check xxx --fix
```


### 2.3 main.py CLI 入口（生产命令）

| 子命令 | 功能 | 前置模块 |
|--------|------|----------|
| `build-knowledge` | Phase 0 知识构建 | §1.2+§1.3+§1.5 |
| `train --phase {0,1,2,3}` | 训练管线 | §1.7+§1.9+§1.10 |
| `eval` | 评测入口 | §1.10 |
| `answer` | 端到端 QA | §1.10 |

用法：`conda run -n RGBtest python main.py [--config path] [--device dev] [--override key=value ...] {子命令}`

模块验证不在 main.py 中，统一通过 `tests/integration/` 执行。

## 3. 标准作业程序 (Standard Operating Procedure)
> **Agent 必须严格遵守以下生命周期执行任务：**

### Phase 1: 规划与设计 (Planning)
1. **查阅规格 (Read Specs)&讨论**: 在撰写计划前，**必须**仔细阅读 `docs/` 下对应的文档与`.report`下的项目整理架构，并使用GitNexus MCP来了解整个项目的最新情况。对于不理解的地方请与人类进行多轮讨论，确保理解人类的设计意图。
2. **计划 (Plan)**: 正式编码前，**必须**使用plan模式输出开发计划，内容必须严格包含：
   - **1.1 摘要 (Summary)**: 1-2句话的简单总结。
   - **1.2 审查点 (User Review Required)**: 明确列出整个计划中不清楚、需要用户审查和确认的部分。若无，请注明"无"。
   - **1.3 拟议变更 (Proposed Changes)**:
     - 以 **文件名 + 修改内容** 的形式列出。
     - 修改内容必须精确到 **函数/方法级别 (Function-level)**。
     - 明确标识 `[NEW]`, `[MODIFY]`, `[DELETE]`。
   - **1.4 验证计划 (Verification Plan)**: 具体描述如何验证修改是否成功（如具体的测试命令、预期日志输出等）。
4. **等待 (Wait)**: **必须** 暂停并等待用户审核开发计划。用户批准后方可进入下一阶段。

### Phase 2: 执行与验证 (Execution & Verification)
1. **编码 (Coding)**: 审核通过后，开始编写代码。
2. **验证 (Verify)**:
   - **环境检查**: 确保所有命令在 RGBtest 环境中执行（使用 `conda run -n RGBtest`）
   - **运行验证命令**:
     - *失败*: 回到编码阶段修复，直到通过。
     - *成功*: 进入下一步。

## 4. 核心规则 (Rules)

### 4.1 代码开发规范 (Code Style)
- **类型系统**: 强制所有函数签名包含完整类型注解 (`Union`, `Dict`, `Optional` 等)。
- **文档**: 所有模块、类、方法必须包含 **中文 Docstring** (功能、参数、返回值、关键实现细节)。
- **MVP原则**:
  - **必须** 必须在`tests/`目录下编写测试代码。
  - **严禁** 使用默认参数掩盖仅需逻辑（必须显式传递关键参数）。
  - **必需** 运行时检查：关键维度、设备一致性必须通过 assertion 或 if 验证。
- **代码组织**:
  - 使用阶段化注释 (`# Phase 1`, `# Phase 2`) 组织复杂逻辑。
  - 接口返回值需包含完整诊断信息（输出、损失、统计），使用条件标志控制。
- **命名与依赖**:
  - 类名 `PascalCase`，变量描述性命名，私有变量前缀 `_`。
  - 导入顺序：标准库 → 第三方库 → 项目内部。
- **日志与错误处理**: 使用 `utils/logger_system.py` 的 `log_msg()`, `log_json()`, `ensure()`, `log_exception()`
  - 禁用 `print()`，`log_msg("ERROR")` 不自动抛出异常，输出到 `logs/system.log` + `logs/metrics.json`
- **功能修改**:
  - **必须** 不考虑向后兼容，直接修改原文件。代码简洁性优先。

### 4.2 配置管理规范
- **优先级**: CLI args > `.env` > YAML，三者统一归口到 dataclass
- **文件**: `config/default.yaml`（全量非敏感配置，必须写全）, `.env`（敏感信息，不提交）, `.env.example`（模板）

### 4.3 测试组织规范
- **目录**: `tests/{unit,integration,e2e}/test_*.py`，最低覆盖率 80%
- **运行**: `conda run -n ExplicitLLM pytest tests/unit/ --cov=utils --cov-report=term-missing`

#### Agent 测试输出规范

> **main.py 是生产 CLI 入口**（build-knowledge / train / eval / answer），不含 demo 或验证逻辑。
> 模块验证通过 `tests/integration/test_{module}_flow.py` + Markdown 报告完成。
> 严禁在 main.py 中使用 MagicMock/玩具参数。

| 要素 | 规范 |
|------|------|
| **输出位置** | `tests/outputs/<test_module>/<test_name>_<timestamp>.md` |
| **触发时机** | 所有涉及 Agent 执行的测试 |
| **内容要求** | 任务描述、每步 Agent 输入/输出/推理过程、工具调用、最终结果 |
| **格式要求** | 结构化 Markdown（标题、代码块、列表），人类可读 |
| **分析方式** | Agent 读取 MD 文件，评估推理质量、任务完成度、代码正确性 |

**示例结构**:
```markdown
# Agent 测试: <test_name>
## 任务: <task>
## Step 1: <AgentName>
- 输入: ...
- 输出: ...
- 推理: ...
## Step 2: ...
## 最终结果: ...
```

**pytest 集成**: 使用 fixture 或工具类自动保存，测试结束后输出文件路径。


## 5. 上下文获取与迷途指南 (Context & Navigation)
！注意 `Reference` 文件夹下的所有代码和文件都来自于参考项目，不是本项目的代码。

| 需求 | 文档路径 | 说明 |
|------|----------|------|
| 项目目标与背景 | `README.md` | 核心业务逻辑与项目定性 |
| 架构与模块设计 | `.report/CODEMAPS/{architecture,backend,data}.md` | 整体架构、分层设计、模块依赖 |
| 该项目最主要的参考项目 | `Reference/Tree-TRM`| 特定模块的详细设计 |
| 其他参考项目 | `Reference/PageIndex_ Next-Generation Vectorless, Reasoning-based RAG.md`和 `Reference/Tree-TRM`|  |
| 模块构建状态 | `docs/TD.md` 各 §1.x 节顶部 | 实现状态、依赖、验证 checkpoint |

## 6. 输出规范

### 6.1 语言要求
- 所有输出语言: **中文**

### 6.2 信息密度原则
- **优先使用**:
  - 简洁文本描述
  - 伪代码（而非完整代码）
  - 表格（对比、配置、参数说明）
  - 流程图（Mermaid）
  - 项目符号列表
- **避免使用**:
  - 大段完整代码（信息密度低，可读性差）
  - 冗长的自然语言解释
- **核心原则**: 用最少的字符传递最多的信息
`

<skills_system priority="1">

## Available Skills

<!-- SKILLS_TABLE_START -->
<usage>
When users ask you to perform tasks, check if any of the available skills below can help complete the task more effectively. Skills provide specialized capabilities and domain knowledge.

How to use skills:
- Invoke: `npx openskills read <skill-name>` (run in your shell)
  - For multiple: `npx openskills read skill-one,skill-two`
- The skill content will load with detailed instructions on how to complete the task
- Base directory provided in output for resolving bundled resources (references/, scripts/, assets/)

Usage notes:
- Only use skills listed in <available_skills> below
- Do not invoke a skill that is already loaded in your context
- Each skill invocation is stateless
</usage>

<available_skills>

<skill>
<name>algorithmic-art</name>
<description>Creating algorithmic art using p5.js with seeded randomness and interactive parameter exploration. Use this when users request creating art using code, generative art, algorithmic art, flow fields, or particle systems. Create original algorithmic art rather than copying existing artists' work to avoid copyright violations.</description>
<location>project</location>
</skill>

<skill>
<name>brand-guidelines</name>
<description>Applies Anthropic's official brand colors and typography to any sort of artifact that may benefit from having Anthropic's look-and-feel. Use it when brand colors or style guidelines, visual formatting, or company design standards apply.</description>
<location>project</location>
</skill>

<skill>
<name>canvas-design</name>
<description>Create beautiful visual art in .png and .pdf documents using design philosophy. You should use this skill when the user asks to create a poster, piece of art, design, or other static piece. Create original visual designs, never copying existing artists' work to avoid copyright violations.</description>
<location>project</location>
</skill>

<skill>
<name>claude-api</name>
<description>"Build, debug, and optimize Claude API / Anthropic SDK apps. Apps built with this skill should include prompt caching. Also handles migrating existing Claude API code between Claude model versions (4.5 → 4.6, 4.6 → 4.7, retired-model replacements). TRIGGER when: code imports `anthropic`/`@anthropic-ai/sdk`; user asks for the Claude API, Anthropic SDK, or Managed Agents; user adds/modifies/tunes a Claude feature (caching, thinking, compaction, tool use, batch, files, citations, memory) or model (Opus/Sonnet/Haiku) in a file; questions about prompt caching / cache hit rate in an Anthropic SDK project. SKIP: file imports `openai`/other-provider SDK, filename like `*-openai.py`/`*-generic.py`, provider-neutral code, general programming/ML."</description>
<location>project</location>
</skill>

<skill>
<name>doc-coauthoring</name>
<description>Guide users through a structured workflow for co-authoring documentation. Use when user wants to write documentation, proposals, technical specs, decision docs, or similar structured content. This workflow helps users efficiently transfer context, refine content through iteration, and verify the doc works for readers. Trigger when user mentions writing docs, creating proposals, drafting specs, or similar documentation tasks.</description>
<location>project</location>
</skill>

<skill>
<name>docx</name>
<description>"Use this skill whenever the user wants to create, read, edit, or manipulate Word documents (.docx files). Triggers include: any mention of 'Word doc', 'word document', '.docx', or requests to produce professional documents with formatting like tables of contents, headings, page numbers, or letterheads. Also use when extracting or reorganizing content from .docx files, inserting or replacing images in documents, performing find-and-replace in Word files, working with tracked changes or comments, or converting content into a polished Word document. If the user asks for a 'report', 'memo', 'letter', 'template', or similar deliverable as a Word or .docx file, use this skill. Do NOT use for PDFs, spreadsheets, Google Docs, or general coding tasks unrelated to document generation."</description>
<location>project</location>
</skill>

<skill>
<name>frontend-design</name>
<description>Create distinctive, production-grade frontend interfaces with high design quality. Use this skill when the user asks to build web components, pages, artifacts, posters, or applications (examples include websites, landing pages, dashboards, React components, HTML/CSS layouts, or when styling/beautifying any web UI). Generates creative, polished code and UI design that avoids generic AI aesthetics.</description>
<location>project</location>
</skill>

<skill>
<name>internal-comms</name>
<description>A set of resources to help me write all kinds of internal communications, using the formats that my company likes to use. Claude should use this skill whenever asked to write some sort of internal communications (status reports, leadership updates, 3P updates, company newsletters, FAQs, incident reports, project updates, etc.).</description>
<location>project</location>
</skill>

<skill>
<name>mcp-builder</name>
<description>Guide for creating high-quality MCP (Model Context Protocol) servers that enable LLMs to interact with external services through well-designed tools. Use when building MCP servers to integrate external APIs or services, whether in Python (FastMCP) or Node/TypeScript (MCP SDK).</description>
<location>project</location>
</skill>

<skill>
<name>pdf</name>
<description>Use this skill whenever the user wants to do anything with PDF files. This includes reading or extracting text/tables from PDFs, combining or merging multiple PDFs into one, splitting PDFs apart, rotating pages, adding watermarks, creating new PDFs, filling PDF forms, encrypting/decrypting PDFs, extracting images, and OCR on scanned PDFs to make them searchable. If the user mentions a .pdf file or asks to produce one, use this skill.</description>
<location>project</location>
</skill>

<skill>
<name>pptx</name>
<description>"Use this skill any time a .pptx file is involved in any way — as input, output, or both. This includes: creating slide decks, pitch decks, or presentations; reading, parsing, or extracting text from any .pptx file (even if the extracted content will be used elsewhere, like in an email or summary); editing, modifying, or updating existing presentations; combining or splitting slide files; working with templates, layouts, speaker notes, or comments. Trigger whenever the user mentions \"deck,\" \"slides,\" \"presentation,\" or references a .pptx filename, regardless of what they plan to do with the content afterward. If a .pptx file needs to be opened, created, or touched, use this skill."</description>
<location>project</location>
</skill>

<skill>
<name>skill-creator</name>
<description>Create new skills, modify and improve existing skills, and measure skill performance. Use when users want to create a skill from scratch, edit, or optimize an existing skill, run evals to test a skill, benchmark skill performance with variance analysis, or optimize a skill's description for better triggering accuracy.</description>
<location>project</location>
</skill>

<skill>
<name>slack-gif-creator</name>
<description>Knowledge and utilities for creating animated GIFs optimized for Slack. Provides constraints, validation tools, and animation concepts. Use when users request animated GIFs for Slack like "make me a GIF of X doing Y for Slack."</description>
<location>project</location>
</skill>

<skill>
<name>template</name>
<description>Replace with description of the skill and when Claude should use it.</description>
<location>project</location>
</skill>

<skill>
<name>theme-factory</name>
<description>Toolkit for styling artifacts with a theme. These artifacts can be slides, docs, reportings, HTML landing pages, etc. There are 10 pre-set themes with colors/fonts that you can apply to any artifact that has been creating, or can generate a new theme on-the-fly.</description>
<location>project</location>
</skill>

<skill>
<name>web-artifacts-builder</name>
<description>Suite of tools for creating elaborate, multi-component claude.ai HTML artifacts using modern frontend web technologies (React, Tailwind CSS, shadcn/ui). Use for complex artifacts requiring state management, routing, or shadcn/ui components - not for simple single-file HTML/JSX artifacts.</description>
<location>project</location>
</skill>

<skill>
<name>webapp-testing</name>
<description>Toolkit for interacting with and testing local web applications using Playwright. Supports verifying frontend functionality, debugging UI behavior, capturing browser screenshots, and viewing browser logs.</description>
<location>project</location>
</skill>

<skill>
<name>xlsx</name>
<description>"Use this skill any time a spreadsheet file is the primary input or output. This means any task where the user wants to: open, read, edit, or fix an existing .xlsx, .xlsm, .csv, or .tsv file (e.g., adding columns, computing formulas, formatting, charting, cleaning messy data); create a new spreadsheet from scratch or from other data sources; or convert between tabular file formats. Trigger especially when the user references a spreadsheet file by name or path — even casually (like \"the xlsx in my downloads\") — and wants something done to it or produced from it. Also trigger for cleaning or restructuring messy tabular data files (malformed rows, misplaced headers, junk data) into proper spreadsheets. The deliverable must be a spreadsheet file. Do NOT trigger when the primary deliverable is a Word document, HTML report, standalone Python script, database pipeline, or Google Sheets API integration, even if tabular data is involved."</description>
<location>project</location>
</skill>

</available_skills>
<!-- SKILLS_TABLE_END -->

</skills_system>
