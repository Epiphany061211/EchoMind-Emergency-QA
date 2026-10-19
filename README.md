# EchoMind-Emergency-QA
Multi-agent RAG system for emergency response plans: DAG-based role-structure retrieval to suppress hallucination, plus an evaluation harness for long-horizon reliability.
地质灾害应急预案智能助手（EchoMind 改造版）使用说明
=====================================================

一、这是什么
基于多 Agent 架构的应急预案问答系统。覆盖省级与市级两份地质灾害应急预案（真实预案为政府公开发文，仓库内以虚构示例数据演示数据结构），支持查询单位的上级/平级/下级关系、职责与响应权限（职能骨架查表，DAG 结构），以及预案条文的语义检索（ChromaDB 向量库，回答标注预案名称与页码）。示例数据见 data/architecture_example/。

二、运行环境
- Windows 10/11
- Anaconda/Miniconda，已创建 conda 环境 emergency-echomind
  （依赖版本见 requirements 及钉版本记录：
   huggingface_hub==0.26.5 transformers==4.46.3
   tokenizers==0.20.3 sentence-transformers==3.1.1）

三、启动步骤（共 3 步）
1. 首次使用：把 .env.example 复制一份改名为 .env，填入两个模型的密钥。
2. 双击 启动.bat，等待终端出现 "EchoMind 已就绪"。
3. 双击 应急预案助手前端.html，在浏览器里提问。

四、演示建议路径
1. 首屏点击小球，查看"我有什么不同"（乐队比喻 + 六道关卡）。
2. 依次点四个示例问题卡片：
   - 宜宾市公安局的上级部门是谁？（架构·上级）
   - 省公安厅有哪些平级单位？（架构·平级，39 个）
   - 三级应急响应由谁决定启动？（架构·权限，省市对照）
   - 地质灾害发生后群众转移安置怎么做？（条文检索）
   注：每条间隔约 20 秒，避开模型限速。
3. 右上角"监控面板"展示实时监控（成功率/延迟/工具调用），
   /docs 为 Swagger 接口文档，/eval/run 触发 LLM 自动评测。

五、目录说明
- api/ agents/ core/ mcp/ memory/ monitor/ evaluation/  后端代码
- org_graph.py        职能骨架引擎
- skills/             三个子 Agent 的业务规范
- data/architecture/  两份预案的职能骨架 JSON
- data/chroma/        预案条文向量库
- 应急预案助手前端.html  前端页面（浏览器直接打开）
