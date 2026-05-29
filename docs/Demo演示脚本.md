# MultiModal-QC Demo 演示脚本

## Demo形态说明

本项目的Demo分三层：

1. **可交互网页Demo**：打开 `http://localhost:7860`，可以选择模型、上传图片、切换 `defect_3d` 和 `drilling_compliance` 场景，并查看检测报告、Pipeline中间结果和历史记录。
2. **作品集截图Demo**：`docs/assets/` 下保存关键页面截图，用于技术报告、README、简历作品集或面试PPT展示。
3. **录屏视频Demo**：后续可基于可交互网页录制60-90秒走查视频，展示启动页面、上传样本、生成报告、查看Pipeline和历史记录。

当前已经完成前两层；录屏视频属于展示包装，不影响项目核心功能。

## 演示前准备

1. 进入项目目录：`E:\MultiModal-QC`
2. 启动前端：`.\.venv\Scripts\python.exe serve\app.py`
3. 打开页面：`http://localhost:7860`
4. 如需真实推理结果，再启动完整服务：`start.bat`

说明：仅展示UI和项目结构时，只启动Gradio前端即可；真实检测需要FastAPI后端加载Qwen2-VL、ResNet50和YOLOv8n-cls模型。

---

## 3分钟讲解顺序

### 1. 项目一句话

这是一个基于 Qwen2-VL-2B + QLoRA 的工业多模态质检平台，目标不是只判断“合格/不合格”，而是输出可复核的中文质检报告。项目覆盖两个场景：MVTec图像缺陷检测和钻井视频合规检查Demo。

### 2. 模型对比

在MVTec-AD测试集173张图片上：

| 方法 | Accuracy | F1 | 说明 |
|------|----------|----|------|
| ResNet50 | 90.12% | 0.905 | 仅二分类 |
| YOLOv8n-cls | 83.24% | 0.827 | 仅二分类 |
| Qwen2-VL-2B + QLoRA | 89.02% | 0.893 | 支持中文缺陷描述 |

讲解重点：Qwen2-VL的F1接近ResNet50，但额外具备缺陷类型、位置、严重程度和处置建议生成能力。

### 3. 系统架构

输入图片或钻井视频后，系统先做数据预处理，再进入 Qwen2-VL + QLoRA 推理层，随后通过 Perceive、Detect、Decide、Report 四阶段 Agent Pipeline 输出报告。前端用 Gradio，后端用 FastAPI，检测记录存入 SQLite。

### 4. Demo页面

打开“快速检测”页，展示三件事：

- 可选择 Qwen2-VL、ResNet50、YOLOv8n-cls 三种模型
- 可选择 `defect_3d` 和 `drilling_compliance` 两种场景
- 输出区域包含检测可视化、报告文本和模型原始输出

再打开“完整 Pipeline”页，说明四阶段中间结果为什么能增强可解释性。

最后打开“历史记录”页，说明检测结果会持久化到 SQLite，方便追溯。

### 5. 钻井场景边界

钻井场景已经完成本地私有视频和检查表的数据接入流程验证，并完成QA转换脚本和视频场景Demo入口。由于原始数据不适合公开分发，且检查记录不能严格对应到每个视频片段，所以当前不宣称独立训练指标。下一步需要补充带时间戳的视频标注。

---

## 截图文件

| 文件 | 内容 |
|------|------|
| `docs/assets/demo-gradio-inspect.png` | 快速检测页 |
| `docs/assets/demo-overview-metrics.png` | 模型对比与项目亮点 |
| `docs/assets/demo-defect-result.png` | capsule裂纹样本检测结果 |
| `docs/assets/demo-pipeline.png` | 完整Pipeline页 |
| `docs/assets/demo-history.png` | 历史记录页 |
