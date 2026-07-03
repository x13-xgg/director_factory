# Director Factory — 问题清单 & 解决方案

## 架构概览

```
输入 Prompt → Writer → Storyboarder → Art Director → Character Director
  → Scheduler → Cinematographer（循环拍摄） → VoiceActor → SoundDesigner
  → Colorist → VFX/Subtitles → Editor → 输出视频
```

---

## P0 — 核心缺陷（影响基本可用性）

### 问题 1：生成的视频没有声音

**现象**：输出 mp4 只有画面轨道，完全没有声音。

**根因**：`EditorAgent`（src/agents/editor.py:17）虽然接收了 `audio_assets` 参数，但完全忽略了它。`timeline_assemble` 工具（src/tools/render.py:39）只用 `ffmpeg concat` 拼接图像序列，没有任何音频轨。而音频管线（TTS → SFX → BGM → Mix）实际运行了，产物 `outputs/final_mix.wav` 已经生成，只是最后一步没合进去。

**修复**：
1. `EditorAgent.handle_task()` 增加 `audio_assets` 参数解析，提取 `final_mix_path`
2. `TimelineAssembleTool` 增加 `audio_path` 参数
3. ffmpeg 命令改为：`ffmpeg -f concat -i concat.txt -i audio.wav -c:v libx264 -c:a aac -map 0:v -map 1:a -shortest output.mp4`

**涉及文件**：
- `src/agents/editor.py` — 接收音频参数
- `src/tools/render.py` — ffmpeg 命令加音频轨

---

### 问题 2：镜头帧之间完全零关联

**现象**：同一场景的不同镜头：角色长相不同、光照色温跳变、背景不一致、构图无逻辑连接。看起来像随机图片合集，不是连续的视频。

**根因**：`CinematographerAgent`（src/agents/cinematographer.py:34）注释明确写了"只为单个镜头负责，不关心前后镜头关系"。每个 shot 独立：
- 独立拼 SD prompt（无前后帧上下文）
- 随机 seed 生成
- 独立评分重试
- 无角色一致性约束，无环境连续性约束，无光照延续

**修复方案**：三级一致性机制

#### a) 场景级 Master Reference Frame
每个场景开拍前，先生成一张"定场镜头"作为该场景的视觉锚点：
- 锁定 seed 基值
- 同场景后续镜头以 master frame 做 ControlNet reference-only / IP-Adapter
- 统一 color grade 和光照参数

#### b) Shot-to-Shot Continuity Context
每个镜头生成时注入前一帧的实际状态：
```python
continuity_context = {
    "prev_frame_path": "...",
    "character_states": {
        "char_001": {"position": "frame_left", "expression": "neutral", "pose": "standing"},
    },
    "lighting_state": {"color_temp": 5600, "key_dir": [0.3, 0.5, -0.8]},
    "camera_direction": "left_to_right",
}
```
Cinematographer 的 `_build_prompt()` 将这些信息编码进 prompt。

#### c) 角色一致性管线（串起已有组件）
项目已有 `LoRATrainerTool`、`FaceConsistencyCheckerTool`、`CharacterConsistencyCheckerTool`，但未真实串联：
```
角色定稿 → 生成参考图 → 训练角色 LoRA（CivitAI 搜索或本地训练）
  → 每个含该角色的镜头挂 LoRA + trigger_word 生成
  → 事后 face_consistency 校验 → 不通过则重试
```

**涉及文件**：
- `src/agents/cinematographer.py` — 加入 continuity_context
- `src/agents/director.py` — `_pipeline_shots` 中管理场景级 master frame 和前后帧状态传递
- `src/agents/character_director.py` — 串起 LoRA 训练→应用→校验流程
- `src/tools/character_tools.py` — LoRA trainer 真实化

---

## P1 — 核心功能缺失

### 问题 3：不能分析文本，只能接受一句话 prompt

**现象**：用户只能输入""废墟中寻找花朵的机器人"这样的短 prompt，不能输入故事脚本、小说片段、分场大纲。

**根因**：`WriterAgent`（src/agents/writer.py:24）把用户输入直接当"创意"，让 LLM 凭空创作。没有"理解已有文本→提取电影化要素"的分析能力。真正的影视制作第一步是剧本分析，这步完全跳过。

**修复方案**：新增 **ScriptAnalystAgent** + **CinematicBreakdown** 协议

```
输入（长文本/剧本/小说）
  ↓
ScriptAnalystAgent（新增）
  ├─ 叙事节拍分析: 开端/发展/冲突/高潮/结局
  ├─ 可视觉化元素提取: 识别哪些情感/动作可通过画面传达
  ├─ 角色关系空间映射: 情感距离 → 画面中的物理距离
  ├─ 情绪节奏图: 逐拍的情绪强度 + 转调点
  ├─ 视觉母题识别: 重复出现的意象、颜色、光线
  └─ 输出 → CinematicBreakdown
      ↓
WriterAgent（改造: 基于 CinematicBreakdown 细化，不再凭空创作）
      ↓
StoryboarderAgent（改造: 基于 CinematicBreakdown 编译分镜，有据可依）
```

CinematicBreakdown 数据结构：
```python
@dataclass
class CinematicBreakdown:
    narrative_beats: list[NarrativeBeat]    # [{beat_id, type, description, intensity}]
    visual_motifs: list[VisualMotif]        # [{motif_id, description, appearances[]}]
    character_spatial_map: dict             # {char_id: {scene_id: [x, y, z]}}
    emotion_rhythm: list[EmotionBeat]       # [{timestamp, emotion, intensity, transition}]
    scene_density: list[SceneDensity]       # [{scene_id, visual_complexity, shot_count_estimate}]
    pov_strategy: str                       # "omniscient" | "single_character" | "alternating"
    key_images: list[str]                   # 必须出现的画面描述
```

**涉及文件**：
- `src/agents/script_analyst.py` — **新增**
- `src/data/cinematic_breakdown.py` — **新增**（协议定义）
- `src/agents/writer.py` — 接收 CinematicBreakdown
- `src/agents/storyboarder.py` — 基于 CinematicBreakdown 编译
- `src/agents/director.py` — 管线中插入 ScriptAnalyst
- `src/pipeline/runner.py` — 注册新 Agent

---

### 问题 4：管线中间进度不可见

**现象**：点"启动生产"后，仪表盘始终显示 `shots: 0/0, progress: 0%`，直到管线完成才跳到 100%。

**根因**：summary.json 只在两个时刻写入：
1. `api_run` 启动时写初始化（全 0）
2. `runner.run()` 结束时写最终结果

中间状态完全不暴露。

**修复方案**：
1. 在 Director 的 `_pipeline_shots` 每次循环后写中间 summary
2. 前端项目详情页定时轮询 `/api/projects/{id}` 获取实时进度
3. 增加 phase 信息：当前阶段（creative / characters / shots / audio / post）

**涉及文件**：
- `src/agents/director.py` — 循环中写中间状态
- `src/web/routes.py` — 前端轮询已有 API，无需改动
- `src/web/templates/project_detail.html` — 加入自动刷新

---

## P2 — 质量提升

### 问题 5：帧是静态图片，没有运动

**现象**：每个"镜头"是一张静态 PNG，Editor 只是把它们按 duration 时长拉伸为视频帧。没有 Ken Burns、没有运镜模拟、没有转场动画。

**修复方案**：对每帧施加程序化运动（ffmpeg zoompan filter）：
- `push_in` → zoom 1.0 → 1.08
- `pull_out` → zoom 1.08 → 1.0
- `pan_left` → x 平移
- `static` → 仅 Ken Burns 微动 (zoom 1.0 → 1.03)
- `handheld` → 加随机微抖动

**涉及文件**：`src/tools/render.py` — ffmpeg 命令加入 zoompan filter

---

### 问题 6：Mock 占位太多，管线实际跑空

| 组件 | 当前状态 | 影响 |
|------|---------|------|
| 图像生成 | 优先 mock（空 PNG），ComfyUI 失败时回退 | 所有帧为空 |
| SFX 匹配 | 返回不存在的 mock 路径（`sfx/wind_ambient.wav`） | 无音效 |
| BGM 匹配 | mock | 无音乐 |
| 构图评分 | 返回固定 ≥0.85 | 质量审核无意义 |
| 光照连续性 | mock | 无法检测光照问题 |
| 情绪对齐 | mock | 无法检测情绪表达 |
| LoRA 训练 | mock | 角色一致性无保障 |

**修复方案**：这些都是"接入真实服务"的工作。
- 图像：ComfyUI 已配置（env: `COMFYUI_URL=http://127.0.0.1:8188`），需排查为何 mock 回退优先于 ComfyUI
- 音频：使用 edge_tts（已配置），音效需接入 freesound API 或本地音效库，BGM 可用生成式音乐 API
- 评分：接入 real CV 模型（CLIP、ArcFace、美学评分模型）

---

### 问题 7：质量审核量化不足

**现状**：`_quality_check`（director.py:674）中光照评分固定 0.88，情绪评分固定 0.85，只有构图分和角色一致性分来自实际计算（且角色分也是 mock）。

**修复方案**：
1. 用 CLIP score 做 prompt-image 对齐度
2. 用 dedicated 美学评分模型做构图判断
3. 用 ArcFace 做人脸相似度
4. 用色温检测做光照连续性

---

## P3 — 基础设施

### 问题 8：Windows GBK 编码问题（已修复）

`Path.write_text()` 在 Windows 中文系统默认使用 cp936/GBK，导致 JSON 文件编码不一致。

**修复**：所有 `write_text()` 显式加 `encoding="utf-8"`，读取端 `_read_json()` 支持 UTF-8/GBK 自动回退。

---

### 问题 9：进度显示 100% 但状态 running（已修复）

**根因**：`_scan_projects` 中进度计算未检查 status，当 `shots_completed == shots_planned` 时直接返回 100%。

**修复**：非 `ok` 状态下进度最高 99%。

---

## 修复优先级

| 优先级 | 问题 | 改动量 | 依赖 |
|--------|------|--------|------|
| **P0-1** | 音频不入视频 | 小 | 无 |
| **P0-2** | 帧间零关联 | 大 | 无 |
| **P1-1** | 文本→镜头分析 | 中 | 无 |
| **P1-2** | 中间进度 | 小 | 无 |
| **P2-1** | 静态帧运动 | 中 | 无 |
| **P2-2** | Mock→真实服务 | 大 | 外部 API/模型 |
| **P2-3** | 质量审核量化 | 中 | CV 模型 |

---

## 理想管线架构（目标状态）

```
输入文本（任意长度）
  │
  ▼
ScriptAnalystAgent ─── 文本 → CinematicBreakdown
  │
  ▼
WriterAgent ─── CinematicBreakdown → Screenplay（结构化）
  │
  ▼
StoryboarderAgent ─── Screenplay + Breakdown → ShotList（精确分镜）
  │
  ▼
ArtDirectorAgent ─── ShotList → StyleGuide（视觉规范）
  │
  ▼
CharacterDirectorAgent ─── Characters → LoRA 训练 → CharacterAssets
  │
  ├─ SceneMasterFrame（每场景定场锚点）
  │
  ▼
SchedulerAgent ─── 调度拍摄计划
  │
  ▼
CinematographerAgent ─── 逐镜头拍摄
  │  每个镜头注入:
  │  ├─ 前帧 continuity context
  │  ├─ 场景 master reference
  │  ├─ 角色 LoRA trigger
  │  └─ 风格约束
  │
  ▼ （实时质量审核 + 重试）
  │
  ▼
VoiceActorAgent ─── TTS 对白生成
  │
  ▼
SoundDesignerAgent ─── SFX 匹配 + BGM + 混音 = final_mix.wav
  │
  ▼
ColoristAgent ─── 场景级色彩匹配
  │
  ▼
VFXSubtitlesAgent ─── 字幕
  │
  ▼
EditorAgent ─── 帧 → 视频片段（含 zoompan）+ 音频 mux
  │
  ▼
最终输出 mp4（有画面 + 有声音 + 有字幕）
```
