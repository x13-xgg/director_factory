"""image_gen 工具 — ComfyUI / SDXL 图像生成 (委托 CloudImageGenTool 多 Provider 回退)"""

from __future__ import annotations

from pathlib import Path

from src.tools.base import BaseTool, ToolCall, ToolResult
from src.tools.cloud_image_gen import CloudImageGenTool
from src.core.config import config
from src.core.logging import get_logger

log = get_logger("ImageGenTool")


class ComfyUIImageGenTool(BaseTool):
    """
    图像生成工具 — 委托 CloudImageGenTool 执行多 Provider 回退链:

      comfyui → runpod → replicate → modal → mock

    向后兼容: 保持 image_gen 工具名不变, 内部自动切换 Provider。
    """

    def __init__(self):
        super().__init__("image_gen")
        self._cloud_tool: CloudImageGenTool | None = None

    @property
    def cloud_tool(self) -> CloudImageGenTool:
        if self._cloud_tool is None:
            self._cloud_tool = CloudImageGenTool()
        return self._cloud_tool

    def schema(self) -> dict:
        return {
            "name": "image_gen",
            "description": "Generate an image via multi-provider fallback (ComfyUI / RunPod / Replicate / Modal / Mock) with LoRA support",
            "parameters": {
                "prompt": {"type": "string"},
                "negative_prompt": {"type": "string", "default": ""},
                "width": {"type": "integer", "default": config.image_gen.default_width},
                "height": {"type": "integer", "default": config.image_gen.default_height},
                "steps": {"type": "integer", "default": config.image_gen.default_steps},
                "cfg": {"type": "number", "default": config.image_gen.default_cfg},
                "seed": {"type": "integer", "default": -1},
                "batch_size": {"type": "integer", "default": config.image_gen.batch_size},
                "output_dir": {"type": "string", "default": "outputs/frames"},
                "filename": {"type": "string", "default": ""},
                "lora_path": {"type": "string", "default": ""},
                "trigger_word": {"type": "string", "default": ""},
            },
        }

    async def execute(self, call: ToolCall) -> ToolResult:
        return await self.cloud_tool.execute(call)

    async def close(self):
        if self._cloud_tool:
            await self._cloud_tool.close()


class SDXLDirectTool(BaseTool):
    """
    直接 SDXL 调用 (无 ComfyUI, 本地 diffusers)

    当不需要 ComfyUI 的节点化工作流时使用，直接加载 diffusers pipeline。
    需要: pip install diffusers accelerate
    """

    def __init__(self):
        super().__init__("sdxl_direct")
        self._pipeline = None

    def schema(self) -> dict:
        return {
            "name": "sdxl_direct",
            "description": "Generate image directly via diffusers SDXL pipeline",
            "parameters": {
                "prompt": {"type": "string"},
                "negative_prompt": {"type": "string", "default": ""},
                "width": {"type": "integer", "default": 1024},
                "height": {"type": "integer", "default": 1024},
                "steps": {"type": "integer", "default": 30},
                "seed": {"type": "integer", "default": -1},
            },
        }

    async def execute(self, call: ToolCall) -> ToolResult:
        try:
            from diffusers import DiffusionPipeline
            import torch

            if self._pipeline is None:
                model_id = config.image_gen.sdxl_model
                self._pipeline = DiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                )
                self._pipeline.to("cuda" if torch.cuda.is_available() else "cpu")

            p = call.params
            prompt = p.get("prompt", "")
            negative = p.get("negative_prompt", "")
            width = p.get("width", 1024)
            height = p.get("height", 1024)
            steps = p.get("steps", 30)
            seed = p.get("seed", -1)

            generator = None
            if seed != -1:
                import torch
                generator = torch.Generator(device=self._pipeline.device).manual_seed(seed)

            image = self._pipeline(
                prompt=prompt,
                negative_prompt=negative or "low quality, blurry",
                width=width,
                height=height,
                num_inference_steps=steps,
                generator=generator,
            ).images[0]

            import uuid
            path = Path(f"outputs/frames/{uuid.uuid4().hex[:8]}.png")
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(str(path))

            return ToolResult.ok(data={"images": [str(path)], "gen_method": "sdxl_direct"})
        except ImportError:
            return ToolResult.fail("diffusers 未安装。pip install diffusers accelerate")
        except Exception as e:
            return ToolResult.fail(str(e))
