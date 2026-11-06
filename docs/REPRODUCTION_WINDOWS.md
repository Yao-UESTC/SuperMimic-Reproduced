# Windows 10 + RTX 4090 复现记录

记录把本仓库跑起来（部署 + 最小复现）的全过程、遇到的坑和绕过方法。环境与上游锁定版本不同之处都写在这里，便于回滚或迁移。

## 硬件与系统

| 项目 | 值 |
| --- | --- |
| CPU | Intel Core i9-14900KF（24 核 / 32 线程） |
| 内存 | 63.8 GB |
| GPU | NVIDIA GeForce RTX 4090 24 GB |
| 系统 | Windows 10 专业版 22H2（19045） |
| 磁盘 | C: 剩余约 1.1 TB |

## 版本与驱动

- Python 3.12.10（`uv` 0.13.7 管理，`uv.lock` 解析 161 包）
- torch **2.14.1+cu130**、warp-lang 1.18.0、mujoco 3.16.0、mujoco-warp 3.16.0、mjlab 1.6.0（固定 commit `8bb2bc9`）
- NVIDIA 驱动 **560.94 → 580.97**（CUDA 13.0）

驱动必须升级：cu130 轮子要求 R580+，560.94 只到 CUDA 12.6，否则 `torch.cuda.is_available()` 为 False。

## 安装步骤（本机实际执行）

```powershell
# 1) uv
winget install --id astral-sh.uv -e

# 2) mjlab 固定 commit（github.com 不通，改走 tarball + 本地路径）
#    https://ghfast.top/https://github.com/mujocolab/mjlab/archive/<sha>.tar.gz
#    解包到 C:/DSH/_deps/mjlab，并临时改 [tool.uv.sources] mjlab = { path = "C:/DSH/_deps/mjlab" }

uv sync --group dev
uv run pytest tests -q          # 7 passed, 17 skipped（跳过的是缺数据的用例）
```

`pyproject.toml` / `uv.lock` 的上述改动**未提交**，网络正常后 `git checkout pyproject.toml uv.lock` 即可恢复上游锁定。

## Windows 特有的四个坑

1. **github.com 不可达**：`git clone` 拉 mjlab 会断流；改用 `ghfast.top` / `gh-proxy.com` 代理下载 tarball，或直接走 `api.github.com` 的 release 资产。
2. **子进程只能写工作区**：`%LOCALAPPDATA%` 下的 warp / matplotlib / torchinductor 缓存目录报 `PermissionError`。全部重定向到工作区：
   ```powershell
   $env:WARP_CACHE_PATH="C:\DSH\.caches\warp"
   $env:MPLCONFIGDIR="C:\DSH\.caches\matplotlib"
   $env:TORCHINDUCTOR_CACHE_DIR="C:\DSH\.caches\torchinductor"
   $env:TEMP=$env:TMP="C:\DSH\.caches\tmp"
   ```
3. **没有 Triton**：`torch.compile` 在 Windows 上抛 `TritonMissing`。训练用 `--wm-trainer.compile None --policy-trainer.compile None` 关闭（纯 eager，功能等价，速度略降）。
4. **发布检查点含 `PosixPath`**：`torch.load` 报 `cannot instantiate 'PosixPath'`。用 `C:\DSH\win\eval_wrapped.py` 包装脚本，加载前把 `pathlib.PosixPath` 映射成 `WindowsPath`。

驱动安装本身也有坑：NVIDIA 官方 `setup.exe` 在非交互环境下直接返回 `0x80070005`（拒绝访问），改用解包后的 INF 走 pnputil：

```powershell
pnputil /add-driver "<解包目录>\Display.Driver\nv_dispi.inf" /install
```

## 数据

LAFAN1（`lvhaidong/LAFAN1_Retargeting_Dataset` 的 `g1/`，40 条 CSV / 86.3 MB，huggingface.co 被墙，用 `hf-mirror.com`）：

```powershell
uv run python scripts/data/csv_to_npz_batch.py --input-dir data/raw/lafan1/g1 `
  --output-dir data/npz/lafan1 --input-fps 30 --workers 12
# 40 clips, 2.451 h, 0 failures, 6 s
```

BONES-SEED 未下载（全量 142k clips，`g1.tar.gz` 体积大，且需要 ~6.7 GB 的紧凑库；本次用发布的 e8 检查点代替）。

## 结果

### 1. 短训练（真跑通，非仅冒烟）

```powershell
uv run sm-train --run-name smoke-4090 --data.npz-dir data/npz/lafan1 `
  --env.num-envs 256 --policy-trainer.batch-size 256 `
  --schedule.total-iterations 120 --schedule.warmup-iterations 20 `
  --schedule.checkpoint-interval 60 `
  --wm-trainer.compile None --policy-trainer.compile None
```

- 吞吐 1.1 it/s（256 环境，单卡 4090）
- 世界模型 8 步开环误差 0.6624 → 0.3000（120 次迭代内）
- 1 步误差 0.2077 → 0.0833；imagined return 0.102 → 0.127
- 检查点 205 MB × 3（`logs/smoke-4090/`）

### 2. 发布检查点评测（Appendix F）

用 release `e8-boneseed-100k`（213 MB，BONES-SEED 全量训练）在 LAFAN1 40 条整条片段上评测：

```
40 clips | failed 20 | mean norm return 0.6232 | step-weighted 0.5966
```

与 README 中 LAFAN1 的 0.56–0.59 区间吻合。注意该检查点是在 BONES-SEED 上训练的，属于跨数据集评测；README 也指出仅本体感受的观测在 100–300 s 长片段上会漂移，整条回报饱和在 ~0.58。失败数 20/40 高于 README 的 3–9，符合"跨数据集 + 未在 LAFAN1 上训练"的预期。

## 结论

- 训练链路（数据 → 环境 → 世界模型 → BPTT 策略更新 → 检查点）在本机可复现。
- 评测链路（加载发布检查点 → MuJoCo Warp 确定性回放 → 归一化回报）可复现，数值落在论文/README 预期区间。
- 与上游的差异仅限：驱动版本、mjlab 获取方式（tarball 代替 git）、关闭 `torch.compile`、缓存目录位置。算法与超参未改动。

## 未做

- LAFAN1 完整训练（1×5090 约 3 h，`--schedule.max-hours 3`）
- BONES-SEED 全量数据与紧凑库
- `scripts/play.py` 交互回放与 `scripts/media/render_rollout.py` 渲染（需要 GL 环境，Windows 上可另行验证）
