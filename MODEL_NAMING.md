# MFQ 模型命名规范

公开模型文件统一由原模型名直接追加量化后缀：

```text
<BaseModel>-<Family><Bits>-<Grade>.mfq
```

`BaseModel` 必须保留原模型名称；`<Family><Bits>-<Grade>` 是量化后缀。
文件扩展名 `.mfq` 已明确表示 MFQ 格式，文件名中不再重复加入 `-MFQ-`。

例如：

```text
Qwen3.5-9B-V3-XXS.mfq
Qwen3.5-9B-S4-L.mfq
Qwen3.5-9B-P1-L.mfq
```

## Family

| MFQ family | 量化类型 | llama.cpp 对照 |
|---|---|---|
| `P` | Product Quantization，乘积量化 | 低于 2 bit 的 PQ 档位 |
| `V` | Vector Quantization，向量量化 | `IQ*` |
| `S` | Scalar Quantization，标量量化 | `Q*_K*` |

主位宽低于 2 bit 的型号统一使用 `P`，例如 `P1-L`；2 bit 及以上继续按实际量化
家族使用 `V` 或 `S`。UD 基线保留其原始名称，例如
`UD-IQ3_XXS`、`UD-Q4_K_XL`；MFQ 文件名不得继续使用 `UD-*` 作为自身型号。

## Bits 与 Grade

- `Bits` 是该公开档位的主位宽，例如 `V3`、`S4`。
- `Grade` 是 MFQ 发布档位，例如 `XXS`、`M`、`L`。
- `IQ` 对标档沿用可辨识的 grade：`IQ3_XXS` 对应 `V3-XXS`。
- `Q_K` 对标档使用已注册的 MFQ grade；当前 `Q*_K_XL` 对标模型均为 `S*-L`。
- 新增档位必须先加入 `mfq/model_naming.py` 的注册表，不能依据 UD 文件名临时拼接。

## 当前注册表

| UD recipe | MFQ 型号 | Qwen3.5-9B 正式文件名 |
|---|---|---|
| `IQ2_M` | `V2-M` | `Qwen3.5-9B-V2-M.mfq` |
| `IQ2_XXS` | `V2-XXS` | `Qwen3.5-9B-V2-XXS.mfq` |
| `IQ3_S` | `V3-S` | `Qwen3.5-9B-V3-S.mfq` |
| `IQ3_XXS` | `V3-XXS` | `Qwen3.5-9B-V3-XXS.mfq` |
| `IQ4_NL` | `V4-NL` | `Qwen3.5-9B-V4-NL.mfq` |
| `IQ4_XS` | `V4-XS` | `Qwen3.5-9B-V4-XS.mfq` |
| `Q2_K_XL` | `S2-L` | `Qwen3.5-9B-S2-L.mfq` |
| `Q3_K_M` | `S3-M` | `Qwen3.5-9B-S3-M.mfq` |
| `Q3_K_XL` | `S3-L` | `Qwen3.5-9B-S3-L.mfq` |
| `Q4_K_M` | `S4-M` | `Qwen3.5-9B-S4-M.mfq` |
| `Q4_K_S` | `S4-S` | `Qwen3.5-9B-S4-S.mfq` |
| `Q4_K_XL` | `S4-L` | `Qwen3.5-9B-S4-L.mfq` |
| `Q5_K_M` | `S5-M` | `Qwen3.5-9B-S5-M.mfq` |
| `Q5_K_S` | `S5-S` | `Qwen3.5-9B-S5-S.mfq` |
| `Q5_K_XL` | `S5-L` | `Qwen3.5-9B-S5-L.mfq` |
| `Q6_K` | `S6` | `Qwen3.5-9B-S6.mfq` |
| `Q6_K_XL` | `S6-L` | `Qwen3.5-9B-S6-L.mfq` |
| `Q8_K_XL` | `S8-L` | `Qwen3.5-9B-S8-L.mfq` |

## 发布名与实验名

正式文件名只表达基础模型和公开档位。`NINT8-0`、`IMATRIX`、`REFWEIGHT`、
`F32ACC`、日期等实现与实验信息写入模型元数据、run contract 和结果清单。

尚未成为正式版本的实验文件使用：

```text
<CanonicalStem>-EXP-<slug>-<YYYYMMDD>.mfq
```

不得用实验文件覆盖同档位的正式模型。
