/**
 * MiniMax 原生生图（image-01）适配层。
 *
 * 背景：小手机（momo.float.phone）默认按 OpenAI 兼容格式调用生图接口
 *   POST {baseUrl}/images/generations  或  POST {baseUrl}/images/edits
 * 但 MiniMax 开放平台**没有** OpenAI 兼容的生图端点（实测 POST /v1/images/generations 返回 404），
 * 它的原生端点是：
 *   POST https://api.minimax.cn/v1/image_generation
 * 请求与响应格式也完全不同：
 *   - 参考图走 subject_reference[].image_file（不是 multipart 的 image 字段）
 *   - 响应里 data 是**对象** {image_urls[], image_base64[]}（不是 OpenAI 的数组）
 *   - 出错时 HTTP 仍然是 200，真正的错误码藏在 base_resp.status_code 里
 *
 * 本模块负责把前者翻译成后者，并把结果翻译回应用内部统一使用的 { b64, mimeType }。
 *
 * 刻意不 import 任何东西（纯函数），原因有两个：
 *   1. 浏览器直连（Key 不出浏览器）与 Netlify 函数服务端转发共用同一份翻译逻辑，避免两边行为漂移；
 *   2. 纯函数可以直接用 node 跑离线测试（node 24 原生支持 .ts），不依赖网络与真实 Key。
 */

export const MINIMAX_DEFAULT_MODEL = "image-01";
export const MINIMAX_IMAGE_PATH = "/image_generation";
export const MINIMAX_PROMPT_MAX_LENGTH = 1500;
export const MINIMAX_DEFAULT_BASE_URL = "https://api.minimax.cn/v1";

/** 官方支持的比例。width/height 只有 image-01 能用，image-01-live 只能靠比例。 */
const MINIMAX_ASPECT_RATIOS = [
  { ratio: "1:1", width: 1024, height: 1024 },
  { ratio: "16:9", width: 1280, height: 720 },
  { ratio: "4:3", width: 1152, height: 864 },
  { ratio: "3:2", width: 1248, height: 832 },
  { ratio: "2:3", width: 832, height: 1248 },
  { ratio: "3:4", width: 864, height: 1152 },
  { ratio: "9:16", width: 720, height: 1280 },
  { ratio: "21:9", width: 1344, height: 576 },
];

/** base_resp.status_code → 可读原因 + 对外 HTTP 状态码（与本地反代 app.py 保持一致）。 */
const MINIMAX_ERROR_CODES: Record<number, { reason: string; httpStatus: number }> = {
  1002: { reason: "触发限流，请稍后再试", httpStatus: 429 },
  1004: { reason: "账号鉴权失败，请检查 API Key 是否填写正确", httpStatus: 401 },
  1008: { reason: "账号余额不足", httpStatus: 402 },
  1026: { reason: "提示词涉及敏感内容，已被内容安全拦截", httpStatus: 400 },
  2013: { reason: "传入参数异常，请检查入参是否按要求填写", httpStatus: 400 },
  2049: { reason: "无效的 API Key", httpStatus: 401 },
};

export type MiniMaxSizeParams = {
  aspect_ratio?: string;
  width?: number;
  height?: number;
};

export type MiniMaxImageRequestInput = {
  model?: string | null;
  prompt: string;
  size?: string | null;
  referenceImageDataUrl?: string | null;
  responseFormat?: "url" | "base64";
  n?: number;
  promptOptimizer?: boolean;
};

export type MiniMaxParseResult = {
  ok: boolean;
  error?: string;
  httpStatus?: number;
  b64?: string;
  url?: string;
  mimeType?: string;
};

/** 从任意 baseUrl 里取出 host（小写、不含端口与 userinfo）。 */
function extractHost(raw: string): string {
  const withScheme = /^[a-z][a-z0-9+.-]*:\/\/([^/?#]+)/i.exec(raw);
  const authority = withScheme ? withScheme[1] : raw.split(/[/?#]/)[0] || "";
  return authority.replace(/^[^@]*@/, "").split(":")[0].trim().toLowerCase();
}

/**
 * 判断一个 baseUrl 是不是 MiniMax。
 * 匹配 api.minimax.cn / api.minimaxi.com / api.minimax.io 等官方域名；
 * 路径里出现 minimax 不算（只看 host），避免误判。
 */
export function isMiniMaxBaseUrl(baseUrl: string | null | undefined): boolean {
  const raw = (baseUrl || "").trim();
  if (!raw) return false;
  return /(^|\.)(minimax|minimaxi)(\.|$)/i.test(extractHost(raw));
}

/** 把用户填的 baseUrl 拼成 MiniMax 原生生图端点。 */
export function buildMiniMaxImageUrl(baseUrl: string | null | undefined): string {
  const trimmed = (baseUrl || "").trim().replace(/\/+$/, "");
  const cleaned = trimmed
    .replace(/\/image_generation$/i, "")
    .replace(/\/images\/(?:generations|edits)$/i, "")
    .replace(/\/images$/i, "");
  return `${cleaned || MINIMAX_DEFAULT_BASE_URL}${MINIMAX_IMAGE_PATH}`;
}

function isValidDimension(value: number): boolean {
  return Number.isFinite(value) && value >= 512 && value <= 2048 && value % 8 === 0;
}

/** 用交叉相乘找最接近的比例，避免浮点误差。 */
function findAspectRatio(width: number, height: number): string | null {
  for (const item of MINIMAX_ASPECT_RATIOS) {
    const parts = item.ratio.split(":");
    const ratioWidth = Number.parseInt(parts[0], 10);
    const ratioHeight = Number.parseInt(parts[1], 10);
    if (width * ratioHeight === height * ratioWidth) return item.ratio;
  }
  return null;
}

/**
 * 把「OpenAI 风格的尺寸」翻译成 MiniMax 的尺寸参数。
 *
 * 官方规则：width/height 必须成对出现，取值 [512,2048] 且是 8 的倍数，
 * 并且**与 aspect_ratio 同时存在时以 aspect_ratio 优先**；width/height 仅 image-01 生效。
 *
 * 所以策略是：
 *   - auto / 空 → 两个都不给，用官方默认（1:1）
 *   - 合法尺寸   → 只给 width/height（精确命中用户选的像素）
 *   - 非法尺寸   → 退化成 aspect_ratio（若比例恰好受支持），再不行就交给官方默认
 */
export function mapMiniMaxSize(
  size: string | null | undefined,
  model?: string | null,
): MiniMaxSizeParams {
  const raw = (size || "").trim().toLowerCase();
  if (!raw || raw === "auto") return {};

  const matched = /^(\d{2,5})\s*[x×*]\s*(\d{2,5})$/.exec(raw);
  if (!matched) return {};

  const width = Number.parseInt(matched[1], 10);
  const height = Number.parseInt(matched[2], 10);
  if (!Number.isFinite(width) || !Number.isFinite(height)) return {};

  const modelName = (model || "").trim();
  const widthHeightSupported = modelName !== "image-01-live";
  if (widthHeightSupported && isValidDimension(width) && isValidDimension(height)) {
    return { width, height };
  }

  const ratio = findAspectRatio(width, height);
  return ratio ? { aspect_ratio: ratio } : {};
}

/**
 * 官方限制 prompt 长度，实测报错原文是 "prompt length must be less than 1500"
 * ——是**严格小于** 1500（发 1501 和 1600 都返回 2013），不是「不超过 1500」。
 * 小手机会把角色描述、锚点、用户身份拼在一起，很容易超长，所以这里主动截断。
 */
export function truncateMiniMaxPrompt(prompt: string | null | undefined): string {
  const text = (prompt || "").trim();
  if (text.length < MINIMAX_PROMPT_MAX_LENGTH) return text;
  // 留一个字符的余量：必须是 < 1500，截到 1500 会被拒。
  const clipped = text.slice(0, MINIMAX_PROMPT_MAX_LENGTH - 1);
  const boundary = clipped.lastIndexOf("\n\n");
  // 末尾附近正好有段落边界就断在那里，避免把提示词腰斩成半句话
  if (boundary >= MINIMAX_PROMPT_MAX_LENGTH - 200) return clipped.slice(0, boundary).trim();
  return clipped.trim();
}

/** 参考图只接受公网 URL 或 Data URL（JPG/JPEG/PNG，<10MB）。 */
export function normalizeMiniMaxReferenceImage(value: string | null | undefined): string | null {
  const raw = (value || "").trim();
  if (!raw) return null;
  if (/^https?:\/\//i.test(raw)) return raw;
  if (/^data:image\/(png|jpe?g);base64,/i.test(raw)) return raw;
  return null;
}

/** 构造 MiniMax 原生生图请求体。 */
export function buildMiniMaxImageRequest(input: MiniMaxImageRequestInput): Record<string, unknown> {
  const model = (input.model || "").trim() || MINIMAX_DEFAULT_MODEL;
  const body: Record<string, unknown> = {
    model,
    prompt: truncateMiniMaxPrompt(input.prompt),
    response_format: input.responseFormat === "url" ? "url" : "base64",
    n: Math.min(9, Math.max(1, Math.round(input.n ?? 1))),
    prompt_optimizer: input.promptOptimizer === true,
  };

  const size = mapMiniMaxSize(input.size, model);
  if (size.aspect_ratio) body.aspect_ratio = size.aspect_ratio;
  if (size.width) body.width = size.width;
  if (size.height) body.height = size.height;

  const reference = normalizeMiniMaxReferenceImage(input.referenceImageDataUrl);
  if (reference) {
    body.subject_reference = [{ type: "character", image_file: reference }];
  }

  return body;
}

function stripDataUrlPrefix(value: string): string {
  const match = /^data:[^;,]*;base64,(.*)$/i.exec(value.trim());
  return match ? match[1] : value.trim();
}

/**
 * 从 base64 的开头几个字符猜出真实图片格式。
 *
 * 为什么需要：MiniMax 的 image-01 实际返回的是 **JPEG**，不是 PNG（已用真实调用验证，
 * 文件头是 FF D8 FF E0 / JFIF）。如果按 PNG 写死 MIME，小手机就会用
 * `new Blob([jpegBytes], {type:"image/png"})` 建 Blob，存下来的 MIME 与字节对不上，
 * 图床上传、canvas 绘制这类场景会出问题。
 *
 * 各格式 base64 的固定前缀：
 *   JPEG  FF D8 FF        -> "/9j/"
 *   PNG   89 50 4E 47     -> "iVBORw0KGgo"
 *   GIF   47 49 46 38     -> "R0lGOD"
 *   WEBP  52 49 46 46     -> "UklGR"
 */
const BASE64_IMAGE_MAGIC: Array<{ prefix: string; mimeType: string }> = [
  { prefix: "/9j/", mimeType: "image/jpeg" },
  { prefix: "iVBORw0KGgo", mimeType: "image/png" },
  { prefix: "R0lGOD", mimeType: "image/gif" },
  { prefix: "UklGR", mimeType: "image/webp" },
];

export function detectImageMimeType(base64: string | null | undefined): string {
  const raw = (base64 || "").trim();
  if (!raw) return "image/png";
  for (const item of BASE64_IMAGE_MAGIC) {
    if (raw.startsWith(item.prefix)) return item.mimeType;
  }
  return "image/png";
}

function firstNonEmptyString(values: unknown): string | null {
  if (!Array.isArray(values)) return null;
  for (const item of values) {
    if (typeof item === "string" && item.trim()) return item.trim();
  }
  return null;
}

/**
 * 解析 MiniMax 原生响应。
 *
 * 注意两点坑：
 *   1. 出错时 HTTP 依然是 200，错误在 base_resp.status_code 里，必须先查它；
 *   2. data 是对象 {image_urls[], image_base64[]}，不是 OpenAI 的数组。
 */
export function parseMiniMaxImageResponse(json: unknown): MiniMaxParseResult {
  if (!json || typeof json !== "object") {
    return { ok: false, error: "MiniMax 返回了无法解析的响应", httpStatus: 502 };
  }

  const record = json as Record<string, unknown>;
  const baseResp = (record.base_resp || {}) as Record<string, unknown>;
  const statusCode = typeof baseResp.status_code === "number" ? baseResp.status_code : undefined;
  const statusMsg = typeof baseResp.status_msg === "string" ? baseResp.status_msg : "";

  if (statusCode !== undefined && statusCode !== 0) {
    const known = MINIMAX_ERROR_CODES[statusCode];
    const reason = known ? known.reason : statusMsg || "未知错误";
    return {
      ok: false,
      error: `MiniMax 生图失败（${statusCode}）：${reason}${statusMsg && !known ? ` - ${statusMsg}` : ""}`,
      httpStatus: known ? known.httpStatus : 502,
    };
  }

  const data = record.data;
  const metadata = (record.metadata || {}) as Record<string, unknown>;
  const successCount = typeof metadata.success_count === "number" ? metadata.success_count : undefined;

  // 正常形态：data 是对象
  if (data && typeof data === "object" && !Array.isArray(data)) {
    const dataRecord = data as Record<string, unknown>;
    const base64 = firstNonEmptyString(dataRecord.image_base64);
    if (base64) {
      const raw = stripDataUrlPrefix(base64);
      return { ok: true, b64: raw, mimeType: detectImageMimeType(raw) };
    }
    const url = firstNonEmptyString(dataRecord.image_urls);
    if (url) {
      return { ok: true, url, mimeType: "image/jpeg" };
    }
  }

  // 兼容形态：data 是数组（部分中转会把 MiniMax 响应转成 OpenAI 形状）
  if (Array.isArray(data)) {
    for (const item of data) {
      if (!item || typeof item !== "object") continue;
      const itemRecord = item as Record<string, unknown>;
      const b64 = itemRecord.b64_json || itemRecord.base64 || itemRecord.b64;
      if (typeof b64 === "string" && b64.trim()) {
        const raw = stripDataUrlPrefix(b64);
        return { ok: true, b64: raw, mimeType: detectImageMimeType(raw) };
      }
      const url = itemRecord.url || itemRecord.image_url;
      if (typeof url === "string" && url.trim()) {
        return { ok: true, url: url.trim(), mimeType: "image/jpeg" };
      }
    }
  }

  if (successCount === 0) {
    return { ok: false, error: "MiniMax 未返回图片（内容安全检查未通过）", httpStatus: 400 };
  }

  return { ok: false, error: "MiniMax 返回中没有图片数据", httpStatus: 502 };
}
