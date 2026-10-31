import type { ImageGenerationSettings, NovelAiPreset } from "./settings-types";
import { loadBindingConfig, loadImageGenerationSettings, loadUserIdentities, resolveBinding, DEFAULT_NOVELAI_PRESET } from "./settings-storage";
import { applyImageGenerationBinding } from "./image-generation-binding";
import { loadChatScope } from "./chat-scope-storage";
import JSZip from "jszip";
import { getChatImageFromIndexedDB } from "./chat-asset-storage";
import { storeMediaBlob } from "./media-cache-storage";
import { throwIfAborted } from "./abort-utils";
import { descriptionNeedsUserReference } from "./image-reference-selection";
import {
  NOVELAI_COMMON_MODELS,
  getNovelAiResolution,
  normalizeNovelAiModel,
  normalizeNovelAiNoiseSchedule,
  normalizeNovelAiSampler,
  normalizeNovelAiScale,
  normalizeNovelAiSteps,
} from "./novelai-image-config";
import {
  buildMiniMaxImageRequest,
  buildMiniMaxImageUrl,
  detectImageMimeType,
  isMiniMaxBaseUrl,
  parseMiniMaxImageResponse,
} from "./minimax-image";

export type ImageGenerationResult = {
  mediaRef: string;
  dataUrl: string;
  blob: Blob;
  mimeType: string;
  prompt: string;
  usedReferenceImage: boolean;
  revisedPrompt?: string;
};

type ExtractedImage =
  | { kind: "b64"; b64: string; mimeType?: string; revisedPrompt?: string }
  | { kind: "url"; url: string; revisedPrompt?: string };

type ImageGenerationApiResponse = {
  b64: string;
  mimeType?: string;
  revisedPrompt?: string;
};

const IMAGE_MODEL_HINTS = [
  "image",
  "img",
  "dall",
  "flux",
  "stable",
  "sd",
  "midjourney",
  "mj",
  "ideogram",
  "imagen",
  "qwen-image",
  "kolors",
  "wan",
];

function mergePrompt(description: string, extraPrompt: string): string {
  const main = description.trim();
  const extra = extraPrompt.trim();
  return extra ? `${main}\n\n${extra}` : main;
}

function base64ToBlob(b64: string, mimeType: string): Blob {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return new Blob([bytes], { type: mimeType });
}

function dataUrlMimeType(dataUrl: string): string {
  const match = /^data:([^;]+);base64,/.exec(dataUrl);
  return match?.[1]?.toLowerCase() || "";
}

function dataUrlToBlob(dataUrl: string): { blob: Blob; mimeType: string } | null {
  const match = /^data:([^;]+);base64,([\s\S]+)$/.exec(dataUrl);
  if (!match) return null;
  return { blob: base64ToBlob(match[2], match[1] || "image/png"), mimeType: match[1] || "image/png" };
}

function blobToDataUrl(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(reader.error);
    reader.onload = () => resolve(String(reader.result ?? ""));
    reader.readAsDataURL(blob);
  });
}

function loadDataUrlImage(dataUrl: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("参考图解码失败"));
    image.src = dataUrl;
  });
}

async function normalizeReferenceImageForEdit(
  dataUrl: string,
  crop?: { x: number; y: number; size: number },
): Promise<string> {
  if (!crop && dataUrlMimeType(dataUrl) === "image/png") return dataUrl;
  if (typeof document === "undefined") return dataUrl;

  try {
    const image = await loadDataUrlImage(dataUrl);
    const width = image.naturalWidth || image.width;
    const height = image.naturalHeight || image.height;
    if (!width || !height) return dataUrl;

    const canvas = document.createElement("canvas");
    const normalizedSize = crop ? Math.max(0.18, Math.min(1, crop.size)) : 1;
    const sourceSize = Math.max(1, Math.round(Math.min(width, height) * normalizedSize));
    const sourceX = crop
      ? Math.round(Math.max(0, Math.min(width - sourceSize, crop.x * width)))
      : 0;
    const sourceY = crop
      ? Math.round(Math.max(0, Math.min(height - sourceSize, crop.y * height)))
      : 0;
    canvas.width = crop ? sourceSize : width;
    canvas.height = crop ? sourceSize : height;
    const context = canvas.getContext("2d");
    if (!context) return dataUrl;
    if (crop) {
      context.drawImage(image, sourceX, sourceY, sourceSize, sourceSize, 0, 0, sourceSize, sourceSize);
    } else {
      context.drawImage(image, 0, 0, width, height);
    }
    return canvas.toDataURL("image/png");
  } catch {
    return dataUrl;
  }
}

async function combineReferenceImagesForEdit(
  references: Array<{ dataUrl: string; label: string }>,
): Promise<string | null> {
  if (references.length === 0) return null;
  if (references.length === 1 || typeof document === "undefined") return references[0].dataUrl;
  try {
    const items = await Promise.all(references.slice(0, 6).map(async reference => ({
      ...reference,
      image: await loadDataUrlImage(reference.dataUrl),
    })));
    const columns = Math.min(2, items.length);
    const rows = Math.ceil(items.length / columns);
    const cell = 512;
    const labelHeight = 46;
    const gap = 12;
    const canvas = document.createElement("canvas");
    canvas.width = columns * cell + (columns + 1) * gap;
    canvas.height = rows * (cell + labelHeight) + (rows + 1) * gap;
    const context = canvas.getContext("2d");
    if (!context) return references[0].dataUrl;
    context.fillStyle = "#f5f5f5";
    context.fillRect(0, 0, canvas.width, canvas.height);
    items.forEach((item, index) => {
      const column = index % columns;
      const row = Math.floor(index / columns);
      const x = gap + column * cell;
      const y = gap + row * (cell + labelHeight);
      context.fillStyle = "#ffffff";
      context.fillRect(x, y, cell, cell + labelHeight);
      const width = item.image.naturalWidth || item.image.width;
      const height = item.image.naturalHeight || item.image.height;
      const scale = Math.min(cell / width, cell / height);
      const drawWidth = width * scale;
      const drawHeight = height * scale;
      context.drawImage(item.image, x + (cell - drawWidth) / 2, y + (cell - drawHeight) / 2, drawWidth, drawHeight);
      context.fillStyle = "#111111";
      context.font = "600 24px sans-serif";
      context.textAlign = "center";
      context.textBaseline = "middle";
      context.fillText(item.label, x + cell / 2, y + cell + labelHeight / 2, cell - 24);
    });
    return canvas.toDataURL("image/png");
  } catch {
    return references[0].dataUrl;
  }
}

function resolveGenerationUserIdentityId(characterId?: string, appId?: string): string | null {
  const scopedIdentityId = appId && ["chat", "moments", "group_chat"].includes(appId)
    ? loadChatScope().userIdentityId
    : null;
  if (scopedIdentityId) return scopedIdentityId;
  const identities = loadUserIdentities();
  if (identities.length === 0) return null;
  const boundId = resolveBinding(loadBindingConfig(), characterId, appId).userIdentityId;
  return boundId && identities.some(identity => identity.id === boundId) ? boundId : identities[0].id;
}

function imageExtension(mimeType: string): string {
  const subtype = mimeType.split("/")[1] || "png";
  return subtype.replace("jpeg", "jpg");
}

function normalizeBaseUrl(baseUrl: string): string {
  return baseUrl
    .trim()
    .replace(/\/+$/, "")
    .replace(/\/images\/(?:generations|edits)$/i, "")
    .replace(/\/images$/i, "");
}

function buildImageUrl(baseUrl: string, mode: "generations" | "edits"): string {
  const trimmed = baseUrl.trim().replace(/\/+$/, "");
  if (/\/images\/(?:generations|edits)$/i.test(trimmed)) {
    return trimmed.replace(/\/images\/(?:generations|edits)$/i, `/images/${mode}`);
  }
  if (/\/images$/i.test(trimmed)) return `${trimmed}/${mode}`;
  return `${normalizeBaseUrl(trimmed)}/images/${mode}`;
}

function buildModelsUrl(baseUrl: string): string {
  const trimmed = baseUrl.trim().replace(/\/+$/, "");
  if (/\/models$/i.test(trimmed)) return trimmed;
  return `${normalizeBaseUrl(trimmed)}/models`;
}

function cleanBase64(value: string): { b64: string; mimeType?: string } {
  const match = /^data:([^;]+);base64,([\s\S]+)$/.exec(value.trim());
  if (match) return { mimeType: match[1], b64: match[2] };
  // 裸 base64：从魔术字节猜真实格式（很多中转返回的是 JPEG 而不是 PNG，
  // 写死 image/png 会让 Blob 的 MIME 与实际字节对不上）。
  const raw = value.trim();
  return { b64: raw, mimeType: detectImageMimeType(raw) };
}

function extractFromObject(data: unknown): ExtractedImage | null {
  if (!data || typeof data !== "object") return null;
  const record = data as Record<string, unknown>;
  const revisedPrompt = typeof record.revised_prompt === "string" ? record.revised_prompt : undefined;

  for (const key of ["b64_json", "base64", "b64", "image", "result"]) {
    const value = record[key];
    if (typeof value === "string" && value.trim()) {
      if (/^https?:\/\//i.test(value.trim())) return { kind: "url", url: value.trim(), revisedPrompt };
      const cleaned = cleanBase64(value);
      return { kind: "b64", ...cleaned, revisedPrompt };
    }
  }

  for (const key of ["url", "image_url"]) {
    const value = record[key];
    if (typeof value === "string" && /^https?:\/\//i.test(value.trim())) {
      return { kind: "url", url: value.trim(), revisedPrompt };
    }
    if (value && typeof value === "object") {
      const nested = (value as Record<string, unknown>).url;
      if (typeof nested === "string" && /^https?:\/\//i.test(nested.trim())) {
        return { kind: "url", url: nested.trim(), revisedPrompt };
      }
    }
  }

  for (const key of ["data", "images", "output", "content"]) {
    const value = record[key];
    if (Array.isArray(value)) {
      for (const item of value) {
        if (typeof item === "string" && item.trim()) {
          if (/^https?:\/\//i.test(item.trim())) return { kind: "url", url: item.trim(), revisedPrompt };
          const cleaned = cleanBase64(item);
          return { kind: "b64", ...cleaned, revisedPrompt };
        }
        const nested = extractFromObject(item);
        if (nested) return { ...nested, revisedPrompt: nested.revisedPrompt || revisedPrompt };
      }
    }
  }

  return null;
}

function extractModels(data: unknown): string[] {
  const results: string[] = [];
  const push = (value: unknown) => {
    if (typeof value !== "string") return;
    const normalized = value.replace(/^models\//, "").trim();
    if (normalized) results.push(normalized);
  };

  if (Array.isArray(data)) {
    data.forEach(item => {
      if (typeof item === "string") push(item);
      else if (item && typeof item === "object") {
        const row = item as Record<string, unknown>;
        push(row.id ?? row.name ?? row.model);
      }
    });
  } else if (data && typeof data === "object") {
    const row = data as Record<string, unknown>;
    for (const key of ["data", "models", "items"]) {
      const value = row[key];
      if (Array.isArray(value)) results.push(...extractModels(value));
    }
    push(row.id ?? row.name ?? row.model);
  }

  return Array.from(new Set(results));
}

async function fetchImageUrlAsBase64(url: string, signal?: AbortSignal): Promise<{ b64: string; mimeType: string }> {
  throwIfAborted(signal);
  const res = await fetch(url, { signal });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`图片 URL 下载失败 ${res.status}: ${text.slice(0, 160)}`);
  }
  const blob = await res.blob();
  const dataUrl = await blobToDataUrl(blob);
  const cleaned = cleanBase64(dataUrl);
  return { b64: cleaned.b64, mimeType: cleaned.mimeType || blob.type || "image/png" };
}

async function parseImageGenerationResponse(res: Response, signal?: AbortSignal): Promise<ImageGenerationApiResponse> {
  throwIfAborted(signal);
  const contentType = (res.headers.get("content-type") || "").toLowerCase();
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`生图 API 错误 ${res.status}: ${text.slice(0, 600)}`);
  }

  if (contentType.startsWith("image/")) {
    const blob = await res.blob();
    throwIfAborted(signal);
    const dataUrl = await blobToDataUrl(blob);
    const cleaned = cleanBase64(dataUrl);
    return { b64: cleaned.b64, mimeType: cleaned.mimeType || contentType };
  }

  const json = await res.json();
  throwIfAborted(signal);
  const extracted = extractFromObject(json);
  if (!extracted) {
    throw new Error(`生图 API 返回中没有找到图片字段：${JSON.stringify(Object.keys(json || {})).slice(0, 200)}`);
  }

  if (extracted.kind === "url") {
    const downloaded = await fetchImageUrlAsBase64(extracted.url, signal);
    return { ...downloaded, revisedPrompt: extracted.revisedPrompt };
  }

  return {
    b64: extracted.b64,
    mimeType: extracted.mimeType || "image/png",
    revisedPrompt: extracted.revisedPrompt,
  };
}

export function filterLikelyImageModels(models: string[]): string[] {
  const filtered = models.filter(model => {
    const lower = model.toLowerCase();
    return IMAGE_MODEL_HINTS.some(hint => lower.includes(hint));
  });
  return filtered.length > 0 ? filtered : models;
}

function ensureMiniMaxImageModels(baseUrl: string, models: string[]): string[] {
  // MiniMax 官方 /v1/models 只返回**对话**模型，不含 image-01 / image-01-live，
  // 但生图用的就是这两个模型。不补的话模型下拉框里根本选不到，只能手填。
  if (!isMiniMaxBaseUrl(baseUrl)) return models;
  const merged = [...models];
  for (const id of ["image-01", "image-01-live"]) {
    if (!merged.includes(id)) merged.push(id);
  }
  return merged;
}

export async function fetchImageGenerationModels(settings: Pick<ImageGenerationSettings, "apiKey" | "baseUrl" | "requestMode">): Promise<string[]> {
  if (settings.requestMode === "direct") {
    try {
      const res = await fetch(buildModelsUrl(settings.baseUrl), {
        method: "GET",
        headers: { Authorization: `Bearer ${settings.apiKey}` },
      });
      const text = await res.text();
      if (!res.ok) {
        throw new Error(`模型列表 API 错误 ${res.status}: ${text.slice(0, 400)}`);
      }
      return ensureMiniMaxImageModels(settings.baseUrl, extractModels(JSON.parse(text)));
    } catch (error) {
      if (error instanceof TypeError) {
        throw new Error("浏览器直连失败：该 API 可能未允许跨域请求。");
      }
      throw error;
    }
  }

  const res = await fetch("/api/image-generation/models", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      apiKey: settings.apiKey,
      baseUrl: settings.baseUrl,
    }),
  });
  const data = await res.json().catch(() => ({})) as { models?: string[]; error?: string };
  if (!res.ok || data.error) {
    throw new Error(data.error || `模型列表请求失败 ${res.status}`);
  }
  return ensureMiniMaxImageModels(settings.baseUrl, Array.isArray(data.models) ? data.models : []);
}

// 通用生图代理(Cloudflare Worker)。配置后,「服务端中转」模式改为:
// 浏览器 → 本代理(带CORS、等待无时长限制) → 用户自己的生图API,
// 不再经过 Netlify 函数(其流式响应有 60s 硬上限,慢生图必死且中转站照样计费)。
// 留空 = 关闭,沿用 Netlify 心跳流式路由。自部署请配置自己的代理地址。
export const IMAGE_GEN_PROXY_URL = (process.env.NEXT_PUBLIC_IMAGE_GEN_PROXY_URL || "").trim().replace(/\/+$/, "");

async function generateImageDirect(params: {
  settings: ImageGenerationSettings;
  prompt: string;
  referenceImageDataUrl: string | null;
  signal?: AbortSignal;
  /** 走通用代理:请求发往代理地址,真实上游放进 x-upstream-base-url 头 */
  proxyBaseUrl?: string;
}): Promise<ImageGenerationApiResponse> {
  const { settings, prompt, referenceImageDataUrl, signal, proxyBaseUrl } = params;
  throwIfAborted(signal);
  const hasReference = Boolean(referenceImageDataUrl);
  const url = buildImageUrl(proxyBaseUrl || settings.baseUrl, hasReference ? "edits" : "generations");
  const headers: Record<string, string> = { Authorization: `Bearer ${settings.apiKey}` };
  if (proxyBaseUrl) headers["x-upstream-base-url"] = normalizeBaseUrl(settings.baseUrl);
  let body: BodyInit;

  if (hasReference) {
    const converted = dataUrlToBlob(referenceImageDataUrl || "");
    if (!converted) throw new Error("参考图格式无效");
    const form = new FormData();
    form.set("model", settings.model);
    form.set("prompt", prompt);
    if (settings.size && settings.size !== "auto") form.set("size", settings.size);
    if (settings.quality && settings.quality !== "auto") form.set("quality", settings.quality);
    form.append("image", converted.blob, `reference.${imageExtension(converted.mimeType)}`);
    body = form;
  } else {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify({
      model: settings.model,
      prompt,
      ...(settings.size && settings.size !== "auto" ? { size: settings.size } : {}),
      ...(settings.quality && settings.quality !== "auto" ? { quality: settings.quality } : {}),
    });
  }

  // 总超时 360s,外部 signal 联动;防止上游悬挂导致界面永久转圈。
  // 部分中转的按次生图（如 gpt-image 系）单张实测 3~5 分钟,180s 会在完成前掐断(钱照扣图丢失)。
  const controller = new AbortController();
  const onOuterAbort = () => controller.abort();
  if (signal) signal.addEventListener("abort", onOuterAbort, { once: true });
  const totalTimer = setTimeout(() => controller.abort(), 360_000);
  try {
    return await parseImageGenerationResponse(await fetch(url, { method: "POST", headers, body, signal: controller.signal }), signal);
  } catch (error) {
    if (controller.signal.aborted && !signal?.aborted) {
      throw new Error(proxyBaseUrl ? "生图代理超时（360 秒未返回）" : "生图请求超时（360 秒未返回）");
    }
    if (error instanceof TypeError) {
      throw new Error(proxyBaseUrl ? "生图代理连接失败" : "浏览器直连失败：该 API 可能未允许跨域请求。");
    }
    throw error;
  } finally {
    clearTimeout(totalTimer);
    if (signal) signal.removeEventListener("abort", onOuterAbort);
  }
}

// MiniMax 原生生图:浏览器直连 api.minimax.cn/v1/image_generation。
// 与 OpenAI 兼容接口的差异很大(端点、请求体、响应体、错误码位置都不同),
// 翻译逻辑集中在 lib/minimax-image.ts,这里只负责发请求和取图。
//
// 为什么优先直连:MiniMax 对任意 Origin 放行 CORS(已实测真实响应也带
// Access-Control-Allow-Origin),所以订阅 Key 可以只留在浏览器里,完全不经过
// Netlify 函数——既不多一个公网暴露面,也不消耗站点的函数额度。
async function generateImageMiniMaxDirect(params: {
  settings: ImageGenerationSettings;
  prompt: string;
  referenceImageDataUrl: string | null;
  signal?: AbortSignal;
}): Promise<ImageGenerationApiResponse> {
  const { settings, prompt, referenceImageDataUrl, signal } = params;
  throwIfAborted(signal);

  const url = buildMiniMaxImageUrl(settings.baseUrl);
  const body = buildMiniMaxImageRequest({
    model: settings.model,
    prompt,
    size: settings.size,
    referenceImageDataUrl,
  });

  const controller = new AbortController();
  const onOuterAbort = () => controller.abort();
  if (signal) signal.addEventListener("abort", onOuterAbort, { once: true });
  const totalTimer = setTimeout(() => controller.abort(), 360_000);
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${settings.apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    const text = await res.text();
    throwIfAborted(signal);
    if (!res.ok) {
      throw new Error(`MiniMax 生图 API 错误 ${res.status}: ${text.slice(0, 600)}`);
    }
    let json: unknown;
    try {
      json = JSON.parse(text);
    } catch {
      throw new Error(`MiniMax 返回的不是有效 JSON：${text.slice(0, 300)}`);
    }
    // 注意:MiniMax 出错时 HTTP 也是 200,错误码在 base_resp.status_code 里,
    // 这里由 parseMiniMaxImageResponse 统一识别。
    const parsed = parseMiniMaxImageResponse(json);
    if (!parsed.ok) throw new Error(parsed.error || "MiniMax 生图失败");
    if (parsed.b64) return { b64: parsed.b64, mimeType: parsed.mimeType || "image/png" };
    if (parsed.url) return await fetchImageUrlAsBase64(parsed.url, signal);
    throw new Error("MiniMax 返回中没有图片数据");
  } catch (error) {
    if (controller.signal.aborted && !signal?.aborted) {
      throw new Error("MiniMax 生图请求超时（360 秒未返回）");
    }
    if (error instanceof TypeError) {
      // 浏览器把跨域拦截、网络断开等都归为 TypeError,交给上层决定是否回落。
      throw new Error("浏览器直连失败：该 API 可能未允许跨域请求。");
    }
    throw error;
  } finally {
    clearTimeout(totalTimer);
    if (signal) signal.removeEventListener("abort", onOuterAbort);
  }
}

// MiniMax 专用入口:先直连(订阅 Key 不出浏览器),只有跨域被拦时才回落 Netlify
// 函数(该函数已打上 MiniMax 适配补丁,走的是同一套翻译逻辑)。
async function generateImageMiniMax(params: {
  settings: ImageGenerationSettings;
  prompt: string;
  referenceImageDataUrl: string | null;
  signal?: AbortSignal;
}): Promise<ImageGenerationApiResponse> {
  try {
    return await generateImageMiniMaxDirect(params);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    // 只有连接层失败(跨域/断网)才回落;API 自身的错误(密钥无效、余额不足、
    // 内容安全拦截)说明请求已经到达上游,换条路走也是同样的错,直接抛出避免重复计费。
    if (!message.includes("浏览器直连失败")) throw error;
    throwIfAborted(params.signal);
    return generateImageViaServer(params);
  }
}

// 「代理中转」模式:配置了通用代理(CF Worker)就只走它——用户选择什么模式就走什么链路,
// 不做隐藏回落(不再兜底到 Netlify 函数,那会消耗站点额度且有 60s 上限)。
// 常量未配置时保留旧的 Netlify 心跳流式路由(自部署无 Worker 的场景)。
//
// Netlify 心跳路由是站点额度的最大计算开销之一(每张图占用函数 30~120s),而不少
// 选了「服务端转发」的用户,其生图 API 其实允许跨域。因此走 Netlify 前先试一次
// 浏览器直连:直连因 CORS 失败(预检被拒,真实请求未发出,上游不会计费)才回落
// 到服务端,并按 baseUrl 记住失败结果,本次会话内不再重复探测。
const directCorsFailedBaseUrls = new Set<string>();

async function generateImageViaServerOrProxy(params: {
  settings: ImageGenerationSettings;
  prompt: string;
  referenceImageDataUrl: string | null;
  signal?: AbortSignal;
}): Promise<ImageGenerationApiResponse> {
  if (IMAGE_GEN_PROXY_URL) {
    try {
      return await generateImageDirect({ ...params, proxyBaseUrl: IMAGE_GEN_PROXY_URL });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      if (message.includes("生图代理连接失败")) {
        throw new Error("生图代理连接失败:当前网络可能无法访问代理服务器(部分地区需开启代理),或稍后重试。");
      }
      throw error;
    }
  }
  const baseUrlKey = normalizeBaseUrl(params.settings.baseUrl);
  if (!directCorsFailedBaseUrls.has(baseUrlKey)) {
    try {
      return await generateImageDirect(params);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      // 只有连接层失败(CORS/断网)才回落;API 自身的错误(密钥无效、余额不足等)
      // 说明直连是通的,换服务端转发也会得到同样错误,直接抛出避免重复计费。
      if (!message.includes("浏览器直连失败")) throw error;
      throwIfAborted(params.signal);
      directCorsFailedBaseUrls.add(baseUrlKey);
    }
  }
  return generateImageViaServer(params);
}

async function generateImageViaServer(params: {
  settings: ImageGenerationSettings;
  prompt: string;
  referenceImageDataUrl: string | null;
  signal?: AbortSignal;
}): Promise<ImageGenerationApiResponse> {
  const { settings, prompt, referenceImageDataUrl, signal } = params;
  throwIfAborted(signal);
  // 防"无限卡住":函数被平台中途击杀时流可能既不关闭也不报错。
  // 总超时 180s + 断流检测(心跳每 3s 一个字节,超过 25s 没有任何字节视为断流)。
  const controller = new AbortController();
  const onOuterAbort = () => controller.abort();
  if (signal) signal.addEventListener("abort", onOuterAbort, { once: true });
  const totalTimer = setTimeout(() => controller.abort(), 180_000);
  try {
    // x-stream-heartbeat:服务端以心跳流响应,真正的结果附在流末尾的 @@RESULT@@ 标记后。
    // 避免托管平台对缓冲响应的 10~26s 超时把慢生图(30~120s)掐成 504。
    const res = await fetch("/api/image-generation", {
      method: "POST",
      headers: { "Content-Type": "application/json", "x-stream-heartbeat": "1" },
      signal: controller.signal,
      body: JSON.stringify({
        apiKey: settings.apiKey,
        baseUrl: settings.baseUrl,
        model: settings.model,
        prompt,
        size: settings.size,
        quality: settings.quality,
        referenceImageDataUrl: referenceImageDataUrl || undefined,
      }),
    });
    throwIfAborted(signal);

    type ServerImagePayload = { httpStatus?: number; b64?: string; mimeType?: string; revisedPrompt?: string; error?: string };
    const contentType = (res.headers.get("content-type") || "").toLowerCase();
    let data: ServerImagePayload;
    if (contentType.includes("text/plain")) {
      let text = "";
      if (res.body) {
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let idleTimer: ReturnType<typeof setTimeout> | undefined;
        try {
          for (;;) {
            idleTimer = setTimeout(() => controller.abort(), 25_000);
            const { done, value } = await reader.read();
            clearTimeout(idleTimer);
            if (done) break;
            text += decoder.decode(value, { stream: true });
          }
          text += decoder.decode();
        } catch (err) {
          clearTimeout(idleTimer);
          if (controller.signal.aborted && !signal?.aborted) {
            throw new Error("生图请求失败（服务器连接中断,超过 25 秒没有响应）");
          }
          throw err;
        }
      } else {
        text = await res.text();
      }
      const marker = "@@RESULT@@";
      const idx = text.lastIndexOf(marker);
      if (idx < 0) throw new Error(`生图请求失败 ${res.status}（流式响应中断,未收到结果）`);
      try {
        data = JSON.parse(text.slice(idx + marker.length)) as ServerImagePayload;
      } catch {
        throw new Error("生图请求失败（流式结果解析出错）");
      }
      throwIfAborted(signal);
      if (data.error || !data.b64) {
        throw new Error(data.error || `生图请求失败 ${data.httpStatus ?? res.status}`);
      }
    } else {
      // 非流式回退(旧服务端等)
      data = await res.json().catch(() => ({})) as ServerImagePayload;
      throwIfAborted(signal);
      if (!res.ok || data.error || !data.b64) {
        throw new Error(data.error || `生图请求失败 ${res.status}`);
      }
    }
    return { b64: data.b64, mimeType: data.mimeType, revisedPrompt: data.revisedPrompt };
  } finally {
    clearTimeout(totalTimer);
    if (signal) signal.removeEventListener("abort", onOuterAbort);
  }
}

async function generateNovelAiDirect(params: {
  apiKey: string;
  customUrl?: string;
  preset: NovelAiPreset;
  prompt: string;
  signal?: AbortSignal;
}): Promise<ImageGenerationApiResponse> {
  const { apiKey, customUrl, preset, prompt, signal } = params;
  throwIfAborted(signal);

  const { width, height } = getNovelAiResolution(preset.resolution);

  const url = customUrl?.trim() || "https://image.novelai.net/ai/generate-image";
  const headers: Record<string, string> = {
    Authorization: `Bearer ${apiKey}`,
    "Content-Type": "application/json",
  };

  const body = JSON.stringify({
    input: prompt,
    model: normalizeNovelAiModel(preset.model),
    action: "generate",
    parameters: {
      width,
      height,
      scale: normalizeNovelAiScale(preset.scale),
      sampler: normalizeNovelAiSampler(preset.sampler),
      steps: normalizeNovelAiSteps(preset.steps),
      n_samples: 1,
      ucPreset: 0,
      qualityToggle: preset.qualityToggle !== false,
      sm: preset.smea === true,
      sm_dyn: preset.smeaDyn === true,
      dynamic_thresholding: false,
      controlnet_strength: 1,
      legacy: false,
      add_original_image: false,
      uncond_scale: 1,
      cfg_rescale: 0,
      noise_schedule: normalizeNovelAiNoiseSchedule(preset.noiseSchedule),
      negative_prompt: preset.negativePrompt || "",
    },
  });

  const controller = new AbortController();
  const onOuterAbort = () => controller.abort();
  if (signal) signal.addEventListener("abort", onOuterAbort, { once: true });
  const totalTimer = setTimeout(() => controller.abort(), 180_000);

  try {
    const res = await fetch(url, { method: "POST", headers, body, signal: controller.signal });
    if (!res.ok) {
      const errText = await res.text().catch(() => "");
      throw new Error(`NovelAI 接口报错 ${res.status}: ${errText.slice(0, 200)}`);
    }

    const arrayBuf = await res.arrayBuffer();
    const uint8 = new Uint8Array(arrayBuf);
    const isZip = (uint8[0] === 0x50 && uint8[1] === 0x4b && uint8[2] === 0x03 && uint8[3] === 0x04)
      || (res.headers.get("content-type") || "").toLowerCase().includes("zip");

    if (isZip) {
      const zip = await JSZip.loadAsync(arrayBuf);
      const files = Object.keys(zip.files);
      const pngFile = files.find(f => f.endsWith(".png")) || files[0];
      if (!pngFile) throw new Error("NovelAI 返回的压缩包内未找到图片");
      const b64 = await zip.files[pngFile].async("base64");
      return { b64, mimeType: "image/png" };
    }

    let binary = "";
    for (let i = 0; i < uint8.length; i++) {
      binary += String.fromCharCode(uint8[i]);
    }
    return { b64: btoa(binary), mimeType: res.headers.get("content-type") || "image/png" };
  } catch (error) {
    if (controller.signal.aborted && !signal?.aborted) {
      throw new Error("NovelAI 直连请求超时（180 秒未返回）");
    }
    if (error instanceof TypeError) {
      throw new Error("NovelAI 浏览器直连失败：可能受到浏览器跨域 (CORS) 限制，请在生图设置中将请求方式切换为「服务端转发」。");
    }
    throw error;
  } finally {
    clearTimeout(totalTimer);
    if (signal) signal.removeEventListener("abort", onOuterAbort);
  }
}

async function generateNovelAiViaServer(params: {
  apiKey: string;
  customUrl?: string;
  preset: NovelAiPreset;
  prompt: string;
  signal?: AbortSignal;
}): Promise<ImageGenerationApiResponse> {
  const { apiKey, customUrl, preset, prompt, signal } = params;
  throwIfAborted(signal);

  const controller = new AbortController();
  const onOuterAbort = () => controller.abort();
  if (signal) signal.addEventListener("abort", onOuterAbort, { once: true });
  const totalTimer = setTimeout(() => controller.abort(), 180_000);

  try {
    const resolution = getNovelAiResolution(preset.resolution);
    const res = await fetch("/api/image-generation", {
      method: "POST",
      headers: { "Content-Type": "application/json", "x-stream-heartbeat": "1" },
      signal: controller.signal,
      body: JSON.stringify({
        provider: "novelai",
        apiKey,
        baseUrl: customUrl?.trim() || undefined,
        model: normalizeNovelAiModel(preset.model),
        prompt,
        size: resolution.value,
        negativePrompt: preset.negativePrompt || "",
        steps: normalizeNovelAiSteps(preset.steps),
        scale: normalizeNovelAiScale(preset.scale),
        sampler: normalizeNovelAiSampler(preset.sampler),
        noiseSchedule: normalizeNovelAiNoiseSchedule(preset.noiseSchedule),
        qualityToggle: preset.qualityToggle,
        smea: preset.smea,
        smeaDyn: preset.smeaDyn,
      }),
    });
    throwIfAborted(signal);

    type ServerImagePayload = { httpStatus?: number; b64?: string; mimeType?: string; revisedPrompt?: string; error?: string };
    const contentType = (res.headers.get("content-type") || "").toLowerCase();
    let data: ServerImagePayload;

    if (contentType.includes("text/plain")) {
      let text = "";
      if (res.body) {
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let idleTimer: ReturnType<typeof setTimeout> | undefined;
        try {
          for (;;) {
            idleTimer = setTimeout(() => controller.abort(), 25_000);
            const { done, value } = await reader.read();
            clearTimeout(idleTimer);
            if (done) break;
            text += decoder.decode(value, { stream: true });
          }
          text += decoder.decode();
        } catch (err) {
          clearTimeout(idleTimer);
          if (controller.signal.aborted && !signal?.aborted) {
            throw new Error("NovelAI 请求失败（服务器连接中断，超过 25 秒没有响应）");
          }
          throw err;
        }
      } else {
        text = await res.text();
      }
      const marker = "@@RESULT@@";
      const idx = text.lastIndexOf(marker);
      if (idx < 0) throw new Error(`NovelAI 请求失败 ${res.status}（流式响应中断，未收到结果）`);
      try {
        data = JSON.parse(text.slice(idx + marker.length)) as ServerImagePayload;
      } catch {
        throw new Error("NovelAI 请求失败（流式结果解析出错）");
      }
      throwIfAborted(signal);
      if (data.error || !data.b64) {
        throw new Error(data.error || `NovelAI 请求失败 ${data.httpStatus ?? res.status}`);
      }
    } else {
      data = await res.json().catch(() => ({})) as ServerImagePayload;
      throwIfAborted(signal);
      if (!res.ok || data.error || !data.b64) {
        throw new Error(data.error || `NovelAI 请求失败 ${res.status}`);
      }
    }
    return { b64: data.b64, mimeType: data.mimeType, revisedPrompt: data.revisedPrompt };
  } finally {
    clearTimeout(totalTimer);
    if (signal) signal.removeEventListener("abort", onOuterAbort);
  }
}

export async function fetchNovelAiModels(apiKey: string): Promise<string[]> {
  const token = apiKey.trim();
  if (!token) throw new Error("请先填写 NovelAI API Token。");

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 15_000);
  try {
    const res = await fetch("https://image.novelai.net/user/data", {
      method: "GET",
      headers: { Authorization: `Bearer ${token}` },
      signal: controller.signal,
    });
    if (!res.ok) {
      const detail = await res.text().catch(() => "");
      if (res.status === 401 || res.status === 403) {
        throw new Error("NovelAI API Token 无效或已失效，请重新获取后再试。");
      }
      throw new Error(`NovelAI Token 验证失败 ${res.status}${detail ? `: ${detail.slice(0, 160)}` : ""}`);
    }
    return [...NOVELAI_COMMON_MODELS];
  } catch (error) {
    if (controller.signal.aborted) {
      throw new Error("NovelAI Token 验证超时，请检查网络后重试。");
    }
    if (error instanceof TypeError) {
      throw new Error("无法连接 NovelAI，请检查网络后重试。");
    }
    if (error instanceof Error) throw error;
    throw new Error("NovelAI Token 验证失败，请检查网络后重试。");
  } finally {
    clearTimeout(timeout);
  }
}

export async function generateImageFromConfiguredApi(params: {
  description: string;
  characterId?: string;
  /** 内容所属 APP；用于按“角色 > APP > 全局”解析生图方案。 */
  appId?: string;
  useReferenceImage?: boolean;
  /** 强制是否加入角色当前绑定用户的参考图；未传时根据合照/用户照片语义自动判断。 */
  includeUserReferenceImage?: boolean;
  settings?: ImageGenerationSettings;
  signal?: AbortSignal;
}): Promise<ImageGenerationResult | null> {
  const storedSettings = params.settings ?? loadImageGenerationSettings();
  // 显式传入 settings 的配置测试/工具调用保持原样；正常业务调用才读取绑定。
  const settings = params.settings
    ? storedSettings
    : applyImageGenerationBinding(
        storedSettings,
        resolveBinding(loadBindingConfig(), params.characterId, params.appId).imageConfigId,
      );
  if (!settings.enabled) return null;

  const description = params.description.trim();
  if (!description) return null;

  // NovelAI 模式
  if (settings.provider === "novelai") {
    const naiApiKey = settings.novelai?.apiKey?.trim();
    if (!naiApiKey) return null;

    const presets = settings.novelai?.presets && settings.novelai.presets.length > 0
      ? settings.novelai.presets
      : [DEFAULT_NOVELAI_PRESET];
    const activePreset = presets.find(p => p.id === settings.novelai?.activePresetId) || presets[0];

    // 本轮画面描述必须排在最前面。NovelAI 对提示词顺序较敏感；此前把预设、
    // 人物外观和锚点放在描述前面，长提示词下会让聊天/朋友圈的当前场景失去权重。
    const positiveParts: string[] = [description];
    const novelAiCharacterReference = params.characterId
      ? settings.characterReferences?.[params.characterId]
      : undefined;
    const characterFeaturePrompt = novelAiCharacterReference?.novelAiFeaturePromptEnabled !== false
      ? novelAiCharacterReference?.featurePrompt?.trim()
      : "";
    if (characterFeaturePrompt) positiveParts.push(characterFeaturePrompt);
    const anchorPrompt = (novelAiCharacterReference?.anchors || [])
      .filter(anchor => anchor.enabled !== false && anchor.description.trim())
      .map(anchor => `${anchor.name}: ${anchor.description.trim()}`)
      .join(", ");
    if (anchorPrompt) positiveParts.push(anchorPrompt);
    if (activePreset.positivePrompt?.trim()) positiveParts.push(activePreset.positivePrompt.trim());
    const fullPrompt = positiveParts.join(", ");

    const novelAiRequestMode = settings.novelai?.requestMode || settings.requestMode;
    const novelAiCustomUrl = settings.novelai?.customUrlEnabled
      ? settings.novelai.customUrl?.trim()
      : undefined;
    if (settings.novelai?.customUrlEnabled && !novelAiCustomUrl) {
      throw new Error("请填写 NovelAI 自定义 URL。");
    }
    const data = novelAiRequestMode === "direct"
      ? await generateNovelAiDirect({ apiKey: naiApiKey, customUrl: novelAiCustomUrl, preset: activePreset, prompt: fullPrompt, signal: params.signal })
      : await generateNovelAiViaServer({ apiKey: naiApiKey, customUrl: novelAiCustomUrl, preset: activePreset, prompt: fullPrompt, signal: params.signal });

    throwIfAborted(params.signal);
    const mimeType = data.mimeType || "image/png";
    const blob = base64ToBlob(data.b64, mimeType);
    throwIfAborted(params.signal);
    const mediaRef = await storeMediaBlob(blob, mimeType, "image");
    throwIfAborted(params.signal);
    return {
      mediaRef,
      dataUrl: `data:${mimeType};base64,${data.b64}`,
      blob,
      mimeType,
      prompt: fullPrompt,
      usedReferenceImage: false,
      revisedPrompt: data.revisedPrompt,
    };
  }

  // OpenAI 兼容模式：使用当前激活预设（旧配置由存储层迁移后仍保持兼容）
  const openaiPreset = settings.openaiPresets?.find(preset => preset.id === settings.activeOpenAiPresetId)
    || settings.openaiPresets?.[0];
  const openaiSettings = openaiPreset ? { ...settings, ...openaiPreset } : settings;
  // MiniMax 走的是原生协议（不是 OpenAI 兼容端点），模型名可以留空，默认 image-01。
  const isMiniMax = isMiniMaxBaseUrl(openaiSettings.baseUrl);
  if (!openaiSettings.apiKey.trim() || !openaiSettings.baseUrl.trim()) return null;
  if (!isMiniMax && !openaiSettings.model.trim()) return null;

  const reference = params.characterId ? settings.characterReferences?.[params.characterId] : undefined;
  // 「非自拍不使用参考图」开关已从生图设置页删除：参考图启用后始终参与生成
  const shouldUseReference = Boolean(
    params.useReferenceImage
    && reference?.assetId
    && reference.enabled !== false,
  );
  const referenceInputs: Array<{ dataUrl: string; label: string }> = [];
  if (shouldUseReference && reference?.assetId) {
    const raw = await getChatImageFromIndexedDB(reference.assetId);
    if (raw) referenceInputs.push({
      dataUrl: await normalizeReferenceImageForEdit(raw, reference.faceCrop),
      label: "角色参考",
    });
  }
  const activeAnchors = (reference?.anchors || []).filter(anchor => anchor.enabled !== false);
  for (const anchor of activeAnchors) {
    if (!anchor.assetId) continue;
    const raw = await getChatImageFromIndexedDB(anchor.assetId);
    if (raw) referenceInputs.push({ dataUrl: raw, label: `锚点：${anchor.name || "特征"}` });
  }
  const identityId = resolveGenerationUserIdentityId(params.characterId, params.appId);
  const userIdentity = identityId ? loadUserIdentities().find(identity => identity.id === identityId) : undefined;
  const userReference = identityId ? settings.userReferences?.[identityId] : undefined;
  const includeUserReference = params.includeUserReferenceImage ?? descriptionNeedsUserReference(description, userIdentity?.name);
  if (includeUserReference && userReference?.assetId && userReference.enabled !== false) {
    const raw = await getChatImageFromIndexedDB(userReference.assetId);
    if (raw) referenceInputs.push({
      dataUrl: await normalizeReferenceImageForEdit(raw, userReference.faceCrop),
      label: `用户参考：${userIdentity?.name || "用户"}`,
    });
  }
  throwIfAborted(params.signal);
  const referenceImageDataUrl = await combineReferenceImagesForEdit(referenceInputs);
  throwIfAborted(params.signal);
  const characterPrompt = reference?.featurePrompt?.trim() || "";
  const anchorsPrompt = activeAnchors
    .filter(anchor => anchor.description.trim())
    .map(anchor => `${anchor.name || "特征"}：${anchor.description.trim()}`)
    .join("；");
  const userPrompt = includeUserReference && userIdentity
    ? `【用户身份参考】${userIdentity.name}${userIdentity.gender && userIdentity.gender !== "保密" ? `，${userIdentity.gender}` : ""}。合照中必须保持用户与角色是两个不同人物，不得混脸。`
    : "";
  const prompt = mergePrompt(
    [
      description,
      characterPrompt ? `【角色固定外观】${characterPrompt}` : "",
      anchorsPrompt ? `【角色特定锚点】${anchorsPrompt}` : "",
      userPrompt,
      referenceInputs.length > 1 ? `【参考图说明】上传的是分格参考板，共 ${referenceInputs.length} 格；请按每格下方标签分别还原人物或物件，不要把不同格的身份混合。` : "",
    ].filter(Boolean).join("\n\n"),
    openaiSettings.extraPrompt,
  );

  // MiniMax 走原生协议：先浏览器直连（订阅 Key 不出浏览器），跨域被拦才回落 Netlify 函数。
  const data = isMiniMax
    ? await generateImageMiniMax({ settings: openaiSettings, prompt, referenceImageDataUrl, signal: params.signal })
    : openaiSettings.requestMode === "direct"
      ? await generateImageDirect({ settings: openaiSettings, prompt, referenceImageDataUrl, signal: params.signal })
      : await generateImageViaServerOrProxy({ settings: openaiSettings, prompt, referenceImageDataUrl, signal: params.signal });

  throwIfAborted(params.signal);
  const mimeType = data.mimeType || "image/png";
  const blob = base64ToBlob(data.b64, mimeType);
  throwIfAborted(params.signal);
  const mediaRef = await storeMediaBlob(blob, mimeType, "image");
  throwIfAborted(params.signal);
  return {
    mediaRef,
    dataUrl: `data:${mimeType};base64,${data.b64}`,
    blob,
    mimeType,
    prompt,
    usedReferenceImage: Boolean(referenceImageDataUrl),
    revisedPrompt: data.revisedPrompt,
  };
}

export function generatedImageFilename(description: string, mimeType = "image/png"): string {
  const safe = description
    .replace(/\s+/g, "-")
    .replace(/[^\u4e00-\u9fa5A-Za-z0-9_-]+/g, "")
    .slice(0, 28) || "generated-image";
  return `${safe}.${imageExtension(mimeType)}`;
}

export function hasCharacterReferenceImage(characterId?: string): boolean {
  if (!characterId) return false;
  const settings = loadImageGenerationSettings();
  const ref = settings.characterReferences?.[characterId];
  return Boolean(ref?.assetId);
}

export function hasUserReferenceImage(characterId?: string, appId = "chat"): boolean {
  const identityId = resolveGenerationUserIdentityId(characterId, appId);
  if (!identityId) return false;
  const ref = loadImageGenerationSettings().userReferences?.[identityId];
  return Boolean(ref?.assetId && ref.enabled !== false);
}
