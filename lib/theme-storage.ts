import { DEFAULT_THEME_PROFILE, normalizeThemeProfile, type ThemeAssetType, type ThemeProfile } from "@/lib/theme-types";
import { kvGet, kvSet, kvRemove, registerKvMigration } from "./kv-db";
import { openIndexedDbAtLeast } from "./idb-open";
import { themePresetUsesAsset } from "./theme-preset-storage";

export const THEME_PROFILE_STORAGE_KEY = "ai_phone_theme_profile_v1";
registerKvMigration(THEME_PROFILE_STORAGE_KEY);

const THEME_DB_NAME = "ai_phone_theme_db_v1";
const THEME_DB_VERSION = 2;
const THEME_ASSET_STORE = "assets";
const IMAGE_MAX_DIMENSION = 2200;
const IMAGE_TARGET_SIZE_BYTES = 2 * 1024 * 1024;
/**
 * 动图（GIF）不经 canvas 重编码，会原样落库，因此单独限制体积。
 * dataUrl 存储比原文件再大约 1/3，超过这个阈值应让用户换图而不是静默拍平成静帧。
 */
export const ANIMATED_ASSET_MAX_BYTES = 3 * 1024 * 1024;

/** 动图体积超限：调用方应提示用户换一张更小的图。 */
export class AnimatedAssetTooLargeError extends Error {
  constructor() {
    super("动图体积超出限制");
    this.name = "AnimatedAssetTooLargeError";
  }
}

/** 该 blob 是否是必须保留原始编码的动图格式。 */
export function isAnimatedImageBlob(blob: Blob): boolean {
  return (blob.type || "").toLowerCase() === "image/gif";
}

/** 上传前预检：返回给用户看的错误文案，通过则返回 null。 */
export function checkAnimatedAssetSize(blob: Blob): string | null {
  if (!isAnimatedImageBlob(blob)) return null;
  if (blob.size <= ANIMATED_ASSET_MAX_BYTES) return null;
  const limitMb = Math.round(ANIMATED_ASSET_MAX_BYTES / (1024 * 1024));
  return `GIF 需小于 ${limitMb}MB（当前 ${(blob.size / (1024 * 1024)).toFixed(1)}MB），请换一张`;
}

/**
 * 本地图片数据库打不开（被禁用、被占用、结构缺失且无法修复）。
 * 以前这里会静默返回 null，导致上传"看起来完全没反应"，所以必须显式抛错。
 */
export class AssetStorageUnavailableError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "AssetStorageUnavailableError";
  }
}

function errorText(error: unknown): string {
  if (error instanceof Error) return error.message || error.name;
  if (typeof error === "string" && error) return error;
  return "未知错误";
}

/** 把上传异常翻译成给用户看的文案。 */
export function describeAssetSaveError(error: unknown, fallback = "上传失败，请重试"): string {
  if (error instanceof AnimatedAssetTooLargeError) {
    const limitMb = Math.round(ANIMATED_ASSET_MAX_BYTES / (1024 * 1024));
    return `GIF 需小于 ${limitMb}MB，请换一张`;
  }
  if (error instanceof AssetStorageUnavailableError) {
    return error.message;
  }
  if (typeof DOMException !== "undefined" && error instanceof DOMException) {
    if (error.name === "QuotaExceededError") {
      return "浏览器存储空间不足，请清理浏览器数据或换一张更小的图片";
    }
    if (error.name === "NotFoundError") {
      return "本地图片数据库结构异常（缺少 assets 表）";
    }
    if (error.name === "InvalidStateError") {
      return "本地图片数据库被其它标签页占用，请关掉本站其它页面后重试";
    }
    if (error.name === "VersionError") {
      return "本地图片数据库版本冲突，请刷新页面后重试";
    }
  }
  const detail = errorText(error);
  return detail && detail !== "未知错误" ? `${fallback}（${detail}）` : fallback;
}

export type ThemeAssetRecord = {
  id: string;
  type: ThemeAssetType;
  mimeType: string;
  dataUrl: string;
  updatedAt: string;
};

function hasBrowserApi(): boolean {
  return typeof window !== "undefined" && typeof indexedDB !== "undefined";
}

function hasLocalStorage(): boolean {
  return typeof window !== "undefined" && typeof window.localStorage !== "undefined";
}

function runRequest<T>(request: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

function runTransactionDone(transaction: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    transaction.oncomplete = () => resolve();
    transaction.onerror = () => reject(transaction.error);
    transaction.onabort = () => reject(transaction.error);
  });
}

/**
 * 打开本地图片库，确保 assets 表存在。
 *
 * 以前这里失败会静默返回 null，于是"保存成功"但其实什么都没写进去，
 * 表现就是上传参考图点了没反应。现在失败一律抛错，让调用方能提示用户。
 *
 * 另外处理一种隐蔽的坏状态：备份/恢复会把库版本抬高，而升级回调只在版本
 * 变化时执行——如果版本被抬到 2 以上、assets 表却没建出来，光 open 是修不好的，
 * 必须再抬一次版本把表补出来。
 */
async function openThemeDb(): Promise<IDBDatabase> {
  if (!hasBrowserApi()) {
    throw new AssetStorageUnavailableError("当前浏览器不支持 IndexedDB，无法保存图片");
  }

  const ensureStore = (database: IDBDatabase) => {
    if (!database.objectStoreNames.contains(THEME_ASSET_STORE)) {
      database.createObjectStore(THEME_ASSET_STORE, { keyPath: "id" });
    }
  };

  let database: IDBDatabase;
  try {
    // Open at >= THEME_DB_VERSION: a backup restore may have bumped the stored
    // version higher, and opening at a fixed lower version would throw a VersionError.
    database = await openIndexedDbAtLeast(THEME_DB_NAME, THEME_DB_VERSION, ensureStore);
  } catch (error) {
    throw new AssetStorageUnavailableError(`无法打开本地图片数据库：${errorText(error)}`);
  }

  if (!database.objectStoreNames.contains(THEME_ASSET_STORE)) {
    const repairVersion = database.version + 1;
    database.close();
    try {
      database = await openIndexedDbAtLeast(THEME_DB_NAME, repairVersion, ensureStore);
    } catch (error) {
      throw new AssetStorageUnavailableError(`本地图片数据库缺少 ${THEME_ASSET_STORE} 表且修复失败：${errorText(error)}`);
    }
  }

  if (!database.objectStoreNames.contains(THEME_ASSET_STORE)) {
    const version = database.version;
    database.close();
    throw new AssetStorageUnavailableError(`本地图片数据库（v${version}）缺少 ${THEME_ASSET_STORE} 表，无法保存图片`);
  }

  database.onversionchange = () => database.close();
  return database;
}

function blobToDataUrl(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(reader.error);
    reader.onload = () => resolve(String(reader.result ?? ""));
    reader.readAsDataURL(blob);
  });
}

function dataUrlToMimeType(dataUrl: string): string {
  const match = /^data:([^;]+);base64,/.exec(dataUrl);
  return match ? match[1] : "application/octet-stream";
}

function loadImageFromBlob(blob: Blob): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(blob);
    const image = new Image();
    image.onload = () => {
      URL.revokeObjectURL(url);
      resolve(image);
    };
    image.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("Failed to decode image"));
    };
    image.src = url;
  });
}

function canvasToBlob(canvas: HTMLCanvasElement, mimeType: string, quality?: number): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => {
        if (!blob) {
          reject(new Error("Failed to encode image"));
          return;
        }
        resolve(blob);
      },
      mimeType,
      quality
    );
  });
}

/**
 * 读文件头判断真实图片格式。
 * 浏览器给的 blob.type 来自扩展名——用户把 JPEG 改名成 .png 时会得到 "image/png"，
 * 结果照片被当成 PNG 重编码，体积直接膨胀好几倍。这里以实际字节为准。
 */
async function detectImageMimeType(blob: Blob): Promise<string | null> {
  try {
    const head = new Uint8Array(await blob.slice(0, 12).arrayBuffer());
    if (head.length < 4) {
      return null;
    }
    if (head[0] === 0xff && head[1] === 0xd8 && head[2] === 0xff) return "image/jpeg";
    if (head[0] === 0x89 && head[1] === 0x50 && head[2] === 0x4e && head[3] === 0x47) return "image/png";
    if (head[0] === 0x47 && head[1] === 0x49 && head[2] === 0x46 && head[3] === 0x38) return "image/gif";
    if (head[0] === 0x52 && head[1] === 0x49 && head[2] === 0x46 && head[3] === 0x46) return "image/webp";
    if (head[0] === 0x42 && head[1] === 0x4d) return "image/bmp";
    return null;
  } catch {
    return null;
  }
}

async function normalizeRasterBlob(blob: Blob, sourceMime: string): Promise<Blob> {
  const image = await loadImageFromBlob(blob);
  const largestSide = Math.max(image.naturalWidth, image.naturalHeight);
  const scale = largestSide > IMAGE_MAX_DIMENSION ? IMAGE_MAX_DIMENSION / largestSide : 1;
  let width = Math.max(1, Math.round(image.naturalWidth * scale));
  let height = Math.max(1, Math.round(image.naturalHeight * scale));

  const canvas = document.createElement("canvas");
  const draw = (): boolean => {
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext("2d");
    if (!context) {
      return false;
    }
    context.drawImage(image, 0, 0, width, height);
    return true;
  };

  if (!draw()) {
    return blob;
  }

  if (sourceMime === "image/jpeg" || sourceMime === "image/jpg") {
    let quality = 0.9;
    let encoded = await canvasToBlob(canvas, "image/jpeg", quality);
    while (encoded.size > IMAGE_TARGET_SIZE_BYTES && quality > 0.5) {
      quality = Number((quality - 0.1).toFixed(2));
      encoded = await canvasToBlob(canvas, "image/jpeg", quality);
    }
    return encoded;
  }

  if (sourceMime === "image/webp") {
    let quality = 0.9;
    let encoded = await canvasToBlob(canvas, "image/webp", quality);
    while (encoded.size > IMAGE_TARGET_SIZE_BYTES && quality > 0.5) {
      quality = Number((quality - 0.1).toFixed(2));
      encoded = await canvasToBlob(canvas, "image/webp", quality);
    }
    return encoded;
  }

  // PNG 存照片很容易膨胀到好几 MB（一张 1600x1600 的照片能到 2.5MB+），
  // 这里逐级缩小，保证落库体积可控，不至于把浏览器存储撑爆。
  let encoded = await canvasToBlob(canvas, "image/png");
  while (encoded.size > IMAGE_TARGET_SIZE_BYTES && Math.max(width, height) > 640) {
    width = Math.max(1, Math.round(width * 0.85));
    height = Math.max(1, Math.round(height * 0.85));
    if (!draw()) {
      break;
    }
    encoded = await canvasToBlob(canvas, "image/png");
  }
  return encoded;
}

async function normalizeIncomingBlob(blob: Blob): Promise<Blob> {
  const declared = (blob.type || "").toLowerCase();
  if (declared === "image/svg+xml") {
    return blob;
  }
  const detected = await detectImageMimeType(blob);
  const mime = detected || declared;
  if (mime === "image/svg+xml") {
    return blob;
  }
  if (mime === "image/gif") {
    // GIF 必须原样保留：canvas 只能取到第一帧，重编码等于把动图拍平成静态图。
    // 代价是无法压缩，所以改为体积超限时报错，交给调用方提示用户。
    if (blob.size > ANIMATED_ASSET_MAX_BYTES) {
      throw new AnimatedAssetTooLargeError();
    }
    return blob;
  }
  if (!mime.startsWith("image/")) {
    return blob;
  }
  return normalizeRasterBlob(blob, mime);
}

function createThemeAssetId(type: ThemeAssetType): string {
  const uuid = typeof crypto !== "undefined" && "randomUUID" in crypto ? crypto.randomUUID() : `${Date.now()}_${Math.random()}`;
  return `${type}_${uuid}`;
}

async function saveAssetRecord(record: ThemeAssetRecord): Promise<void> {
  const database = await openThemeDb();

  try {
    const transaction = database.transaction(THEME_ASSET_STORE, "readwrite");
    const store = transaction.objectStore(THEME_ASSET_STORE);
    store.put(record);
    await runTransactionDone(transaction);
  } finally {
    database.close();
  }
}

async function readAssetRecord(id: string): Promise<ThemeAssetRecord | null> {
  let database: IDBDatabase;
  try {
    database = await openThemeDb();
  } catch {
    // 读取失败不应该让整个界面崩掉，调用方拿不到预览而已。
    return null;
  }

  try {
    const transaction = database.transaction(THEME_ASSET_STORE, "readonly");
    const store = transaction.objectStore(THEME_ASSET_STORE);
    const result = await runRequest(store.get(id));
    await runTransactionDone(transaction);
    return (result as ThemeAssetRecord | undefined) ?? null;
  } catch {
    return null;
  } finally {
    database.close();
  }
}

export async function getThemeAssetDataUrl(id: string): Promise<string | null> {
  const record = await readAssetRecord(id);
  return record?.dataUrl ?? null;
}

export async function getThemeAssetMap(ids: string[]): Promise<Record<string, string>> {
  const map: Record<string, string> = {};
  const uniqueIds = Array.from(new Set(ids.filter(Boolean)));
  const storedIds = uniqueIds.filter((id) => !/^(?:https?:|data:|blob:)/i.test(id));
  uniqueIds.forEach((id) => {
    if (/^(?:https?:|data:|blob:)/i.test(id)) map[id] = id;
  });
  const rows = await Promise.all(storedIds.map((id) => readAssetRecord(id)));
  rows.forEach((row) => {
    if (!row) {
      return;
    }
    map[row.id] = row.dataUrl;
  });
  return map;
}

export async function readThemeAssetRecords(ids: string[]): Promise<ThemeAssetRecord[]> {
  const uniqueIds = Array.from(new Set(ids.filter(Boolean)));
  const rows = await Promise.all(uniqueIds.map((id) => readAssetRecord(id)));
  return rows.filter((row): row is ThemeAssetRecord => Boolean(row));
}

export async function writeThemeAssetRecords(records: ThemeAssetRecord[]): Promise<void> {
  const uniqueRecords = new Map<string, ThemeAssetRecord>();
  records.forEach((record) => {
    if (!record.id || !record.dataUrl) {
      return;
    }
    uniqueRecords.set(record.id, {
      ...record,
      updatedAt: record.updatedAt || new Date().toISOString()
    });
  });

  await Promise.all(Array.from(uniqueRecords.values()).map((record) => saveAssetRecord(record)));
}

export async function deleteThemeAsset(id: string): Promise<void> {
  // 本地主题预设只保存资源引用；只要仍有预设使用该资源，就不能物理删除，
  // 否则用户切回预设时会出现壁纸、图标或字体丢失。
  if (themePresetUsesAsset(id)) {
    return;
  }
  const database = await openThemeDb().catch(() => null);
  if (!database) {
    return;
  }
  try {
    const transaction = database.transaction(THEME_ASSET_STORE, "readwrite");
    transaction.objectStore(THEME_ASSET_STORE).delete(id);
    await runTransactionDone(transaction);
  } catch {
    // 删除失败不影响主流程，资产残留最多占一点空间。
  } finally {
    database.close();
  }
}

export async function saveThemeAssetFromBlob(
  blob: Blob,
  type: ThemeAssetType,
  providedId?: string
): Promise<string> {
  const normalized = await normalizeIncomingBlob(blob);
  const dataUrl = await blobToDataUrl(normalized);
  const mimeType = normalized.type || dataUrlToMimeType(dataUrl);
  const id = providedId ?? createThemeAssetId(type);
  const record: ThemeAssetRecord = {
    id,
    type,
    mimeType,
    dataUrl,
    updatedAt: new Date().toISOString()
  };
  await saveAssetRecord(record);
  return id;
}

export function readThemeProfile(): ThemeProfile {
  if (!hasLocalStorage()) {
    return { ...DEFAULT_THEME_PROFILE };
  }

  const raw = kvGet(THEME_PROFILE_STORAGE_KEY);
  if (!raw) {
    return { ...DEFAULT_THEME_PROFILE };
  }

  try {
    return normalizeThemeProfile(JSON.parse(raw) as unknown);
  } catch {
    kvRemove(THEME_PROFILE_STORAGE_KEY);
    return { ...DEFAULT_THEME_PROFILE };
  }
}

export function writeThemeProfile(profile: ThemeProfile): ThemeProfile {
  const normalized = normalizeThemeProfile({
    ...profile,
    updatedAt: new Date().toISOString()
  });

  if (hasLocalStorage()) {
    kvSet(THEME_PROFILE_STORAGE_KEY, JSON.stringify(normalized));
  }

  return normalized;
}

export function collectThemeAssetIds(profile: ThemeProfile): string[] {
  const iconIds = Object.values(profile.iconSkins).filter((value): value is string => Boolean(value));
  const iconSchemeIds = profile.iconSchemes.flatMap((scheme) =>
    Object.values(scheme.iconSkins).filter((value): value is string => Boolean(value))
  );
  return [
    profile.wallpaperAssetId ?? "",
    profile.fontAssetId ?? "",
    profile.dockSkinAssetId ?? "",
    ...profile.wallpaperLibrary,
    ...iconIds,
    ...iconSchemeIds,
  ].filter(Boolean);
}
