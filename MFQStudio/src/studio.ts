export type StudioRuntimeMode = "local" | "remote";

export interface StudioConfig {
  mode: StudioRuntimeMode;
  remote_url: string;
  local_service_port: number;
}

export interface StudioStatus {
  config: StudioConfig;
  service_url: string;
  reachable: boolean;
  managed_pid: number | null;
}

interface TauriInternals {
  invoke<T>(command: string, args?: Record<string, unknown>): Promise<T>;
}

declare global {
  interface Window {
    __TAURI_INTERNALS__?: TauriInternals;
  }
}

function internals(): TauriInternals | null {
  return window.__TAURI_INTERNALS__ ?? null;
}

export function isStudio(): boolean {
  return internals() !== null;
}

export async function studioStatus(): Promise<StudioStatus | null> {
  const tauri = internals();
  return tauri ? tauri.invoke<StudioStatus>("studio_status") : null;
}

export async function configureStudio(config: StudioConfig): Promise<StudioStatus> {
  const tauri = internals();
  if (!tauri) throw new Error("MFQ Studio runtime is unavailable");
  return tauri.invoke<StudioStatus>("studio_configure", { config });
}

export async function startLocalStudio(): Promise<StudioStatus> {
  const tauri = internals();
  if (!tauri) throw new Error("MFQ Studio runtime is unavailable");
  return tauri.invoke<StudioStatus>("studio_start_local");
}

export async function selectLocalModelDirectory(): Promise<string[] | null> {
  const tauri = internals();
  if (!tauri) throw new Error("MFQ Studio runtime is unavailable");
  return tauri.invoke<string[] | null>("studio_select_model_directory");
}

export async function studioConfirm(message: string): Promise<boolean> {
  const tauri = internals();
  return tauri
    ? tauri.invoke<boolean>("studio_confirm", { message })
    : window.confirm(message);
}

export async function studioCredential(): Promise<string> {
  const tauri = internals();
  return tauri ? tauri.invoke<string>("studio_credential_get") : "";
}

export async function saveStudioCredential(token: string): Promise<void> {
  const tauri = internals();
  if (!tauri) return;
  await tauri.invoke<void>("studio_credential_set", { token });
}
