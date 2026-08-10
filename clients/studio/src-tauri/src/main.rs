use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;
use tauri::{AppHandle, Manager, State};
use tokio::sync::Mutex;
use url::Url;

const CONFIG_FILE: &str = "studio.json";
const DATABASE_FILE: &str = "mfqd.sqlite3";
const LOG_FILE: &str = "mfqd.log";
const PID_FILE: &str = "mfqd.pid";

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum RuntimeMode {
    Local,
    Remote,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct StudioConfig {
    mode: RuntimeMode,
    remote_url: String,
    local_backend_url: String,
    local_service_port: u16,
    mfqd_executable: Option<String>,
}

impl Default for StudioConfig {
    fn default() -> Self {
        Self {
            mode: RuntimeMode::Local,
            remote_url: "http://127.0.0.1:8090".into(),
            local_backend_url: "http://127.0.0.1:8080".into(),
            local_service_port: 8090,
            mfqd_executable: None,
        }
    }
}

#[derive(Debug, Serialize)]
struct StudioStatus {
    config: StudioConfig,
    service_url: String,
    reachable: bool,
    managed_pid: Option<u32>,
}

#[derive(Default)]
struct StudioState {
    start_lock: Mutex<()>,
}

fn app_data_dir(app: &AppHandle) -> Result<PathBuf, String> {
    let path = app
        .path()
        .app_local_data_dir()
        .map_err(|error| error.to_string())?;
    fs::create_dir_all(&path).map_err(|error| error.to_string())?;
    Ok(path)
}

fn normalized_http_url(value: &str, field: &str) -> Result<String, String> {
    let trimmed = value.trim().trim_end_matches('/');
    let parsed = Url::parse(trimmed).map_err(|error| format!("invalid {field}: {error}"))?;
    if !matches!(parsed.scheme(), "http" | "https") || parsed.host_str().is_none() {
        return Err(format!("{field} must be an HTTP or HTTPS URL"));
    }
    if !parsed.username().is_empty() || parsed.password().is_some() {
        return Err(format!("{field} must not contain credentials"));
    }
    Ok(trimmed.to_string())
}

fn validate_config(mut config: StudioConfig) -> Result<StudioConfig, String> {
    config.remote_url = normalized_http_url(&config.remote_url, "remote_url")?;
    config.local_backend_url = normalized_http_url(&config.local_backend_url, "local_backend_url")?;
    if config.local_service_port == 0 {
        return Err("local_service_port must be greater than zero".into());
    }
    config.mfqd_executable = config
        .mfqd_executable
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    Ok(config)
}

fn config_path(app: &AppHandle) -> Result<PathBuf, String> {
    Ok(app_data_dir(app)?.join(CONFIG_FILE))
}

fn load_config(app: &AppHandle) -> Result<StudioConfig, String> {
    let path = config_path(app)?;
    if !path.exists() {
        return Ok(StudioConfig::default());
    }
    let bytes = fs::read(path).map_err(|error| error.to_string())?;
    let config = serde_json::from_slice(&bytes).map_err(|error| error.to_string())?;
    validate_config(config)
}

fn save_config(app: &AppHandle, config: &StudioConfig) -> Result<(), String> {
    let path = config_path(app)?;
    let bytes = serde_json::to_vec_pretty(config).map_err(|error| error.to_string())?;
    fs::write(path, bytes).map_err(|error| error.to_string())
}

fn local_service_url(config: &StudioConfig) -> String {
    format!("http://127.0.0.1:{}", config.local_service_port)
}

fn service_url(config: &StudioConfig) -> String {
    match config.mode {
        RuntimeMode::Local => local_service_url(config),
        RuntimeMode::Remote => config.remote_url.clone(),
    }
}

fn read_pid(path: &Path) -> Option<u32> {
    fs::read_to_string(path).ok()?.trim().parse().ok()
}

async fn is_mfqd(client: &reqwest::Client, base_url: &str) -> bool {
    let response = match client.get(format!("{base_url}/health")).send().await {
        Ok(response) if response.status().is_success() => response,
        _ => return false,
    };
    let payload = match response.json::<serde_json::Value>().await {
        Ok(payload) => payload,
        Err(_) => return false,
    };
    payload.get("service").and_then(|value| value.as_str()) == Some("mfqd")
}

fn open_log(path: &Path) -> Result<File, String> {
    OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|error| error.to_string())
}

fn configure_detached(command: &mut Command) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
        const DETACHED_PROCESS: u32 = 0x0000_0008;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        command.creation_flags(CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS | CREATE_NO_WINDOW);
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
}

fn executable_on_path(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    #[cfg(windows)]
    let names = [
        format!("{name}.exe"),
        format!("{name}.cmd"),
        name.to_string(),
    ];
    #[cfg(not(windows))]
    let names = [name.to_string()];
    for directory in std::env::split_paths(&path) {
        for candidate in &names {
            let path = directory.join(candidate);
            if path.is_file() {
                return Some(path);
            }
        }
    }
    None
}

fn mfqd_program(app: &AppHandle, config: &StudioConfig) -> Result<(PathBuf, bool), String> {
    if let Some(value) = &config.mfqd_executable {
        let path = PathBuf::from(value);
        let is_python = path
            .file_stem()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.to_ascii_lowercase().starts_with("python"));
        return Ok((path, is_python));
    }
    let executable_name = if cfg!(windows) { "mfqd.exe" } else { "mfqd" };
    if let Ok(current) = std::env::current_exe() {
        if let Some(parent) = current.parent() {
            let adjacent = parent.join(executable_name);
            if adjacent.is_file() {
                return Ok((adjacent, false));
            }
        }
    }
    if let Ok(resource) = app.path().resource_dir() {
        let bundled = resource.join(executable_name);
        if bundled.is_file() {
            return Ok((bundled, false));
        }
    }
    if let Some(path) = executable_on_path("mfqd") {
        return Ok((path, false));
    }
    let python_name = if cfg!(windows) { "python" } else { "python3" };
    executable_on_path(python_name)
        .map(|path| (path, true))
        .ok_or_else(|| "MFQd is not bundled and neither mfqd nor Python is on PATH".into())
}

async fn status_for(app: &AppHandle, config: StudioConfig) -> Result<StudioStatus, String> {
    let service_url = service_url(&config);
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
        .map_err(|error| error.to_string())?;
    let reachable = is_mfqd(&client, &service_url).await;
    let managed_pid = match config.mode {
        RuntimeMode::Local if reachable => read_pid(&app_data_dir(app)?.join(PID_FILE)),
        _ => None,
    };
    Ok(StudioStatus {
        config,
        service_url,
        reachable,
        managed_pid,
    })
}

async fn start_local(app: &AppHandle, config: StudioConfig) -> Result<StudioStatus, String> {
    let service_url = local_service_url(&config);
    let client = reqwest::Client::builder()
        .timeout(Duration::from_millis(500))
        .build()
        .map_err(|error| error.to_string())?;
    if is_mfqd(&client, &service_url).await {
        return status_for(app, config).await;
    }

    let data_dir = app_data_dir(app)?;
    let (program, python_module) = mfqd_program(app, &config)?;
    let stdout = open_log(&data_dir.join(LOG_FILE))?;
    let stderr = stdout.try_clone().map_err(|error| error.to_string())?;
    let mut command = Command::new(program);
    if python_module {
        command.arg("-m").arg("mfqd.cli");
    }
    command
        .arg("--backend-url")
        .arg(&config.local_backend_url)
        .arg("--db")
        .arg(data_dir.join(DATABASE_FILE))
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(config.local_service_port.to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr))
        .current_dir(&data_dir);
    configure_detached(&mut command);
    let mut child = command
        .spawn()
        .map_err(|error| format!("failed to start MFQd: {error}"))?;
    fs::write(data_dir.join(PID_FILE), format!("{}\n", child.id()))
        .map_err(|error| error.to_string())?;

    for _ in 0..100 {
        if is_mfqd(&client, &service_url).await {
            return status_for(app, config).await;
        }
        if let Some(status) = child.try_wait().map_err(|error| error.to_string())? {
            return Err(format!("MFQd exited during startup with {status}"));
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    Err("MFQd did not become ready within 10 seconds; see mfqd.log".into())
}

#[tauri::command]
async fn studio_status(app: AppHandle) -> Result<StudioStatus, String> {
    status_for(&app, load_config(&app)?).await
}

#[tauri::command]
async fn studio_configure(
    app: AppHandle,
    state: State<'_, StudioState>,
    config: StudioConfig,
) -> Result<StudioStatus, String> {
    let config = validate_config(config)?;
    save_config(&app, &config)?;
    match config.mode {
        RuntimeMode::Remote => status_for(&app, config).await,
        RuntimeMode::Local => {
            let _guard = state.start_lock.lock().await;
            start_local(&app, config).await
        }
    }
}

#[tauri::command]
async fn studio_start_local(
    app: AppHandle,
    state: State<'_, StudioState>,
) -> Result<StudioStatus, String> {
    let mut config = load_config(&app)?;
    config.mode = RuntimeMode::Local;
    save_config(&app, &config)?;
    let _guard = state.start_lock.lock().await;
    start_local(&app, config).await
}

fn main() {
    tauri::Builder::default()
        .manage(StudioState::default())
        .invoke_handler(tauri::generate_handler![
            studio_status,
            studio_configure,
            studio_start_local
        ])
        .run(tauri::generate_context!())
        .expect("MFQ Studio failed");
}
