use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;
use tauri::{AppHandle, Manager, State};
use tokio::sync::Mutex;
use url::Url;

const CONFIG_FILE: &str = "studio.json";
const DATABASE_FILE: &str = "mfq-server.sqlite3";
const LOG_FILE: &str = "mfq-server.log";
const PID_FILE: &str = "mfq-server.pid";
const MODEL_DIRECTORY: &str = "models";
const CREDENTIAL_SERVICE: &str = "MFQ Studio";
const CREDENTIAL_ACCOUNT: &str = "mfq-server-api-key";

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
    local_service_port: u16,
}

impl Default for StudioConfig {
    fn default() -> Self {
        Self {
            mode: RuntimeMode::Local,
            remote_url: "http://127.0.0.1:8090".into(),
            local_service_port: 8090,
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

#[derive(Debug, Deserialize)]
struct RegisteredModel {
    name: String,
}

#[derive(Debug, Deserialize)]
struct RegisteredModelList {
    data: Vec<RegisteredModel>,
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
    if config.local_service_port == 0 {
        return Err("local_service_port must be greater than zero".into());
    }
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

async fn is_mfq_server(client: &reqwest::Client, base_url: &str) -> bool {
    let response = match client.get(format!("{base_url}/health")).send().await {
        Ok(response) if response.status().is_success() => response,
        _ => return false,
    };
    let payload = match response.json::<serde_json::Value>().await {
        Ok(payload) => payload,
        Err(_) => return false,
    };
    payload.get("service").and_then(|value| value.as_str()) == Some("mfq-server")
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

fn packaged_executable(app: &AppHandle, names: &[&str]) -> Option<PathBuf> {
    let mut directories = Vec::new();
    if let Ok(current) = std::env::current_exe() {
        if let Some(parent) = current.parent() {
            directories.push(parent.to_path_buf());
        }
    }
    if let Ok(resource) = app.path().resource_dir() {
        directories.push(resource);
    }
    for directory in directories {
        for name in names {
            let candidate = directory.join(name);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

fn mfq_program(app: &AppHandle) -> Result<(PathBuf, bool), String> {
    #[cfg(windows)]
    let bundled_names = ["mfq.exe", "mfq.cmd", "mfq"];
    #[cfg(not(windows))]
    let bundled_names = ["mfq"];
    if let Ok(resource_dir) = app.path().resource_dir() {
        #[cfg(windows)]
        let bundled_cli = resource_dir.join("mfq-cli").join("mfq-cli.exe");
        #[cfg(not(windows))]
        let bundled_cli = resource_dir.join("mfq-cli").join("mfq-cli");
        if bundled_cli.is_file() {
            return Ok((bundled_cli, false));
        }
    }
    if let Some(path) = packaged_executable(app, &bundled_names) {
        return Ok((path, false));
    }
    if let Some(path) = executable_on_path("mfq") {
        return Ok((path, false));
    }
    let python_name = if cfg!(windows) { "python" } else { "python3" };
    executable_on_path(python_name)
        .map(|path| (path, true))
        .ok_or_else(|| "MFQ is not bundled and neither mfq nor Python is on PATH".into())
}

fn packaged_runtime(app: &AppHandle) -> Option<PathBuf> {
    #[cfg(target_os = "macos")]
    let names = ["mfq-decode-metal"];
    #[cfg(target_os = "windows")]
    let names = ["mfq-decode.exe"];
    #[cfg(all(unix, not(target_os = "macos")))]
    let names = ["mfq-decode"];
    packaged_executable(app, &names)
}

fn packaged_resource(app: &AppHandle, name: &str) -> Option<PathBuf> {
    let resource_dir = app.path().resource_dir().ok()?;
    let candidate = resource_dir.join(name);
    candidate.is_file().then_some(candidate)
}

#[cfg(target_os = "macos")]
fn packaged_framework(name: &str) -> Option<PathBuf> {
    let executable = std::env::current_exe().ok()?;
    let contents = executable.parent()?.parent()?;
    let candidate = contents.join("Frameworks").join(name);
    candidate.is_file().then_some(candidate)
}

async fn status_for(app: &AppHandle, config: StudioConfig) -> Result<StudioStatus, String> {
    let service_url = service_url(&config);
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
        .map_err(|error| error.to_string())?;
    let reachable = is_mfq_server(&client, &service_url).await;
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
    if is_mfq_server(&client, &service_url).await {
        return status_for(app, config).await;
    }

    let data_dir = app_data_dir(app)?;
    let model_dir = data_dir.join(MODEL_DIRECTORY);
    fs::create_dir_all(&model_dir).map_err(|error| error.to_string())?;
    let (program, python_module) = mfq_program(app)?;
    let stdout = open_log(&data_dir.join(LOG_FILE))?;
    let stderr = stdout.try_clone().map_err(|error| error.to_string())?;
    let mut command = Command::new(program);
    if python_module {
        command.arg("-m").arg("mfq.cli");
    }
    command
        .arg("serve")
        .arg("--no-web-ui")
        .arg("--db")
        .arg(data_dir.join(DATABASE_FILE))
        .arg("--model-dir")
        .arg(&model_dir)
        .arg("--work-dir")
        .arg(&data_dir)
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(config.local_service_port.to_string());
    if let Some(runtime) = packaged_runtime(app) {
        command.arg("--running-executable").arg(runtime);
    }
    #[cfg(windows)]
    if let Ok(resource_dir) = app.path().resource_dir() {
        // The CUDA, PyTorch, and MSVC DLLs are bundled as resources rather
        // than alongside the signed Studio executable.  Put that directory
        // first in the worker's inherited DLL search path.
        let mut paths = vec![resource_dir];
        if let Some(existing) = std::env::var_os("PATH") {
            paths.extend(std::env::split_paths(&existing));
        }
        if let Ok(path) = std::env::join_paths(paths) {
            command.env("PATH", path);
        }
    }
    if let Some(metallib) = packaged_resource(app, "mlx.metallib") {
        command.env("MFQ_MLX_METALLIB", metallib);
    }
    #[cfg(target_os = "macos")]
    if let Some(video_library) = packaged_framework("libmfq_avfoundation_video.dylib") {
        command.env("MFQ_AVFOUNDATION_VIDEO_LIBRARY", video_library);
    }
    command
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr))
        .current_dir(&data_dir);
    configure_detached(&mut command);
    let mut child = command
        .spawn()
        .map_err(|error| format!("failed to start MFQ Server: {error}"))?;
    fs::write(data_dir.join(PID_FILE), format!("{}\n", child.id()))
        .map_err(|error| error.to_string())?;

    for _ in 0..3600 {
        if is_mfq_server(&client, &service_url).await {
            return status_for(app, config).await;
        }
        if let Some(status) = child.try_wait().map_err(|error| error.to_string())? {
            return Err(format!(
                "MFQ Server exited during startup with {status}; see {LOG_FILE}"
            ));
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    Err(format!(
        "MFQ Server did not become ready within 30 minutes; see {LOG_FILE}"
    ))
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

#[tauri::command]
async fn studio_select_model_directory(app: AppHandle) -> Result<Option<Vec<String>>, String> {
    let config = load_config(&app)?;
    if !matches!(config.mode, RuntimeMode::Local) {
        return Err("the native directory picker is available only for the local server".into());
    }
    let selected = rfd::AsyncFileDialog::new()
        .set_title("Select a folder containing MFQ models")
        .pick_folder()
        .await;
    let Some(directory) = selected else {
        return Ok(None);
    };
    let path = directory
        .path()
        .to_str()
        .ok_or_else(|| "selected model directory is not valid UTF-8".to_string())?;
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(300))
        .build()
        .map_err(|error| error.to_string())?;
    let response = client
        .post(format!(
            "{}/api/v1/models/directories/register",
            local_service_url(&config)
        ))
        .json(&serde_json::json!({"path": path}))
        .send()
        .await
        .map_err(|error| format!("failed to register model directory: {error}"))?;
    if !response.status().is_success() {
        let status = response.status();
        let detail = response.text().await.unwrap_or_default();
        return Err(format!("model directory registration failed ({status}): {detail}"));
    }
    let registered = response
        .json::<RegisteredModelList>()
        .await
        .map_err(|error| format!("invalid model registration response: {error}"))?;
    Ok(Some(
        registered.data.into_iter().map(|model| model.name).collect(),
    ))
}

#[tauri::command]
fn studio_credential_get() -> Result<String, String> {
    let entry = keyring::Entry::new(CREDENTIAL_SERVICE, CREDENTIAL_ACCOUNT)
        .map_err(|error| error.to_string())?;
    match entry.get_password() {
        Ok(value) => Ok(value),
        Err(keyring::Error::NoEntry) => Ok(String::new()),
        Err(error) => Err(error.to_string()),
    }
}

#[tauri::command]
fn studio_credential_set(token: String) -> Result<(), String> {
    let entry = keyring::Entry::new(CREDENTIAL_SERVICE, CREDENTIAL_ACCOUNT)
        .map_err(|error| error.to_string())?;
    let token = token.trim();
    if token.is_empty() {
        match entry.delete_credential() {
            Ok(()) | Err(keyring::Error::NoEntry) => Ok(()),
            Err(error) => Err(error.to_string()),
        }
    } else {
        entry.set_password(token).map_err(|error| error.to_string())
    }
}

fn main() {
    tauri::Builder::default()
        .manage(StudioState::default())
        .invoke_handler(tauri::generate_handler![
            studio_status,
            studio_configure,
            studio_start_local,
            studio_select_model_directory,
            studio_credential_get,
            studio_credential_set
        ])
        .run(tauri::generate_context!())
        .expect("MFQ Studio failed");
}
