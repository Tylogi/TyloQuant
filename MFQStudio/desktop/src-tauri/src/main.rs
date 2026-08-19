use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::time::Duration;
use tauri::{AppHandle, Manager};
use url::Url;

const CONFIG_FILE: &str = "studio.json";
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

fn service_url(config: &StudioConfig) -> String {
    match config.mode {
        RuntimeMode::Local => format!("http://127.0.0.1:{}", config.local_service_port),
        RuntimeMode::Remote => config.remote_url.clone(),
    }
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

async fn status_for(config: StudioConfig) -> Result<StudioStatus, String> {
    let service_url = service_url(&config);
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
        .map_err(|error| error.to_string())?;
    let reachable = is_mfq_server(&client, &service_url).await;
    Ok(StudioStatus {
        config,
        service_url,
        reachable,
        managed_pid: None,
    })
}

#[tauri::command]
async fn studio_status(app: AppHandle) -> Result<StudioStatus, String> {
    status_for(load_config(&app)?).await
}

#[tauri::command]
async fn studio_configure(app: AppHandle, config: StudioConfig) -> Result<StudioStatus, String> {
    let config = validate_config(config)?;
    save_config(&app, &config)?;
    status_for(config).await
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
        .invoke_handler(tauri::generate_handler![
            studio_status,
            studio_configure,
            studio_credential_get,
            studio_credential_set
        ])
        .run(tauri::generate_context!())
        .expect("MFQ Studio failed");
}
