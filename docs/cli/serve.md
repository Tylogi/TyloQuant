# `mfq serve`

`mfq serve` starts the API and Web UI, with or without an initial model. It
selects the C++ executable and manages workers on private loopback ports.

## Quick start

Sync the server and backend dependencies once:

```shell
cd /path/to/MFQ
uv sync --extra daemon --extra metal
```

On a CUDA host, use `uv sync --extra daemon` instead. Native CUDA inference
does not require PyTorch or LibTorch.

Start an idle server:

```shell
uv run mfq serve
```

Open the model catalog in MFQ Studio or the Web UI to load a discovered model.
To load one model before the server becomes available, pass its existing MFQ
path:

```shell
uv run mfq serve --model /models/model.mfq
```

The server listens on `http://127.0.0.1:8090` by default. `Ctrl-C` stops the
server and its private workers. `--model` must point to an existing `.mfq`
file.

Without `--running-executable`, run through uv from the checkout that supplies
the installed `mfq` package. The first run may compile the runtime, so the
[`mfq build` requirements](build.md#backend-requirements) apply. Packaged
runtimes do not need CMake, `nvcc`, or a source tree.

## How the native runtime is selected

`mfq serve` detects the host backend and selects the runtime in this order:

1. `--running-executable` uses a prebuilt binary and skips managed-build
   lookup, compiler detection, and compilation.
2. If the recorded executable exists, MFQ uses it directly.
3. If the manifest exists but the executable is missing, MFQ rebuilds it in the
   recorded build directory with the recorded generator, build type, and CMake
   arguments.
4. If no matching manifest exists, MFQ performs a default build and creates the
   manifest.

A manifest must match the source checkout, operating system, machine
architecture, and backend. Mismatched manifests are ignored.

The manifest is always read from `<repo>/build/mfq-runtime.json`, where
`<repo>` is the checkout supplying the running `mfq` command. A default build
uses `<repo>/cpp_runtime` as its source and `<repo>/build/cpp_runtime` as its
build directory. `--build-dir` changes only the CMake build tree; the manifest
remains under `<repo>/build`.

Source installations use the managed build automatically, including a custom
`mfq build --build-dir`. Packaged distributions pass their bundled worker with
`--running-executable`.

## Application and private listeners

`--host` and `--port` control only the user-facing FastAPI/Uvicorn listener:

```shell
uv run mfq serve \
  --model /models/model.mfq \
  --host 0.0.0.0 \
  --port 9000
```

The native worker always binds to `127.0.0.1` on a dynamically reserved port.
MFQ passes requests to that private address and disables environment HTTP proxy
handling for internal connections.

`0.0.0.0` is a bind address, not a browser destination. On the same machine,
open `http://127.0.0.1:9000/`; from another machine, use the server's reachable
IP address or hostname. Any non-loopback bind can expose both the API and Web UI
to the network. The CLI does not require authentication for this, so configure
an API key before exposing the listener:

```shell
MFQ_SERVER_API_KEY='replace-with-a-secret' \
  uv run mfq serve --model /models/model.mfq --host 0.0.0.0
```

`--api-key-env` changes the name of the environment variable read by the
server; it does not accept a secret value on the command line.

When the selected environment variable is unset or empty, API authentication is
disabled. When it is set, its value is the administrator credential. HTTP
requests under `/api/` must send it as a bearer token, and realtime WebSocket
connections must authenticate as well. Static Web UI files and `/health` remain
readable.

```shell
curl -H 'Authorization: Bearer replace-with-a-secret' \
  http://127.0.0.1:8090/api/v1/models
```

To use another variable name:

```shell
MFQ_PRODUCTION_KEY='replace-with-a-secret' \
  uv run mfq serve --model /models/model.mfq --api-key-env MFQ_PRODUCTION_KEY
```

In PowerShell, set `$env:MFQ_SERVER_API_KEY = 'replace-with-a-secret'` before
running `uv run mfq serve`. MFQ Studio desktop can store this token in its
**Remote MFQ Server API key** field. The standalone browser UI currently has no
credential prompt, so use it without server authentication only on a trusted
loopback listener, or connect through the desktop application when
authentication is enabled.

## Web UI

When Web UI assets are available, open the listener root, such as
`http://127.0.0.1:8090/`. Asset selection follows this order:

1. `--no-web-ui` disables the Web UI.
2. `--web-root` uses the specified prebuilt directory.
3. `MFQ_SERVER_WEB_ROOT` supplies a prebuilt directory when the CLI option is
   absent.
4. Otherwise, MFQ looks for `MFQStudio` in the source checkout.

A prebuilt directory must contain `index.html`. MFQ checks explicit and
environment-provided directories before starting the backend or loading a
model.

In automatic source mode, MFQ rebuilds the UI when the sources are newer than
`dist/index.html`:

1. `npm ci`
2. `npm run build`

This needs Node.js/npm, write access to the checkout, and network access when
npm must download packages. Without npm, the API still starts. A failed build
or missing `dist/index.html` stops startup before model loading.

Explicit Web UI modes:

```shell
# Serve an existing Web UI build.
uv run mfq serve --model /models/model.mfq --web-root /srv/mfq-web

# Start only the API.
uv run mfq serve --model /models/model.mfq --no-web-ui
```

`--web-root` and `--no-web-ui` are mutually exclusive. When neither is given,
`MFQ_SERVER_WEB_ROOT` can select a prebuilt directory.

## Model discovery and runtime pool

`--model` loads the initial model and adds its parent directory to the catalog.
Without it, the server stays idle until Studio, the Web UI, or the model API
loads a model. Repeat `--model-dir` to add discovery roots:

```shell
uv run mfq serve \
  --model /models/default.mfq \
  --model-dir /models/team-a \
  --model-dir /models/team-b
```

If no `--model-dir` is supplied, `MFQ_SERVER_MODEL_DIRS` is read as an
OS-separated path list. `--max-runtime-instances` limits managed workers and
`--max-requests-per-runtime` limits concurrent inference requests accepted by
each worker.

Load a catalog model after startup with:

```shell
curl -X POST http://127.0.0.1:8090/api/v1/models/load \
  -H 'Content-Type: application/json' \
  -d '{"model":"model","context_size":32768}'
```

The request returns `202 Accepted` and an `operation_id` for the background
load job.

The catalog name is the filename without `.mfq`. A shard such as
`model-00001-of-00003.mfq` belongs to catalog entry `model`.

Catalog names must be unique across discovery roots. Startup and refresh fail
if two artifacts have the same name. Runtime model IDs and inference routing
use the catalog name; a separate hash tracks the artifact and profile drift.

## Storage and workspace

By default, persistent server state is stored in `./mfq-server.sqlite3`, and
tool jobs are restricted to the current working directory.

```shell
uv run mfq serve \
  --model /models/model.mfq \
  --db /var/lib/mfq/server.sqlite3 \
  --work-dir /srv/mfq-work
```

For long-running servers, place the database and work directory on writable,
persistent paths. MFQ resolves database, work, and model paths before startup.

## Runtime controls

```shell
uv run mfq serve \
  --model /models/model.mfq \
  --context-size 32768 \
  --prefill-chunk-size 4096 \
  --runtime-startup-timeout 1800
```

- `--context-size 0` leaves context sizing to the native runtime.
- `--prefill-chunk-size` is forwarded to the Metal worker; the CUDA worker does
  not receive this option.
- `--runtime-startup-timeout` covers native worker startup and model loading.

## Command options

| Option | Meaning | Default |
| --- | --- | --- |
| `--model PATH` | Optional initial MFQ model file. | None |
| `--running-executable PATH` | Prebuilt native worker used by packaged deployments. | Managed build |
| `--host HOST` | Public API bind address. | `127.0.0.1` |
| `--port PORT` | Public API port, from 1 to 65535. | `8090` |
| `--context-size N` | Native context size; `0` keeps the runtime default. | `0` |
| `--prefill-chunk-size N` | Metal prefill chunk size. | `2048` |
| `--runtime-startup-timeout SECONDS` | Time allowed for native startup. | `1800` |
| `--db PATH` | SQLite server database. | `./mfq-server.sqlite3` |
| `--web-root PATH` | Prebuilt Web UI directory. | Auto-detect |
| `--no-web-ui` | Disable Web UI discovery and building. | Off |
| `--model-dir PATH` | Add a model discovery root; repeatable. | Environment/model parent |
| `--work-dir PATH` | Workspace boundary for tool jobs. | Current directory |
| `--api-key-env NAME` | Environment variable containing the administrator API key. | `MFQ_SERVER_API_KEY` |
| `--max-runtime-instances N` | Maximum managed native workers. | `2` |
| `--max-requests-per-runtime N` | Concurrent requests per worker. | `1` |
| `--log-level LEVEL` | Uvicorn log level. | `info` |
| `--backend {auto,cuda,metal}` | Select or detect the native backend. | `auto` |

## Troubleshooting

### The model file is rejected immediately

When supplied, `--model` must be an existing file. Check the path before
debugging runtime or server startup.

### Native startup times out

Model loading is included in the startup deadline. Increase
`--runtime-startup-timeout` for large models or slower storage, then inspect the
native worker output printed by `mfq serve`.

### The public port is already in use

Choose another `--port`. The private worker port is allocated automatically and
does not need configuration.

### Web UI assets are missing

Install Node.js/npm so MFQ can build `MFQStudio`, pass a valid
`--web-root`, or use `--no-web-ui` for an API-only process.
