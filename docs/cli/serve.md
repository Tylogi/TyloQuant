# `mfq serve`

`mfq serve` starts the MFQ API and Web UI. It may start with an initial model or
with an empty model catalog. MFQ owns every native worker lifecycle: it selects
the C++ executable, runs workers on private loopback ports, and stops them with
the application server.

## Quick start

Sync the server and backend dependencies once:

```shell
cd /path/to/MFQ
uv sync --extra daemon --extra metal
```

On a CUDA host, use `uv sync --extra daemon --extra train` instead. Start with
an initial model:

```shell
uv run mfq serve
```

Or start an idle server and load a discovered model later through MFQ Studio or
the model API:

Open the model catalog in MFQ Studio or the Web UI to load a discovered model.
Pass `--model /models/model.mfq` only when one model should be loaded before the
server becomes available.

The public server listens on `http://127.0.0.1:8090` by default. Stop it with
`Ctrl-C`; MFQ then stops every private native worker as part of server shutdown.
When supplied, `--model` must point to an existing `.mfq` file.

Without `--running-executable`, run the command through uv in the source checkout
that contains the native runtime sources. MFQ derives that checkout from the
installed Python package, not from the current working directory. This mode may
compile the runtime on first use, so the backend requirements in
[`mfq build`](build.md#backend-requirements) still apply. A packaged prebuilt
runtime does not require CMake, `nvcc`, or the source tree.

## How the native runtime is selected

`mfq serve` detects the host backend and selects the runtime in this order:

1. `--running-executable` uses that prebuilt binary directly. This is intended
   for packaged releases and skips managed-build lookup, compiler detection,
   and compilation.
2. If the recorded executable exists, MFQ uses it directly.
3. If the manifest exists but the executable is missing, MFQ rebuilds it in the
   recorded build directory with the recorded generator, build type, and CMake
   arguments.
4. If no matching manifest exists, MFQ performs a default build and creates the
   manifest.

A manifest is accepted only for the same source checkout, operating system,
machine architecture, and backend. This prevents a stale path from another
checkout or host from being launched.

The manifest is always read from `<repo>/build/mfq-runtime.json`, where
`<repo>` is the checkout supplying the running `mfq` command. A default build
uses `<repo>/cpp_runtime` as its source and `<repo>/build/cpp_runtime` as its
build directory. `--build-dir` changes only the CMake build tree; the manifest
remains under `<repo>/build`.

Normal source installations do not pass the native executable to `mfq serve`.
A custom directory selected earlier with `mfq build --build-dir` is resolved
automatically. Self-contained distributions use `--running-executable` to bind
the packaged server to its bundled native worker without compiling or consulting
a source-tree build manifest.

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

A prebuilt directory must contain `index.html` at its root. MFQ validates
explicit and environment-provided directories before detecting or building the
native backend and before loading the model. In automatic source mode, if the
sources are newer than `dist/index.html`, `mfq serve` runs `npm ci` followed by
`npm run build`. This requires a compatible Node.js/npm, write access to the
source tree, and possibly network access for dependencies. If npm is
unavailable, MFQ continues with the API only. If npm starts but the build
command fails or does not produce `dist/index.html`, `mfq serve` stops with an
error before loading the model.

Choose one of the explicit modes when needed:

```shell
# Serve an existing Web UI build.
uv run mfq serve --model /models/model.mfq --web-root /srv/mfq-web

# Start only the API.
uv run mfq serve --model /models/model.mfq --no-web-ui
```

`--web-root` and `--no-web-ui` are mutually exclusive. When neither is given,
`MFQ_SERVER_WEB_ROOT` can select a prebuilt directory.

## Model discovery and runtime pool

When present, the model passed through `--model` is loaded by the initial private
worker and its parent directory is added to the model catalog. Without it, the
server remains idle until a model is loaded through Studio, the Web UI, or the
model-management API. Add discovery roots by repeating `--model-dir`:

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

The request returns `202 Accepted`; model loading continues as a background
job whose identifier is returned as `operation_id`.

The catalog name is the MFQ filename without its `.mfq` suffix. Split files
such as `model-00001-of-00003.mfq` use `model` as the catalog name. Catalog
names must be unique across every configured discovery root; startup and
catalog refresh fail explicitly if two different MFQ artifacts have the same
name. Runtime model IDs, worker identities, and inference routing use this
catalog name. The hashed artifact ID remains separate and is used only to
track the exact artifact and detect profile drift.

## Storage and workspace

By default, persistent server state is stored in `./mfq-server.sqlite3`, and
tool jobs are restricted to the current working directory.

```shell
uv run mfq serve \
  --model /models/model.mfq \
  --db /var/lib/mfq/server.sqlite3 \
  --work-dir /srv/mfq-work
```

Use dedicated writable paths for long-running deployments. The database path,
work directory, and model paths are resolved to absolute paths before the
server starts.

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
