# SageAttention

Building from source.

- Forked from: https://github.com/woct0rdho/SageAttention
- Original source: https://github.com/thu-ml/SageAttention

## Build

```bash
source /path/to/your/venv/bin/activate
./build.sh
```

`./build.sh` compiles for Ampere + Ada (sm80/86/89) by default. Other options: `./build.sh full` (adds Hopper + Blackwell), `./build.sh clean`, `./build.sh verify`. Requires `VIRTUAL_ENV` to be set so the install lands in the right venv.
